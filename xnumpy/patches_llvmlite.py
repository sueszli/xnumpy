import ctypes
from functools import cache

import llvmlite.binding as llvm_binding
import llvmlite.ir as ir
from xdsl.backend.llvm.convert_op import convert_op as _xdsl_convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.llvm import FNegOp
from xdsl.ir import Block, Operation, SSAValue

from xnumpy.patches_llvm import BrOp, CondBrOp, FCmpOp, SelectOp

ValMap = dict[SSAValue, ir.Value]
BlockMap = dict[Block, ir.Block]
PhiMap = dict[SSAValue, ir.PhiInstr]

_FCMP_PREDICATES: dict[str, tuple[str, bool]] = {  # mlir predicate -> (op, ordered?)
    "oeq": ("==", True),
    "ogt": (">", True),
    "oge": (">=", True),
    "olt": ("<", True),
    "ole": ("<=", True),
    "one": ("!=", True),
    "ord": ("ord", True),
    "ueq": ("==", False),
    "ugt": (">", False),
    "uge": (">=", False),
    "ult": ("<", False),
    "ule": ("<=", False),
    "une": ("!=", False),
    "uno": ("uno", False),
}


def _add_phis(phi_map: PhiMap, val_map: ValMap, block_args, operands, cur_block):
    # wire block operands into phi nodes for the target block
    for a, v in zip(block_args, operands):
        if a in phi_map:
            phi_map[a].add_incoming(val_map[v], cur_block)


def _convert_op(op: Operation, builder: ir.IRBuilder, block_map: BlockMap, phi_map: PhiMap, val_map: ValMap) -> None:
    # translate one xdsl op to llvmlite ir. unmatched ops fall back to xdsl's convert_op
    match op:
        case llvm.ConstantOp():
            val_map[op.result] = ir.Constant(convert_type(op.result.type), op.value.value.data)
        case FNegOp():
            val_map[op.res] = builder.fneg(val_map[op.arg])
        case FCmpOp():
            pred, is_ordered = _FCMP_PREDICATES[op.predicate.data]
            val_map[op.res] = (builder.fcmp_ordered if is_ordered else builder.fcmp_unordered)(pred, val_map[op.lhs], val_map[op.rhs])
        case SelectOp():
            val_map[op.res] = builder.select(val_map[op.cond], val_map[op.lhs], val_map[op.rhs])
        case BrOp():
            _add_phis(phi_map, val_map, op.successor.args, op.operands, builder.block)
            builder.branch(block_map[op.successor])
        case CondBrOp():
            _add_phis(phi_map, val_map, op.successors[0].args, op.then_arguments, builder.block)
            _add_phis(phi_map, val_map, op.successors[1].args, op.else_arguments, builder.block)
            builder.cbranch(val_map[op.cond], block_map[op.successors[0]], block_map[op.successors[1]])
        case vector.BroadcastOp():
            source_val = val_map[op.source]
            vec_type = convert_type(op.vector.type)
            n_lanes = op.vector.type.get_shape()[0]
            undef = ir.Constant(vec_type, ir.Undefined)
            inserted = builder.insert_element(undef, source_val, ir.Constant(ir.IntType(32), 0))
            mask = ir.Constant(ir.VectorType(ir.IntType(32), n_lanes), [0] * n_lanes)
            val_map[op.vector] = builder.shuffle_vector(inserted, undef, mask)
        case vector.FMAOp():
            lhs = val_map[op.lhs]
            rhs = val_map[op.rhs]
            acc = val_map[op.acc]
            vec_type = convert_type(op.res.type)
            n = vec_type.count
            elem = "f32" if vec_type.element == ir.FloatType() else "f64"
            intrinsic_name = f"llvm.fma.v{n}{elem}"
            try:
                fma_fn = builder.module.get_global(intrinsic_name)
            except KeyError:
                fma_type = ir.FunctionType(vec_type, [vec_type, vec_type, vec_type])
                fma_fn = ir.Function(builder.module, fma_type, name=intrinsic_name)
            val_map[op.res] = builder.call(fma_fn, [lhs, rhs, acc])
        case _:
            _xdsl_convert_op(op, builder, val_map)


def _emit_func(func_op: llvm.FuncOp, llvm_module: ir.Module) -> None:
    # emit one xdsl func: create blocks, insert phis, translate ops
    ir_func = llvm_module.get_global(func_op.sym_name.data)
    mlir_blocks = list(func_op.body.blocks)

    block_map: BlockMap = {block: ir_func.append_basic_block() for block in mlir_blocks}
    phi_map: PhiMap = {arg: ir.IRBuilder(block_map[blk]).phi(convert_type(arg.type)) for blk in mlir_blocks[1:] for arg in blk.args}
    val_map: ValMap = dict(zip(mlir_blocks[0].args, ir_func.args)) | phi_map

    for mlir_block in mlir_blocks:
        builder = ir.IRBuilder(block_map[mlir_block])
        for op in mlir_block.ops:
            _convert_op(op, builder, block_map, phi_map, val_map)


llvm_binding.initialize_native_target()
llvm_binding.initialize_native_asmprinter()


@cache
def _create_target_machine() -> llvm_binding.TargetMachine:
    # host cpu + features never change
    target = llvm_binding.Target.from_default_triple()
    cpu = llvm_binding.get_host_cpu_name()
    features = llvm_binding.get_host_cpu_features().flatten()
    return target.create_target_machine(cpu=cpu, features=features, opt=2)


def _optimize_module(llvm_mod: llvm_binding.ModuleRef, target_machine: llvm_binding.TargetMachine) -> None:
    # run llvm -O2 pass pipeline
    pto = llvm_binding.PipelineTuningOptions()
    pto.speed_level = 2
    pb = llvm_binding.create_pass_builder(target_machine, pto)
    pm = pb.getModulePassManager()
    pm.run(llvm_mod, pb)


def _emit_repeat_wrapper(llvm_module: ir.Module, func_name: str) -> None:
    # emit `{name}_repeat(args..., count)`. calls kernel in a counted loop
    i64 = ir.IntType(64)
    kernel = llvm_module.get_global(func_name)
    orig_args = list(kernel.function_type.args)
    wrap_ftype = ir.FunctionType(ir.VoidType(), orig_args + [i64])
    wrapper = ir.Function(llvm_module, wrap_ftype, name=f"{func_name}_repeat")

    entry = wrapper.append_basic_block("entry")
    loop_bb = wrapper.append_basic_block("loop")
    exit_bb = wrapper.append_basic_block("exit")

    b = ir.IRBuilder(entry)
    count = wrapper.args[-1]
    kernel_args = list(wrapper.args[:-1])
    b.cbranch(b.icmp_unsigned(">", count, ir.Constant(i64, 0)), loop_bb, exit_bb)

    b = ir.IRBuilder(loop_bb)
    i = b.phi(i64)
    i.add_incoming(ir.Constant(i64, 0), entry)
    b.call(kernel, kernel_args)
    i_next = b.add(i, ir.Constant(i64, 1))
    i.add_incoming(i_next, loop_bb)
    b.cbranch(b.icmp_unsigned("<", i_next, count), loop_bb, exit_bb)

    ir.IRBuilder(exit_bb).ret_void()


def _lower(module: ModuleOp, *, repeat_wrappers: bool) -> tuple[llvm_binding.ModuleRef, llvm_binding.TargetMachine]:
    # xdsl module -> llvmlite ir -> parsed llvm module, optimized at -O2
    llvm_module = ir.Module()
    func_ops = list(module.ops)

    for op in func_ops:
        assert isinstance(op, llvm.FuncOp)
        ftype = ir.FunctionType(convert_type(op.function_type.output), [convert_type(t) for t in op.function_type.inputs])
        ir.Function(llvm_module, ftype, name=op.sym_name.data)

    for op in func_ops:
        if op.body.blocks:
            _emit_func(op, llvm_module)

    if repeat_wrappers:
        for op in func_ops:
            if op.body.blocks:
                _emit_repeat_wrapper(llvm_module, op.sym_name.data)

    llvm_mod = llvm_binding.parse_assembly(str(llvm_module))
    target_machine = _create_target_machine()
    _optimize_module(llvm_mod, target_machine)
    return llvm_mod, target_machine


def jit_compile(module: ModuleOp) -> dict[str, ctypes._CFuncPtr]:
    # lower + jit. returns {name: cfunc, name_repeat: cfunc} for each func with a body
    llvm_mod, target_machine = _lower(module, repeat_wrappers=True)
    engine = llvm_binding.create_mcjit_compiler(llvm_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()

    fns: dict[str, ctypes._CFuncPtr] = {}
    for op in module.ops:
        if not isinstance(op, llvm.FuncOp) or not op.body.blocks:
            continue
        name = op.sym_name.data
        n = len(op.function_type.inputs)
        fn = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * n))(engine.get_function_address(name))
        fn._engine = engine
        fns[name] = fn
        fn_r = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * (n + 1)))(engine.get_function_address(f"{name}_repeat"))
        fn_r._engine = engine
        fns[f"{name}_repeat"] = fn_r
    return fns


def emit_assembly(module: ModuleOp) -> str:
    # lower + emit native assembly text (no repeat wrappers)
    llvm_mod, target_machine = _lower(module, repeat_wrappers=False)
    return target_machine.emit_assembly(llvm_mod)
