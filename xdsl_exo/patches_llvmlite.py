import llvmlite.binding as llvm_binding
import llvmlite.ir as ir
from xdsl.backend.llvm.convert_op import convert_op
from xdsl.backend.llvm.convert_type import convert_type as _xdsl_convert_type
from xdsl.dialects import arith, cf, func, llvm
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.dialects.llvm import LLVMVoidType
from xdsl.ir import Block, Operation, SSAValue

from xdsl_exo.patches import FCmpOp, FNegOp, SelectOp

ValMap = dict[SSAValue, ir.Value]
BlockMap = dict[Block, ir.Block]
PhiMap = dict[SSAValue, ir.PhiInstr]


def _convert_type(xdsl_type) -> ir.Type:
    if isinstance(xdsl_type, IndexType):
        return ir.IntType(64)
    if isinstance(xdsl_type, LLVMVoidType):
        return ir.VoidType()
    return _xdsl_convert_type(xdsl_type)


def _return_type(func_op: func.FuncOp | llvm.FuncOp) -> ir.Type:
    if isinstance(func_op, llvm.FuncOp):
        return _convert_type(func_op.function_type.output)
    outputs = list(func_op.function_type.outputs)
    return _convert_type(outputs[0]) if outputs else ir.VoidType()


# FCmpOp predicate string -> (operator string, is_ordered)
_CMPF_PRED: dict[str, tuple[str, bool]] = {"oeq": ("==", True), "ogt": (">", True), "oge": (">=", True), "olt": ("<", True), "ole": ("<=", True), "one": ("!=", True), "ord": ("ord", True), "ueq": ("==", False), "ugt": (">", False), "uge": (">=", False), "ult": ("<", False), "ule": ("<=", False), "une": ("!=", False), "uno": ("uno", False)}


def _emit_op(op: Operation, builder: ir.IRBuilder, block_map: BlockMap, phi_map: PhiMap, val_map: ValMap) -> None:
    match op:
        case llvm.ConstantOp() | arith.ConstantOp():
            val_map[op.result] = ir.Constant(_convert_type(op.result.type), op.value.value.data)
        case arith.MuliOp() | arith.AddiOp():
            method = "mul" if isinstance(op, arith.MuliOp) else "add"
            val_map[op.result] = getattr(builder, method)(val_map[op.lhs], val_map[op.rhs])
        case arith.IndexCastOp():
            src, dst_type = val_map[op.operands[0]], _convert_type(op.results[0].type)
            val_map[op.results[0]] = src if src.type == dst_type else builder.bitcast(src, dst_type)
        case FNegOp():
            val_map[op.res] = builder.fneg(val_map[op.arg])
        case FCmpOp():
            pred, is_ordered = _CMPF_PRED[op.predicate.data]
            cmp_fn = builder.fcmp_ordered if is_ordered else builder.fcmp_unordered
            val_map[op.res] = cmp_fn(pred, val_map[op.lhs], val_map[op.rhs])
        case SelectOp():
            val_map[op.res] = builder.select(val_map[op.cond], val_map[op.lhs], val_map[op.rhs])
        case cf.BranchOp():
            cur = builder.block
            for a, v in zip(op.successor.args, op.operands):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            builder.branch(block_map[op.successor])
        case cf.ConditionalBranchOp():
            cur = builder.block
            for a, v in zip(op.successors[0].args, op.then_arguments):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            for a, v in zip(op.successors[1].args, op.else_arguments):
                if a in phi_map:
                    phi_map[a].add_incoming(val_map[v], cur)
            builder.cbranch(val_map[op.cond], block_map[op.successors[0]], block_map[op.successors[1]])
        case func.ReturnOp() | llvm.ReturnOp():
            builder.ret(val_map[op.operands[0]]) if op.operands else builder.ret_void()
        case func.CallOp():
            callee = builder.module.get_global(op.callee.string_value())
            result = builder.call(callee, [val_map[arg] for arg in op.arguments])
            if op.res:
                val_map[op.res[0]] = result
        case _:
            convert_op(op, builder, val_map)


def _emit_func_body(func_op: func.FuncOp | llvm.FuncOp, llvm_module: ir.Module) -> None:
    ir_func = llvm_module.get_global(func_op.sym_name.data)
    mlir_blocks = list(func_op.body.blocks)

    block_map: BlockMap = {block: ir_func.append_basic_block() for block in mlir_blocks}
    phi_map: PhiMap = {}
    val_map: ValMap = dict(zip(mlir_blocks[0].args, ir_func.args))

    for mlir_block in mlir_blocks[1:]:
        for block_arg in mlir_block.args:
            phi = ir.IRBuilder(block_map[mlir_block]).phi(_convert_type(block_arg.type))
            phi_map[block_arg] = val_map[block_arg] = phi

    for mlir_block in mlir_blocks:
        builder = ir.IRBuilder(block_map[mlir_block])
        for op in mlir_block.ops:
            _emit_op(op, builder, block_map, phi_map, val_map)


def to_llvmlite(module: ModuleOp) -> ir.Module:
    llvm_module = ir.Module()
    top_level_ops = list(module.ops)

    # forward-declare all functions so call sites can resolve them regardless of order
    for op in top_level_ops:
        match op:
            case func.FuncOp() | llvm.FuncOp():
                ftype = ir.FunctionType(_return_type(op), [_convert_type(t) for t in op.function_type.inputs])
                ir.Function(llvm_module, ftype, name=op.sym_name.data)
            case _:
                assert False

    # emit bodies
    for op in top_level_ops:
        if isinstance(op, (func.FuncOp, llvm.FuncOp)) and op.body.blocks:
            _emit_func_body(op, llvm_module)

    return llvm_module


def jit_compile(ir_module: ir.Module) -> llvm_binding.ExecutionEngine:
    llvm_binding.initialize_native_target()
    llvm_binding.initialize_native_asmprinter()
    llvm_mod = llvm_binding.parse_assembly(str(ir_module))
    llvm_mod.verify()
    target_machine = llvm_binding.Target.from_default_triple().create_target_machine()
    engine = llvm_binding.create_mcjit_compiler(llvm_mod, target_machine)
    engine.finalize_object()
    engine.run_static_constructors()
    return engine
