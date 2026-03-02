from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cache, reduce
from pathlib import Path

from exo.API import Procedure
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.parallel_analysis import ParallelAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import LoopIR, T
from exo.main import get_procs_from_module, load_user_code
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, memref, ptr, scf
from xdsl.dialects.arith import AddfOp, AddiOp, AndIOp, CmpfOp, CmpiOp, ConstantOp, DivfOp, DivSIOp, FastMathFlagsAttr, MulfOp, MuliOp, NegfOp, OrIOp, RemSIOp, SubfOp, SubiOp
from xdsl.dialects.builtin import BoolAttr, Builtin, FloatAttr, FunctionType, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, UnrealizedConversionCastOp, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.scf import ForOp, IfOp, YieldOp
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_ptr_to_llvm import ConvertPtrToLLVMPass
from xdsl.transforms.convert_ptr_type_offsets import ConvertPtrTypeOffsetsPass
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict

from xdsl_exo.patches import ExtendedConvertMemRefToPtr, LLVMIntrinsics
from xdsl_exo.rewrites import ConvertAllocFreeToLLVM, ConvertExternPass, ConvertIntrinsicsPass, LowerMemRefTypesPass


class IRGenerator:
    module: ModuleOp
    builder: Builder
    symbol_table: ScopedDict[str, SSAValue] | None
    type_table: ScopedDict[str, Attribute] | None
    seen_procs: set[str]

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        self.symbol_table = None
        self.type_table = None
        self.seen_procs = set()

    def _emit(self, op: Operation) -> SSAValue:
        self.builder.insert(op)
        return op.results[0]

    @contextmanager
    def _tmp_state(self, *, inherit=True):
        # save and restore builder/symbol/type state across nested scopes
        parent_builder = self.builder
        parent_symbol_table = self.symbol_table
        parent_type_table = self.type_table
        if not inherit:
            self.symbol_table = ScopedDict[str, SSAValue]()
            self.type_table = ScopedDict[str, Attribute]()
        try:
            yield
        finally:
            self.builder = parent_builder
            self.symbol_table = parent_symbol_table
            self.type_table = parent_type_table

    def _type(self, exo_type, mem_space: StringAttr | None = None) -> Attribute:
        # map exo type (t.f32, t.tensor, etc.) to mlir type (f32, memref, etc.)
        match exo_type:
            case SSAValue():
                return exo_type.type
            case T.F16():
                return f16
            case T.F32() | T.Num():
                return f32
            case T.F64():
                return f64
            case T.INT8() | T.UINT8():
                return i8
            case T.UINT16():
                return i16
            case T.INT32():
                return i32
            case T.Index() | T.Size() | T.Int():
                return i64
            case T.Bool():
                return i1
            case T.Tensor():
                assert mem_space is not None
                inner = self._type(exo_type.type)
                assert inner in {f16, f32, f64, i8, i16, i32}
                shape = self._shape(exo_type)
                return MemRefType(inner, shape, NoneAttr(), mem_space)
            case _:
                assert False

    def _shape(self, tensor, *, dynamic=False) -> list:
        # extract tensor dimensions as ints (static) or ssa values (dynamic)
        assert isinstance(tensor, T.Tensor)

        def from_expr(expr):
            match expr:
                case LoopIR.Const():
                    # literal (e.g. `f32[16, 16]`)
                    return expr.val
                case LoopIR.Read():
                    # variable (e.g. `f32[m, k]`)
                    return self.symbol_table[repr(expr.name)] if dynamic else -1
                case LoopIR.BinOp():
                    # computed (e.g. `f32[m+1, k*2]`)
                    return self._binop_expr(expr) if dynamic else -1
                case _:
                    assert False

        return [from_expr(expr) for expr in tensor.shape()]

    def _const_expr(self, const):
        # lower loopir literal to arith.constant op
        assert isinstance(const, LoopIR.Const)
        type = self._type(const.type)

        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(int(const.val)), type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            assert False

        return self._emit(ConstantOp(attr, type))

    def _memref_load(self, memref_val, idx):
        if len(idx) == 0:
            idx = [self._emit(arith.ConstantOp(IntegerAttr(0, i64)))]
        indices = [self._emit(arith.IndexCastOp(i, IndexType())) for i in idx]
        self.builder.insert(load := memref.LoadOp.get(memref_val, indices))
        return load.res

    def _memref_store(self, value, memref_val, idx):
        # emit memref.store with i64→index casts, handling scalar memref cases
        if len(idx) == 0:
            assert isinstance(memref_val.type, MemRefType) and memref_val.type.get_shape() == (1,)
            idx = [self._emit(arith.ConstantOp(IntegerAttr(0, i64)))]

        index_indices = [self._emit(arith.IndexCastOp(i, IndexType())) for i in idx]

        # if value is a scalar memref, load it first
        if isinstance(value.type, MemRefType):
            assert value.type.get_shape() == (1,)
            zero_idx = self._emit(arith.ConstantOp(IntegerAttr(0, IndexType())))
            self.builder.insert(load := memref.LoadOp.get(value, [zero_idx]))
            value = load.res

        self.builder.insert(memref.StoreOp.get(value, memref_val, index_indices))

    def _read_expr(self, read):
        # lower loopir read to arith/memref ops
        assert isinstance(read, LoopIR.Read)
        idx = [self._expr(e) for e in read.idx]
        operand = self.symbol_table[repr(read.name)]

        if not isinstance(operand.type, MemRefType):
            return operand
        if operand.type == self._type(read.type):
            return operand
        return self._memref_load(operand, idx)

    def _usub_expr(self, usub):
        # lower unary negation to negf (float) or 0-x subi (int)
        assert isinstance(usub, LoopIR.USub)
        expr = self._expr(usub.arg)
        type = self._type(usub.type)

        if type in [f16, f32, f64]:
            return self._emit(NegfOp(expr))
        elif type in [i8, i16, i32, i64]:
            zero = self._emit(ConstantOp(IntegerAttr(0, type)))
            return self._emit(SubiOp(zero, expr, result_type=type))
        else:
            assert False

    def _binop_expr(self, binop):
        # lower binary op to typed arith op; delegates to _binop_expr_cmp for i1
        assert isinstance(binop, LoopIR.BinOp)
        type = self._type(binop.type)
        if type == i1:
            return self._binop_expr_cmp(binop)

        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        float_ops = {"+": AddfOp, "-": SubfOp, "*": MulfOp, "/": DivfOp}
        int_ops = {"+": AddiOp, "-": SubiOp, "*": MuliOp, "/": DivSIOp, "%": RemSIOp}

        if type in [f16, f32, f64]:
            return self._emit(float_ops[binop.op](lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none")))
        elif type in [i8, i16, i32, i64]:
            return self._emit(int_ops[binop.op](lhs, rhs, result_type=type))
        else:
            assert False

    def _binop_expr_cmp(self, binop):
        # lower comparison/logical binop to cmpi, cmpf, andi, or ori
        assert isinstance(binop, LoopIR.BinOp)
        bool_ops = {"and": AndIOp, "or": OrIOp}
        integer_cmp_table = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
        float_cmp_table = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}

        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        assert lhs.type == rhs.type, f"cannot compare {lhs.type} and {rhs.type} with operator '{binop.op}'"

        if lhs.type == i1:
            return self._emit(bool_ops[binop.op](lhs, rhs))
        elif lhs.type in [i8, i16, i32, i64]:
            return self._emit(CmpiOp(lhs, rhs, integer_cmp_table[binop.op]))
        else:
            return self._emit(CmpfOp(lhs, rhs, float_cmp_table[binop.op]))

    def _window_expr(self, window):
        # lower window expression to stride/offset computation + memref.subview
        assert isinstance(window, LoopIR.WindowExpr)

        def w_access(w):
            match w:
                case LoopIR.Point():
                    return self._expr(w.pt)
                case LoopIR.Interval():
                    return self._expr(w.lo)
                case _:
                    assert False

        def compute_strides(sizes):
            # stride[i] = product(sizes[i+1:]), computed right-to-left
            strides: list[SSAValue | int] = [1]
            for size in reversed(sizes):
                last = strides[0]
                if isinstance(last, int) and isinstance(size, int):
                    strides.insert(0, last * size)
                    continue
                if isinstance(last, int):
                    last = self._emit(ConstantOp(IntegerAttr(last, i64)))
                if isinstance(size, int):
                    size = self._emit(ConstantOp(IntegerAttr(size, i64)))
                strides.insert(0, self._emit(MuliOp(operand1=last, operand2=size)))
            return strides

        def compute_offsets(indices, strides):
            # offset[i] = index[i] * stride[i]
            offsets: list[SSAValue] = []
            for idx, stride in zip(indices, strides):
                if isinstance(stride, int):
                    stride = self._emit(ConstantOp(IntegerAttr(stride, i64)))
                offsets.append(self._emit(MuliOp(operand1=idx, operand2=stride)))
            return offsets

        def to_index(values):
            # cast i64 ssavalues to index type, pass through static ints as-is for subviewop
            static, dynamic = split_dynamic_index_list(values, memref.DYNAMIC_INDEX)
            casted = [self._emit(arith.IndexCastOp(v, IndexType())) for v in dynamic]
            return get_dynamic_index_list(static, casted, memref.DYNAMIC_INDEX)

        idx = [w_access(w) for w in window.idx]
        input = self.symbol_table[repr(window.name)]
        dest_type = self._type(window.type.as_tensor, input.type.memory_space)
        input_sizes = self._shape(self.type_table[repr(window.name)], dynamic=True)
        output_sizes = self._shape(window.type.as_tensor, dynamic=True)

        # compute strides/offsets in i64, then cast to index for subview
        strides = compute_strides(input_sizes)
        offsets = compute_offsets(idx, strides)
        strides_idx = to_index(strides)
        offsets_idx = to_index(offsets)
        sizes_idx = to_index(output_sizes)

        self.builder.insert(subview := memref.SubviewOp.get(input, offsets_idx, sizes_idx, strides_idx, dest_type))
        return subview.result

    def _extern_expr(self, extern):
        # lower extern function call to func.call with return value
        assert isinstance(extern, LoopIR.Extern)
        output_type = self._type(extern.f.typecheck(extern.args))
        args = [self._expr(e) for e in extern.args]
        return self._emit(CallOp(extern.f.name(), args, [output_type]))

    def _expr(self, expr) -> OpResult | SSAValue:
        # dispatch loopir expression node to its typed lowering method
        match expr:
            case LoopIR.Read():
                return self._read_expr(expr)
            case LoopIR.Const():
                return self._const_expr(expr)
            case LoopIR.USub():
                return self._usub_expr(expr)
            case LoopIR.BinOp():
                return self._binop_expr(expr)
            case LoopIR.WindowExpr():
                return self._window_expr(expr)
            case LoopIR.Extern():
                return self._extern_expr(expr)
            case _:
                assert False

    def _store_stmt(self, stmt):
        # lower assignment to memref.store
        assert isinstance(stmt, LoopIR.Assign)
        idx = [self._expr(e) for e in stmt.idx]
        value = self._expr(stmt.rhs)
        memref_val = self.symbol_table[repr(stmt.name)]
        self._memref_store(value, memref_val, idx)

    def _reduce_stmt(self, stmt):
        # lower reduce to load + add + store (accumulate into buffer)
        assert isinstance(stmt, LoopIR.Reduce)
        idx = [self._expr(e) for e in stmt.idx]
        value = self._expr(stmt.rhs)
        memref_val = self.symbol_table[repr(stmt.name)]

        current = self._memref_load(memref_val, idx)
        if value.type in [f16, f32, f64]:
            add_op = AddfOp(current, value, result_type=value.type, flags=FastMathFlagsAttr("none"))
        else:
            add_op = AddiOp(current, value, result_type=value.type)
        self.builder.insert(add_op)
        self._memref_store(add_op.result, memref_val, idx)

    def _if_stmt(self, if_stmt):
        # lower if/else to scf.if with true and false regions
        assert isinstance(if_stmt, LoopIR.If)
        cond = self._expr(if_stmt.cond)

        with self._tmp_state():
            true_block = Block()
            self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
            for s in if_stmt.body:
                self._stmt(s)
            self.builder.insert(YieldOp())

            false_block = Block()
            self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
            for s in if_stmt.orelse:
                self._stmt(s)
            self.builder.insert(YieldOp())

        self.builder.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def _for_stmt(self, for_stmt):
        # lower for loop to scf.for with lo/hi bounds and step inferred from bound types
        assert isinstance(for_stmt, LoopIR.For)
        lo = self._expr(for_stmt.lo)
        hi = self._expr(for_stmt.hi)
        assert lo.type == hi.type
        step = self._emit(ConstantOp(IntegerAttr(1, lo.type)))

        with self._tmp_state():
            loop_block = Block(arg_types=[lo.type])
            self.builder = Builder(insertion_point=InsertPoint.at_end(loop_block))
            self.symbol_table = ScopedDict(self.symbol_table)

            self.symbol_table[repr(for_stmt.iter)] = loop_block.args[0]
            self.type_table[repr(for_stmt.iter)] = T.Index

            for s in for_stmt.body:
                self._stmt(s)
            self.builder.insert(YieldOp())

        self.builder.insert(ForOp(lo, hi, step, [], Region(loop_block)))

    def _alloc_stmt(self, alloc):
        # lower alloc to memref.alloc (dram) or llvm.alloca (other)
        assert isinstance(alloc, LoopIR.Alloc)
        mem_name = alloc.mem.name()
        mem_space = StringAttr(mem_name)
        type = self._type(alloc.type, mem_space)

        # scalar allocs: wrap as memref<1x...>
        if not isinstance(type, MemRefType):
            type = MemRefType(type, [1], NoneAttr(), mem_space)

        if mem_name == "DRAM":
            self.builder.insert(op := memref.AllocOp.get(type.element_type, shape=type.shape, layout=type.layout, memory_space=mem_space))
            result = op.memref
            self.symbol_table[repr(alloc.name)] = result
            self.type_table[repr(alloc.name)] = alloc.type
            return result

        # vec_avx2 or other
        total_size = reduce(lambda x, y: x * y, type.get_shape())
        const_val = self._emit(arith.ConstantOp(IntegerAttr(total_size, i64)))
        alloc_res = self._emit(llvm.AllocaOp(const_val, type.element_type))
        result = self._emit(UnrealizedConversionCastOp.get(alloc_res, type))
        self.symbol_table[repr(alloc.name)] = result
        self.type_table[repr(alloc.name)] = alloc.type
        return result

    def _free_stmt(self, free):
        # lower free to memref.dealloc
        # memory space is already on the memref type, so standard memref.dealloc suffices
        assert isinstance(free, LoopIR.Free)
        self.builder.insert(memref.DeallocOp.get(self.symbol_table[repr(free.name)]))

    def _call_stmt(self, call):
        # lower call to func.call. emit extern decl for intrinsics, recurse for procs
        assert isinstance(call, LoopIR.Call)
        args = [self._expr(arg) for arg in call.args]

        if call.f.instr is None:
            self._procedure(call.f)
            assert len(call.args) == len(call.f.args)
        elif call.f.name not in self.seen_procs:
            self.seen_procs.add(call.f.name)
            input_types = [SSAValue.get(a).type for a in args]
            module_builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
            module_builder.insert(FuncOp.external(call.f.name, input_types, []))

        self.builder.insert(CallOp(call.f.name, args, []))

    def _stmt(self, stmt):
        # dispatch loopir statement node to its typed lowering method
        match stmt:
            case LoopIR.Assign():
                self._store_stmt(stmt)
            case LoopIR.Reduce():
                self._reduce_stmt(stmt)
            case LoopIR.WriteConfig():
                raise NotImplementedError()
            case LoopIR.Pass():
                pass
            case LoopIR.If():
                self._if_stmt(stmt)
            case LoopIR.For():
                self._for_stmt(stmt)
            case LoopIR.Alloc():
                self._alloc_stmt(stmt)
            case LoopIR.Free():
                self._free_stmt(stmt)
            case LoopIR.Call():
                self._call_stmt(stmt)
            case LoopIR.Window():
                raise NotImplementedError()
            case _:
                assert False

    @staticmethod
    def _is_mutated(name: str, body) -> bool:
        # check if a variable is assigned to or reduced into in the body
        def check(stmt):
            match stmt:
                case LoopIR.Assign() | LoopIR.Reduce():
                    return repr(stmt.name) == name
                case LoopIR.For():
                    return IRGenerator._is_mutated(name, stmt.body)
                case LoopIR.If():
                    return IRGenerator._is_mutated(name, stmt.body) or IRGenerator._is_mutated(name, stmt.orelse)
            return False

        return any(check(s) for s in body)

    def _procedure(self, procedure):
        # lower loopir proc to func.func
        assert isinstance(procedure, LoopIR.proc)
        if procedure.name in self.seen_procs:
            return
        self.seen_procs.add(procedure.name)

        if procedure.instr is not None:
            raise NotImplementedError()

        input_types = []
        for arg in procedure.args:
            mem = StringAttr(arg.mem.name()) if hasattr(arg, "mem") else None
            t = self._type(arg.type, mem)
            if not isinstance(t, MemRefType) and self._is_mutated(repr(arg.name), procedure.body):
                t = MemRefType(t, [1], NoneAttr())
            input_types.append(t)
        func_type = FunctionType.from_lists(input_types, [])

        with self._tmp_state(inherit=False):
            block = Block(arg_types=input_types)
            self.builder = Builder(insertion_point=InsertPoint.at_end(block))

            self.symbol_table = ScopedDict(local_scope={repr(a.name): b for a, b in zip(procedure.args, block.args)})
            self.type_table = ScopedDict(local_scope={repr(a.name): a.type for a in procedure.args})

            for s in procedure.body:
                self._stmt(s)

            self.builder.insert(ReturnOp())

        module_builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        module_builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate(self, procs) -> ModuleOp:
        for proc in procs:
            self._procedure(proc)
        return self.module


@cache
def _context() -> Context:
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(LLVMIntrinsics)
    ctx.load_dialect(ptr.Ptr)
    return ctx


def _transform(analyzed_procs: list) -> ModuleOp:
    ctx = _context()

    # exo loopir -> raw exo ir
    module = IRGenerator().generate(analyzed_procs)

    # partial lowering
    ConvertExternPass().apply(ctx, module)  # select → arith.cmpf + arith.select
    module.verify()

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    # full lowering to llvm dialect
    ConvertAllocFreeToLLVM().apply(ctx, module)  # VEC_AVX2 dealloc erasure + DRAM alloc→malloc, dealloc→free
    ExtendedConvertMemRefToPtr().apply(ctx, module)  # memref.{load,store,subview,cast} → ptr ops
    ConvertPtrTypeOffsetsPass().apply(ctx, module)  # ptr.TypeOffsetOp → arith.constant(sizeof)
    ConvertPtrToLLVMPass().apply(ctx, module)  # ptr.* → llvm.*
    LowerMemRefTypesPass().apply(ctx, module)  # MemRefType → LLVMPointerType
    ConvertIntrinsicsPass().apply(ctx, module)  # mm256_*/vec_* intrinsic calls → llvm/vector ops
    ConvertScfToCf().apply(ctx, module)  # scf → cf
    ReconcileUnrealizedCastsPass().apply(ctx, module)  # remove unrealized cast chains
    module.verify()

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    return module


def compile_procs(
    library: Procedure | Sequence[Procedure],  # exo funcs decorated with @proc
) -> ModuleOp:
    if isinstance(library, Procedure):
        library = [library]
    compilable = [proc._loopir_proc for proc in library if not proc.is_instr()]
    all_procs = sorted(find_all_subprocs(compilable), key=lambda x: x.name)
    unique_procs = list({p.name: p for p in all_procs}.values())

    def exo_analyze(proc):
        assert isinstance(proc, LoopIR.proc)
        proc = ParallelAnalysis().run(proc)
        proc = PrecisionAnalysis().run(proc)
        proc = WindowAnalysis().apply_proc(proc)
        return MemoryAnalysis().run(proc)

    analyzed_procs = [exo_analyze(proc) for proc in unique_procs]
    return _transform(analyzed_procs)


def main():
    parser = ArgumentParser(description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, help="Source file to compile")
    parser.add_argument("-o", "--output", help="Output file. Defaults to stdout.")
    args = parser.parse_args()

    src = Path(args.source)
    assert src.is_file() and src.suffix == ".py"

    library = get_procs_from_module(load_user_code(src))
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_procs(library)

    if not args.output or args.output == "-":
        print(module)
        return

    dst = Path(args.output)
    os.makedirs(dst.parent, exist_ok=True)
    dst.write_text(str(module))
