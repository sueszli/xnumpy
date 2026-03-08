from __future__ import annotations

import math
import tempfile
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path

from exo import compile_procs as exo_compile_procs
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
from xdsl.dialects import llvm, memref
from xdsl.dialects.builtin import BoolAttr, Builtin, FloatAttr, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, UnrealizedConversionCastOp, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.llvm import FNegOp
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict

from xnumpy.patches_intrinsics import ConvertVecIntrinsic
from xnumpy.patches_llvm import BrOp, CondBrOp, ExtendedConvertMemRefToPtr, FCmpOp, RewriteMemRefTypes, SelectOp
from xnumpy.patches_llvmlite import emit_assembly


def _is_mutated(name: str, body: list) -> bool:
    # check if a variable is assigned to or reduced into in the body
    def check(stmt: object) -> bool:
        match stmt:
            case LoopIR.Assign() | LoopIR.Reduce():
                return repr(stmt.name) == name
            case LoopIR.For():
                return _is_mutated(name, stmt.body)
            case LoopIR.If():
                return _is_mutated(name, stmt.body) or _is_mutated(name, stmt.orelse)
        return False

    return any(check(stmt) for stmt in body)


def _window_access(access: object, expr_fn: Callable[[object], SSAValue]) -> SSAValue:
    match access:
        case LoopIR.Point():
            return expr_fn(access.pt)
        case LoopIR.Interval():
            return expr_fn(access.lo)
        case _:
            assert False


def _coerce_arg(
    arg_val: SSAValue,
    callee_arg: object,
    callee_body: list,
    type_fn: Callable[[object, StringAttr | None], Attribute],
    emit_fn: Callable[[Operation], SSAValue],
) -> SSAValue:
    # reconcile mlir type and shape mismatches (e.g. caller has memref<8xf32>, callee expects memref<?xf32>) via memref.cast
    mem_space = StringAttr(callee_arg.mem.name()) if hasattr(callee_arg, "mem") else None
    callee_type = type_fn(callee_arg.type, mem_space)

    # scalars passed by reference (callee writes to them) must arrive as memref<1xT>
    scalar_passed_by_ref = not isinstance(callee_type, MemRefType) and _is_mutated(repr(callee_arg.name), callee_body)
    if scalar_passed_by_ref:
        callee_type = MemRefType(callee_type, [1], NoneAttr())

    shape_mismatch = isinstance(arg_val.type, MemRefType) and isinstance(callee_type, MemRefType) and arg_val.type != callee_type
    if shape_mismatch:
        return emit_fn(memref.CastOp.get(arg_val, callee_type))

    return arg_val


def _to_index_list(values: list[SSAValue | int], emit: Callable[[Operation], SSAValue]) -> list:
    # cast i64 ssavalues to index type, pass through static ints as-is for subviewop
    static, dynamic = split_dynamic_index_list(values, memref.DYNAMIC_INDEX)
    casted = [emit(UnrealizedConversionCastOp.get([value], [IndexType()])) for value in dynamic]
    return get_dynamic_index_list(static, casted, memref.DYNAMIC_INDEX)


class IRGenerator:
    module: ModuleOp
    builder: Builder
    symbol_table: ScopedDict[str, SSAValue] | None
    type_table: ScopedDict[str, Attribute] | None
    seen_proc_names: set[str]
    seen_extern_decls: set[str]

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        self.symbol_table = None
        self.type_table = None
        self.seen_proc_names = set()
        self.seen_extern_decls = set()

    @property
    def _syms(self) -> ScopedDict[str, SSAValue]:
        assert self.symbol_table is not None
        return self.symbol_table

    @property
    def _types(self) -> ScopedDict[str, Attribute]:
        assert self.type_table is not None
        return self.type_table

    def _emit(self, op: Operation) -> SSAValue:
        self.builder.insert(op)
        assert op.results
        return op.results[0]

    def _insert_at_module(self, op: Operation) -> None:
        Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0])).insert(op)

    @contextmanager
    def _tmp_state(self, *, inherit: bool = True):
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

    def _type(self, exo_type: object, mem_space: StringAttr | None = None) -> Attribute:
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

    def _shape(self, tensor: T.Tensor, *, emit: bool = False) -> list[int | SSAValue]:
        # extract tensor dimensions as ints (static) or SSA values (variable/computed).
        # emit=False: variable dims -> DYNAMIC_INDEX (-1), for MemRefType declarations.
        # emit=True:  variable dims -> live SSA values from symbol table, for stride/offset arithmetic.
        assert isinstance(tensor, T.Tensor)

        def from_expr(expr: object) -> int | SSAValue:
            match expr:
                case LoopIR.Const():
                    # literal (e.g. `f32[16, 16]`)
                    return expr.val
                case LoopIR.Read():
                    # variable (e.g. `f32[m, k]`)
                    return self._syms[repr(expr.name)] if emit else memref.DYNAMIC_INDEX
                case LoopIR.BinOp():
                    # computed (e.g. `f32[m+1, k*2]`)
                    return self._expr_binop(expr) if emit else memref.DYNAMIC_INDEX
                case _:
                    assert False

        return [from_expr(expr) for expr in tensor.shape()]

    def _zero_index(self) -> list[SSAValue]:
        return [self._emit(llvm.ConstantOp(IntegerAttr(0, i64), i64))]

    def _memref_load(self, memref_val: SSAValue, idx: list[SSAValue]) -> SSAValue:
        if len(idx) == 0:
            idx = self._zero_index()
        indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]
        self.builder.insert(load := memref.LoadOp.get(memref_val, indices))
        return load.res

    def _memref_store(self, value: SSAValue, memref_val: SSAValue, idx: list[SSAValue]) -> None:
        # emit memref.store with i64->index casts, handling scalar memref cases
        if len(idx) == 0:
            assert isinstance(memref_val.type, MemRefType) and memref_val.type.get_shape() == (1,)
            idx = self._zero_index()

        index_indices = [self._emit(UnrealizedConversionCastOp.get([index], [IndexType()])) for index in idx]

        # if value is a scalar memref, load it first
        if isinstance(value.type, MemRefType):
            assert value.type.get_shape() == (1,)
            value = self._memref_load(value, [])

        self.builder.insert(memref.StoreOp.get(value, memref_val, index_indices))

    #
    # expressions
    #

    def _expr_const(self, const: LoopIR.Const) -> SSAValue:
        # lower loopir literal to llvm.mlir.constant op
        mlir_type = self._type(const.type)

        if mlir_type in [f16, f32, f64]:
            attr = FloatAttr(const.val, mlir_type)
        elif mlir_type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(int(const.val)), mlir_type)
        elif mlir_type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            assert False

        return self._emit(llvm.ConstantOp(attr, mlir_type))

    def _expr_read(self, read: LoopIR.Read) -> SSAValue:
        # lower loopir read to arith/memref ops
        idx = [self._expr(expr) for expr in read.idx]
        operand = self._syms[repr(read.name)]

        # only emit a load when the operand is a memref holding scalar elements
        # (not a window/tensor pass-through, and not a type-matched scalar already)
        needs_load = isinstance(operand.type, MemRefType) and not isinstance(read.type, (T.Window, T.Tensor)) and operand.type != self._type(read.type)
        return self._memref_load(operand, idx) if needs_load else operand

    def _expr_usub(self, usub: LoopIR.USub) -> SSAValue:
        # lower unary negation to llvm.fneg (float) or 0-x llvm.sub (int)
        expr = self._expr(usub.arg)
        mlir_type = self._type(usub.type)

        if mlir_type in [f16, f32, f64]:
            return self._emit(FNegOp(expr))
        elif mlir_type in [i8, i16, i32, i64]:
            zero = self._emit(llvm.ConstantOp(IntegerAttr(0, mlir_type), mlir_type))
            return self._emit(llvm.SubOp(zero, expr))
        else:
            assert False

    @staticmethod
    def _cmp_binop(lhs: SSAValue, rhs: SSAValue, op: str, emit: Callable[[Operation], SSAValue]) -> SSAValue:
        integer_cmp_table = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
        float_cmp_table = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}
        assert lhs.type == rhs.type
        if lhs.type == i1:
            bool_ops = {"and": llvm.AndOp, "or": llvm.OrOp}
            return emit(bool_ops[op](lhs, rhs))
        if lhs.type in [i8, i16, i32, i64]:
            return emit(llvm.ICmpOp(lhs, rhs, IntegerAttr(integer_cmp_table[op], i64)))
        return emit(FCmpOp(lhs, rhs, float_cmp_table[op]))

    def _expr_binop(self, binop: LoopIR.BinOp) -> SSAValue:
        # lower binary op to typed llvm op
        mlir_type = self._type(binop.type)
        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        if mlir_type == i1:
            return self._cmp_binop(lhs, rhs, binop.op, self._emit)

        float_ops = {"+": llvm.FAddOp, "-": llvm.FSubOp, "*": llvm.FMulOp, "/": llvm.FDivOp}
        int_ops = {"+": llvm.AddOp, "-": llvm.SubOp, "*": llvm.MulOp, "/": llvm.SDivOp, "%": llvm.SRemOp}
        if mlir_type in [f16, f32, f64]:
            return self._emit(float_ops[binop.op](lhs, rhs))
        if mlir_type in [i8, i16, i32, i64]:
            return self._emit(int_ops[binop.op](lhs, rhs))
        assert False

    def _expr_window(self, window: LoopIR.WindowExpr) -> SSAValue:
        # lower window expression to memref.subview
        indices = [_window_access(access, self._expr) for access in window.idx]
        source = self._syms[repr(window.name)]
        dest_type = self._type(window.type.as_tensor, source.type.memory_space)
        output_sizes = self._shape(window.type.as_tensor, emit=True)

        offsets_idx = _to_index_list(indices, self._emit)
        sizes_idx = _to_index_list(output_sizes, self._emit)
        strides_idx = _to_index_list([1] * len(indices), self._emit)

        self.builder.insert(subview := memref.SubviewOp.get(source, offsets_idx, sizes_idx, strides_idx, dest_type))
        return subview.result

    def _expr_extern(self, extern: LoopIR.Extern) -> SSAValue:
        # lower extern function call to func.call with return value
        args = [self._expr(arg) for arg in extern.args]

        if extern.f.name() == "select":
            # select(a, b, c, d) -> (a < b) ? c : d
            cmp = self._emit(FCmpOp(args[0], args[1], "olt"))
            return self._emit(SelectOp(cmp, args[2], args[3]))

        output_type = self._type(extern.f.typecheck(extern.args))
        return self._emit(llvm.CallOp(extern.f.name(), *args, return_type=output_type))

    def _expr(self, expr: object) -> OpResult | SSAValue:
        # dispatch loopir expression node to its typed lowering method
        match expr:
            case LoopIR.Read():
                return self._expr_read(expr)
            case LoopIR.Const():
                return self._expr_const(expr)
            case LoopIR.USub():
                return self._expr_usub(expr)
            case LoopIR.BinOp():
                return self._expr_binop(expr)
            case LoopIR.WindowExpr():
                return self._expr_window(expr)
            case LoopIR.Extern():
                return self._expr_extern(expr)
            case _:
                assert False

    #
    # statements
    #

    def _stmt_assign(self, stmt: LoopIR.Assign) -> None:
        # lower assignment to memref.store
        idx = [self._expr(expr) for expr in stmt.idx]
        value = self._expr(stmt.rhs)
        memref_val = self._syms[repr(stmt.name)]
        self._memref_store(value, memref_val, idx)

    def _stmt_reduce(self, stmt: LoopIR.Reduce) -> None:
        # lower reduce to load + add + store (accumulate into buffer)
        idx = [self._expr(expr) for expr in stmt.idx]
        value = self._expr(stmt.rhs)
        memref_val = self._syms[repr(stmt.name)]

        current = self._memref_load(memref_val, idx)
        if value.type in [f16, f32, f64]:
            result = self._emit(llvm.FAddOp(current, value))
        else:
            result = self._emit(llvm.AddOp(current, value))
        self._memref_store(result, memref_val, idx)

    def _stmt_if(self, if_stmt: LoopIR.If) -> None:
        # lower if/else to cf.cond_br with true, false, and merge blocks
        cond = self._expr(if_stmt.cond)

        region = self.builder.insertion_point.block.parent_region()
        true_block = Block()
        false_block = Block()
        merge_block = Block()
        region.add_block(true_block)
        region.add_block(false_block)

        self.builder.insert(CondBrOp(cond, true_block, [], false_block, []))

        # true branch
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        for stmt in if_stmt.body:
            self._stmt(stmt)
        self.builder.insert(BrOp(merge_block))

        # false branch
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        for stmt in if_stmt.orelse:
            self._stmt(stmt)
        self.builder.insert(BrOp(merge_block))

        # continue at merge
        region.add_block(merge_block)
        self.builder = Builder(insertion_point=InsertPoint.at_end(merge_block))

    def _stmt_for(self, for_stmt: LoopIR.For) -> None:
        # lower for loop to cf.br/cond_br with header, body, and exit blocks
        lo = self._expr(for_stmt.lo)
        hi = self._expr(for_stmt.hi)
        assert lo.type == hi.type
        step = self._emit(llvm.ConstantOp(IntegerAttr(1, lo.type), lo.type))

        region = self.builder.insertion_point.block.parent_region()
        header_block = Block(arg_types=[lo.type])
        body_block = Block()
        exit_block = Block()
        region.add_block(header_block)
        region.add_block(body_block)

        # branch from current block to header with initial IV
        self.builder.insert(BrOp(header_block, lo))

        # header: condition check
        self.builder = Builder(insertion_point=InsertPoint.at_end(header_block))
        iv = header_block.args[0]
        cond = self._emit(llvm.ICmpOp(iv, hi, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64)))
        self.builder.insert(CondBrOp(cond, body_block, [], exit_block, []))

        # body: emit loop body in a child symbol scope
        with self._tmp_state():
            self.builder = Builder(insertion_point=InsertPoint.at_end(body_block))
            self.symbol_table = ScopedDict(self._syms)
            self.type_table = ScopedDict(self._types)
            self._syms[repr(for_stmt.iter)] = iv
            self._types[repr(for_stmt.iter)] = T.Index

            for stmt in for_stmt.body:
                self._stmt(stmt)

            # after body: increment IV and branch back to header
            next_iv = self._emit(llvm.AddOp(iv, step))
            self.builder.insert(BrOp(header_block, next_iv))

        # continue at exit block
        region.add_block(exit_block)
        self.builder = Builder(insertion_point=InsertPoint.at_end(exit_block))

    def _stmt_alloc(self, alloc: LoopIR.Alloc) -> None:
        # lower alloc to llvm.call @malloc (DRAM) or llvm.alloca (stack)
        mem_name = alloc.mem.name()
        mem_space = StringAttr(mem_name)
        mlir_type = self._type(alloc.type, mem_space)

        # scalar allocs: wrap as memref<1x...>
        if not isinstance(mlir_type, MemRefType):
            mlir_type = MemRefType(mlir_type, [1], NoneAttr(), mem_space)

        shape = mlir_type.get_shape()
        assert all(dim != memref.DYNAMIC_INDEX for dim in shape), "dynamic-sized allocs are not supported"
        total_elements = math.prod(shape)

        if mem_name == "DRAM":
            elem_bytes = {f16: 2, f32: 4, f64: 8, i8: 1, i16: 2, i32: 4, i64: 8}[mlir_type.element_type]
            size_val = self._emit(llvm.ConstantOp(IntegerAttr(total_elements * elem_bytes, i64), i64))  # malloc takes bytes
            raw_ptr = self._emit(llvm.CallOp("malloc", size_val, return_type=llvm.LLVMPointerType()))
        else:
            size_val = self._emit(llvm.ConstantOp(IntegerAttr(total_elements, i64), i64))  # alloca takes element count
            raw_ptr = self._emit(llvm.AllocaOp(size_val, mlir_type.element_type))

        result = self._emit(UnrealizedConversionCastOp.get(raw_ptr, mlir_type))
        self._syms[repr(alloc.name)] = result
        self._types[repr(alloc.name)] = alloc.type

    def _stmt_free(self, free: LoopIR.Free) -> None:
        # lower free to llvm.call @free (DRAM) or no-op (stack)
        is_heap_mem = free.mem.name() == "DRAM"
        if not is_heap_mem:
            return
        memref_val = self._syms[repr(free.name)]
        cast = self._emit(UnrealizedConversionCastOp.get([memref_val], [llvm.LLVMPointerType()]))
        self.builder.insert(llvm.CallOp("free", cast))

    def _stmt_window(self, stmt: LoopIR.WindowStmt) -> None:
        # lower window statement to subview and bind result in symbol/type tables
        result = self._expr_window(stmt.rhs)
        self._syms[repr(stmt.name)] = result
        self._types[repr(stmt.name)] = stmt.rhs.type.as_tensor

    def _stmt_call(self, call: LoopIR.Call) -> None:
        # lower call to func.call. emit extern decl for intrinsics, recurse for procs
        args = [self._expr(arg) for arg in call.args]

        if call.f.instr is None:
            self._procedure(call.f)
            assert len(call.args) == len(call.f.args)
            args = [_coerce_arg(arg_val, callee_arg, call.f.body, self._type, self._emit) for arg_val, callee_arg in zip(args, call.f.args)]
        elif call.f.name not in self.seen_extern_decls:
            self.seen_extern_decls.add(call.f.name)
            input_types = [SSAValue.get(arg).type for arg in args]
            self._insert_at_module(llvm.FuncOp(call.f.name, llvm.LLVMFunctionType(input_types, llvm.LLVMVoidType()), llvm.LinkageAttr("external")))

        self.builder.insert(llvm.CallOp(call.f.name, *args))

    def _stmt(self, stmt: object) -> None:
        # dispatch loopir statement node to its typed lowering method
        match stmt:
            case LoopIR.Assign():
                self._stmt_assign(stmt)
            case LoopIR.Reduce():
                self._stmt_reduce(stmt)
            case LoopIR.WriteConfig():
                raise NotImplementedError()
            case LoopIR.Pass():
                pass
            case LoopIR.If():
                self._stmt_if(stmt)
            case LoopIR.For():
                self._stmt_for(stmt)
            case LoopIR.Alloc():
                self._stmt_alloc(stmt)
            case LoopIR.Free():
                self._stmt_free(stmt)
            case LoopIR.Call():
                self._stmt_call(stmt)
            case LoopIR.WindowStmt():
                self._stmt_window(stmt)
            case _:
                assert False

    def _procedure(self, procedure: LoopIR.proc) -> None:
        # lower loopir proc to llvm.func
        if procedure.name in self.seen_proc_names:
            return
        self.seen_proc_names.add(procedure.name)

        # build func signature: map each arg to its mlir type, wrapping mutated scalars in memref<1x>
        input_types = []
        for arg in procedure.args:
            mem = StringAttr(arg.mem.name()) if hasattr(arg, "mem") else None
            mlir_type = self._type(arg.type, mem)
            if not isinstance(mlir_type, MemRefType) and _is_mutated(repr(arg.name), procedure.body):
                mlir_type = MemRefType(mlir_type, [1], NoneAttr())
            input_types.append(mlir_type)
        func_type = llvm.LLVMFunctionType(input_types, llvm.LLVMVoidType())

        with self._tmp_state(inherit=False):
            block = Block(arg_types=input_types)
            func_region = Region(block)
            self.builder = Builder(insertion_point=InsertPoint.at_end(block))

            self.symbol_table = ScopedDict(local_scope={repr(arg.name): val for arg, val in zip(procedure.args, block.args)})
            self.type_table = ScopedDict(local_scope={repr(arg.name): arg.type for arg in procedure.args})

            for stmt in procedure.body:
                self._stmt(stmt)

            self.builder.insert(llvm.ReturnOp())

        self._insert_at_module(llvm.FuncOp(procedure.name, func_type, linkage=llvm.LinkageAttr("external"), body=func_region))

    def generate(self, procs: list[LoopIR.proc]) -> ModuleOp:
        for proc in procs:
            self._procedure(proc)
        # declare external malloc/free for DRAM alloc/free lowering
        self._insert_at_module(llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()), llvm.LinkageAttr("external")))
        self._insert_at_module(llvm.FuncOp("free", llvm.LLVMFunctionType([llvm.LLVMPointerType()]), llvm.LinkageAttr("external")))
        return self.module


@cache
def _context() -> Context:
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(llvm.LLVM)
    ctx.load_op(FCmpOp)
    ctx.load_op(SelectOp)
    ctx.load_op(BrOp)
    ctx.load_op(CondBrOp)
    ctx.load_dialect(memref.MemRef)
    return ctx


def _transform(analyzed_procs: list) -> ModuleOp:
    ctx = _context()

    module = IRGenerator().generate(analyzed_procs)

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    # full lowering to llvm dialect
    _rewrite = lambda patterns: PatternRewriteWalker(GreedyRewritePatternApplier(patterns)).rewrite_module(module)
    ExtendedConvertMemRefToPtr().apply(ctx, module)  # memref.{load,store,subview,cast} -> llvm
    _rewrite([RewriteMemRefTypes()])  # MemRefType -> llvm.ptr on all values
    _rewrite([ConvertVecIntrinsic()])  # vec_*/neon_* calls -> llvm/vector ops
    ReconcileUnrealizedCastsPass().apply(ctx, module)  # fold paired unrealized casts
    module.verify()

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    return module


def compile_procs(library: Procedure | Sequence[Procedure]) -> ModuleOp:
    if isinstance(library, Procedure):
        library = [library]
    compilable = [proc._loopir_proc for proc in library if not proc.is_instr()]
    all_procs = sorted(find_all_subprocs(compilable), key=lambda proc: proc.name)
    unique_procs = list({proc.name: proc for proc in all_procs if proc.instr is None}.values())

    def exo_analyze(proc: LoopIR.proc) -> LoopIR.proc:
        proc = ParallelAnalysis().run(proc)
        proc = PrecisionAnalysis().run(proc)
        proc = WindowAnalysis().apply_proc(proc)
        return MemoryAnalysis().run(proc)

    analyzed_procs = [exo_analyze(proc) for proc in unique_procs]
    return _transform(analyzed_procs)


def main():
    parser = ArgumentParser(description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, help="Exo source file (.py)")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    fmt = parser.add_mutually_exclusive_group(required=True)
    fmt.add_argument("--mlir", action="store_const", dest="fmt", const="mlir", help="Emit xDSL MLIR text")
    fmt.add_argument("--asm", action="store_const", dest="fmt", const="asm", help="Emit assembly via llvmlite")
    fmt.add_argument("--c", action="store_const", dest="fmt", const="c", help="Emit Exo's native C codegen")
    args = parser.parse_args()

    src = Path(args.source)
    assert src.is_file() and src.suffix == ".py"

    library = get_procs_from_module(load_user_code(src))
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    match args.fmt:
        case "c":
            d = Path(tempfile.mkdtemp())
            exo_compile_procs(library, d, "o.c", "o.h")
            output = (d / "o.c").read_text()
        case "mlir":
            output = str(compile_procs(library))
        case "asm":
            output = str(emit_assembly(compile_procs(library)))
        case _:
            assert False

    if not args.output or args.output == "-":
        print(output)
        return

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(output)
