from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import click
import llvmlite.binding
import llvmlite.ir
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.parallel_analysis import ParallelAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import LoopIR, T
from exo.main import get_procs_from_module, load_user_code
from xdsl.backend.llvm.convert_op import convert_op as _xdsl_convert_op
from xdsl.backend.llvm.convert_type import convert_type
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import llvm, memref, vector
from xdsl.dialects.builtin import BoolAttr, Builtin, DenseIntOrFPElementsAttr, FloatAttr, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, UnrealizedConversionCastOp, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.llvm import FNegOp
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict

from xnumpy.jitcall import JitFunc
from xnumpy.patches_xdsl_intrinsics import ConvertVecIntrinsic
from xnumpy.patches_xdsl_llvm import BrOp, CondBrOp, ExtendedConvertMemRefToPtr, FCmpOp, RewriteMemRefTypes, SelectOp

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


#
# generate xDSL MLIR
#


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
    def _scoped_state(self, *, inherit: bool = True):
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

    def _to_mlir_type(self, exo_type: object, mem_space: StringAttr | None = None) -> Attribute:
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
                inner = self._to_mlir_type(exo_type.type)
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

    def _expr_const(self, const: LoopIR.Const) -> SSAValue:
        # lower loopir literal to llvm.mlir.constant op
        mlir_type = self._to_mlir_type(const.type)

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
        needs_load = isinstance(operand.type, MemRefType) and not isinstance(read.type, (T.Window, T.Tensor)) and operand.type != self._to_mlir_type(read.type)
        return self._memref_load(operand, idx) if needs_load else operand

    def _expr_usub(self, usub: LoopIR.USub) -> SSAValue:
        # lower unary negation to llvm.fneg (float) or 0-x llvm.sub (int)
        expr = self._expr(usub.arg)
        mlir_type = self._to_mlir_type(usub.type)

        if mlir_type in [f16, f32, f64]:
            return self._emit(FNegOp(expr, fast_math=llvm.FastMathAttr("fast")))
        elif mlir_type in [i8, i16, i32, i64]:
            zero = self._emit(llvm.ConstantOp(IntegerAttr(0, mlir_type), mlir_type))
            return self._emit(llvm.SubOp(zero, expr))
        else:
            assert False

    @staticmethod
    def _cmp_binop(lhs: SSAValue, rhs: SSAValue, op: str, emit: Callable[[Operation], SSAValue]) -> SSAValue:
        integer_cmp_table = {"==": 0, "!=": 1, "<": 2, "<=": 3, ">": 4, ">=": 5}
        float_cmp_table = {op: pred for pred, (op, ordered) in _FCMP_PREDICATES.items() if ordered and op not in ("ord", "uno")}
        assert lhs.type == rhs.type
        if lhs.type == i1:
            bool_ops = {"and": llvm.AndOp, "or": llvm.OrOp}
            return emit(bool_ops[op](lhs, rhs))
        if lhs.type in [i8, i16, i32, i64]:
            return emit(llvm.ICmpOp(lhs, rhs, IntegerAttr(integer_cmp_table[op], i64)))
        return emit(FCmpOp(lhs, rhs, float_cmp_table[op]))

    def _expr_binop(self, binop: LoopIR.BinOp) -> SSAValue:
        # lower binary op to typed llvm op
        mlir_type = self._to_mlir_type(binop.type)
        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        if mlir_type == i1:
            return self._cmp_binop(lhs, rhs, binop.op, self._emit)

        float_ops = {"+": llvm.FAddOp, "-": llvm.FSubOp, "*": llvm.FMulOp, "/": llvm.FDivOp}
        int_ops = {"+": llvm.AddOp, "-": llvm.SubOp, "*": llvm.MulOp, "/": llvm.SDivOp, "%": llvm.SRemOp}
        if mlir_type in [f16, f32, f64]:
            return self._emit(float_ops[binop.op](lhs, rhs, fast_math=llvm.FastMathAttr("fast")))
        if mlir_type in [i8, i16, i32, i64]:
            return self._emit(int_ops[binop.op](lhs, rhs))
        assert False

    @staticmethod
    def _window_access(access: object, expr_fn: Callable[[object], SSAValue]) -> SSAValue:
        match access:
            case LoopIR.Point():
                return expr_fn(access.pt)
            case LoopIR.Interval():
                return expr_fn(access.lo)
            case _:
                assert False

    @staticmethod
    def _to_index_list(values: list[SSAValue | int], emit: Callable[[Operation], SSAValue]) -> list:
        # cast i64 ssavalues to index type, pass through static ints as-is for subviewop
        static, dynamic = split_dynamic_index_list(values, memref.DYNAMIC_INDEX)
        casted = [emit(UnrealizedConversionCastOp.get([value], [IndexType()])) for value in dynamic]
        return get_dynamic_index_list(static, casted, memref.DYNAMIC_INDEX)

    def _expr_window(self, window: LoopIR.WindowExpr) -> SSAValue:
        # lower window expression to memref.subview
        indices = [self._window_access(access, self._expr) for access in window.idx]
        source = self._syms[repr(window.name)]
        dest_type = self._to_mlir_type(window.type.as_tensor, source.type.memory_space)
        output_sizes = self._shape(window.type.as_tensor, emit=True)

        offsets_idx = self._to_index_list(indices, self._emit)
        sizes_idx = self._to_index_list(output_sizes, self._emit)
        strides_idx = self._to_index_list([1] * len(indices), self._emit)

        self.builder.insert(subview := memref.SubviewOp.get(source, offsets_idx, sizes_idx, strides_idx, dest_type))
        return subview.result

    def _expr_extern(self, extern: LoopIR.Extern) -> SSAValue:
        # lower extern function call to func.call with return value
        args = [self._expr(arg) for arg in extern.args]

        if extern.f.name() == "select":
            # select(a, b, c, d) -> (a < b) ? c : d
            cmp = self._emit(FCmpOp(args[0], args[1], "olt"))
            return self._emit(SelectOp(cmp, args[2], args[3]))

        output_type = self._to_mlir_type(extern.f.typecheck(extern.args))
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
            result = self._emit(llvm.FAddOp(current, value, fast_math=llvm.FastMathAttr("fast")))
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
        with self._scoped_state():
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
        mlir_type = self._to_mlir_type(alloc.type, mem_space)

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

    @staticmethod
    def _is_mutated(name: str, body: list) -> bool:
        # check if a variable is assigned to or reduced into in the body
        def check(stmt: object) -> bool:
            match stmt:
                case LoopIR.Assign() | LoopIR.Reduce():
                    return repr(stmt.name) == name
                case LoopIR.For():
                    return IRGenerator._is_mutated(name, stmt.body)
                case LoopIR.If():
                    return IRGenerator._is_mutated(name, stmt.body) or IRGenerator._is_mutated(name, stmt.orelse)
            return False

        return any(check(stmt) for stmt in body)

    @staticmethod
    def _coerce_arg(arg_val: SSAValue, callee_arg: object, callee_body: list, type_fn: Callable[[object, StringAttr | None], Attribute], emit_fn: Callable[[Operation], SSAValue]) -> SSAValue:
        # reconcile mlir type and shape mismatches (e.g. caller has memref<8xf32>, callee expects memref<?xf32>) via memref.cast
        mem_space = StringAttr(callee_arg.mem.name()) if hasattr(callee_arg, "mem") else None
        callee_type = type_fn(callee_arg.type, mem_space)

        # scalars passed by reference (callee writes to them) must arrive as memref<1xT>
        scalar_passed_by_ref = not isinstance(callee_type, MemRefType) and IRGenerator._is_mutated(repr(callee_arg.name), callee_body)
        if scalar_passed_by_ref:
            callee_type = MemRefType(callee_type, [1], NoneAttr())

        shape_mismatch = isinstance(arg_val.type, MemRefType) and isinstance(callee_type, MemRefType) and arg_val.type != callee_type
        if shape_mismatch:
            return emit_fn(memref.CastOp.get(arg_val, callee_type))

        return arg_val

    def _stmt_call(self, call: LoopIR.Call) -> None:
        # lower call to func.call. emit extern decl for intrinsics, recurse for procs
        args = [self._expr(arg) for arg in call.args]

        if call.f.instr is None:
            self._generate_procedure(call.f)
            assert len(call.args) == len(call.f.args)
            args = [self._coerce_arg(arg_val, callee_arg, call.f.body, self._to_mlir_type, self._emit) for arg_val, callee_arg in zip(args, call.f.args)]
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

    def _generate_procedure(self, procedure: LoopIR.proc) -> None:
        # lower loopir proc to llvm.func
        if procedure.name in self.seen_proc_names:
            return
        self.seen_proc_names.add(procedure.name)

        # build func signature: map each arg to its mlir type, wrapping mutated scalars in memref<1x>
        input_types = []
        for arg in procedure.args:
            mem = StringAttr(arg.mem.name()) if hasattr(arg, "mem") else None
            mlir_type = self._to_mlir_type(arg.type, mem)
            if not isinstance(mlir_type, MemRefType) and self._is_mutated(repr(arg.name), procedure.body):
                mlir_type = MemRefType(mlir_type, [1], NoneAttr())
            input_types.append(mlir_type)
        func_type = llvm.LLVMFunctionType(input_types, llvm.LLVMVoidType())

        with self._scoped_state(inherit=False):
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
            self._generate_procedure(proc)
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


def _lower(procs: list[LoopIR.proc]) -> ModuleOp:
    ctx = _context()

    module = IRGenerator().generate(procs)

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


def to_mlir(library: Procedure | Sequence[Procedure]) -> ModuleOp:
    # exo procedures -> xDSL MLIR (LLVM dialect)
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

    return _lower([exo_analyze(proc) for proc in unique_procs])


#
# generate llvmlite IR, then JIT compile
#


class LLVMLiteGenerator:
    @staticmethod
    def _add_phis(phi_map: dict[SSAValue, llvmlite.ir.PhiInstr], val_map: dict[SSAValue, llvmlite.ir.Value], block_args, operands, cur_block):
        # wire block operands into phi nodes for the target block
        for a, v in zip(block_args, operands):
            if a in phi_map:
                phi_map[a].add_incoming(val_map[v], cur_block)

    @staticmethod
    def _convert_op(op: Operation, builder: llvmlite.ir.IRBuilder, block_map: dict[Block, llvmlite.ir.Block], phi_map: dict[SSAValue, llvmlite.ir.PhiInstr], val_map: dict[SSAValue, llvmlite.ir.Value]) -> None:
        # translate one xdsl op to llvmlite ir. unmatched ops fall back to xdsl's convert_op
        match op:
            case llvm.ConstantOp():
                if isinstance(op.value, DenseIntOrFPElementsAttr):
                    val_map[op.result] = llvmlite.ir.Constant(convert_type(op.result.type), list(op.value.iter_values()))
                else:
                    val_map[op.result] = llvmlite.ir.Constant(convert_type(op.result.type), op.value.value.data)
            case FNegOp():
                val_map[op.res] = builder.fneg(val_map[op.arg])
            case FCmpOp():
                pred, is_ordered = _FCMP_PREDICATES[op.predicate.data]
                val_map[op.res] = (builder.fcmp_ordered if is_ordered else builder.fcmp_unordered)(pred, val_map[op.lhs], val_map[op.rhs])
            case SelectOp():
                val_map[op.res] = builder.select(val_map[op.cond], val_map[op.lhs], val_map[op.rhs])
            case BrOp():
                LLVMLiteGenerator._add_phis(phi_map, val_map, op.successor.args, op.operands, builder.block)
                builder.branch(block_map[op.successor])
            case CondBrOp():
                LLVMLiteGenerator._add_phis(phi_map, val_map, op.successors[0].args, op.then_arguments, builder.block)
                LLVMLiteGenerator._add_phis(phi_map, val_map, op.successors[1].args, op.else_arguments, builder.block)
                builder.cbranch(val_map[op.cond], block_map[op.successors[0]], block_map[op.successors[1]])
            case vector.BroadcastOp():
                source_val = val_map[op.source]
                vec_type = convert_type(op.vector.type)
                n_lanes = op.vector.type.get_shape()[0]
                undef = llvmlite.ir.Constant(vec_type, llvmlite.ir.Undefined)
                inserted = builder.insert_element(undef, source_val, llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0))
                mask = llvmlite.ir.Constant(llvmlite.ir.VectorType(llvmlite.ir.IntType(32), n_lanes), [0] * n_lanes)
                val_map[op.vector] = builder.shuffle_vector(inserted, undef, mask)
            case vector.FMAOp():
                lhs = val_map[op.lhs]
                rhs = val_map[op.rhs]
                acc = val_map[op.acc]
                vec_type = convert_type(op.res.type)
                n = vec_type.count
                elem = "f32" if vec_type.element == llvmlite.ir.FloatType() else "f64"
                intrinsic_name = f"llvm.fma.v{n}{elem}"
                try:
                    fma_fn = builder.module.get_global(intrinsic_name)
                except KeyError:
                    fma_type = llvmlite.ir.FunctionType(vec_type, [vec_type, vec_type, vec_type])
                    fma_fn = llvmlite.ir.Function(builder.module, fma_type, name=intrinsic_name)
                val_map[op.res] = builder.call(fma_fn, [lhs, rhs, acc])
            case _:
                _xdsl_convert_op(op, builder, val_map)

    @staticmethod
    def _generate_func(func_op: llvm.FuncOp, llvm_module: llvmlite.ir.Module) -> None:
        # generate one xdsl func: create blocks, insert phis, translate ops
        ir_func = llvm_module.get_global(func_op.sym_name.data)
        mlir_blocks = list(func_op.body.blocks)

        block_map: dict[Block, llvmlite.ir.Block] = {block: ir_func.append_basic_block() for block in mlir_blocks}
        phi_map: dict[SSAValue, llvmlite.ir.PhiInstr] = {arg: llvmlite.ir.IRBuilder(block_map[blk]).phi(convert_type(arg.type)) for blk in mlir_blocks[1:] for arg in blk.args}
        val_map: dict[SSAValue, llvmlite.ir.Value] = dict(zip(mlir_blocks[0].args, ir_func.args)) | phi_map

        for mlir_block in mlir_blocks:
            builder = llvmlite.ir.IRBuilder(block_map[mlir_block])
            for op in mlir_block.ops:
                LLVMLiteGenerator._convert_op(op, builder, block_map, phi_map, val_map)

    @staticmethod
    @cache
    def generate(module: ModuleOp) -> llvmlite.ir.Module:
        llvm_module = llvmlite.ir.Module()
        # enable loop vectorizer
        tm = _target_machine()
        llvm_module.triple = tm.triple
        llvm_module.data_layout = str(tm.target_data)
        func_ops = list(module.ops)

        for op in func_ops:
            assert isinstance(op, llvm.FuncOp)
            ftype = llvmlite.ir.FunctionType(convert_type(op.function_type.output), [convert_type(t) for t in op.function_type.inputs])
            fn = llvmlite.ir.Function(llvm_module, ftype, name=op.sym_name.data)

            # mark all pointer args as noalias for vectorization
            for arg in fn.args:
                if isinstance(arg.type, llvmlite.ir.PointerType):
                    arg.add_attribute("noalias")

        for op in func_ops:
            if op.body.blocks:
                LLVMLiteGenerator._generate_func(op, llvm_module)

        return llvm_module


llvmlite.binding.initialize_native_target()
llvmlite.binding.initialize_native_asmprinter()


@cache
def _target_machine() -> llvmlite.binding.TargetMachine:
    target = llvmlite.binding.Target.from_default_triple()
    cpu = llvmlite.binding.get_host_cpu_name()
    features = llvmlite.binding.get_host_cpu_features().flatten()
    return target.create_target_machine(cpu=cpu, features=features, opt=3)


@cache
def _to_llvmlite_moduleref(llvmlite_module: llvmlite.ir.Module) -> tuple[llvmlite.binding.ModuleRef, llvmlite.binding.TargetMachine]:
    # llvmlite IR -> parsed + optimized LLVM module ref
    mod_ref = llvmlite.binding.parse_assembly(str(llvmlite_module))
    tm = _target_machine()
    pto = llvmlite.binding.PipelineTuningOptions()
    pto.speed_level = 3  # O3 optimization
    pto.loop_vectorization = True  # enable loop vectorizer
    pto.slp_vectorization = True  # enable SLP vectorizer (straight-line code)
    pb = llvmlite.binding.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod_ref, pb)
    return mod_ref, tm


@cache
def _to_jit_engine(module: ModuleOp) -> llvmlite.binding.ExecutionEngine:
    # xDSL MLIR -> MCJIT execution engine (in-memory)
    mod_ref, tm = _to_llvmlite_moduleref(LLVMLiteGenerator.generate(module))
    engine = llvmlite.binding.create_mcjit_compiler(mod_ref, tm)
    engine.finalize_object()
    engine.run_static_constructors()
    return engine


@cache
def to_asm(module: ModuleOp) -> str:
    # xDSL MLIR -> native assembly text
    mod_ref, tm = _to_llvmlite_moduleref(LLVMLiteGenerator.generate(module))
    return tm.emit_assembly(mod_ref)


@cache
def _extract_jit_funcs(module: ModuleOp, engine: llvmlite.binding.ExecutionEngine) -> dict[str, object]:
    # call our custom c bridge to minimize ffi overhead
    # constraint: no native-code loop wrappers. each op pays the same per-call FFI cost as numpy for fair comparison
    fns: dict[str, object] = {}
    for op in module.ops:
        if not isinstance(op, llvm.FuncOp) or not op.body.blocks:
            continue
        name = op.sym_name.data
        fns[name] = JitFunc(engine.get_function_address(name), engine)
    return fns


def compile_jit(proc: Procedure) -> dict[str, object]:
    module = to_mlir(proc)
    engine = _to_jit_engine(module)
    return _extract_jit_funcs(module, engine)


#
# cli entry point
#


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--c", "fmt", flag_value="c", help="Output C source")
@click.option("--mlir", "fmt", flag_value="mlir", help="Output MLIR")
@click.option("--asm", "fmt", flag_value="asm", help="Output assembly")
def cli(source: Path, fmt: str | None):
    if not fmt:
        raise click.UsageError("Specify one of --c, --mlir, or --asm")
    source = get_procs_from_module(load_user_code(source))

    match fmt:
        case "c":
            tmpdir = Path(tempfile.mkdtemp())
            exo_compile_procs(source, tmpdir, "o.c", "o.h")
            text = (tmpdir / "o.c").read_text()
        case "mlir":
            text = to_mlir(source)
        case "asm":
            text = to_asm(to_mlir(source))

    click.echo(text)
