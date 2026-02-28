from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from exo.API import Procedure, Sym
from exo.backend.LoopIR_compiler import find_all_subprocs
from exo.backend.mem_analysis import MemoryAnalysis
from exo.backend.parallel_analysis import ParallelAnalysis
from exo.backend.prec_analysis import PrecisionAnalysis
from exo.backend.win_analysis import WindowAnalysis
from exo.core.LoopIR import LoopIR, T
from exo.main import get_procs_from_module, load_user_code

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.arith import AddfOp, AddiOp, AndIOp, CmpfOp, CmpiOp, ConstantOp, DivfOp, DivSIOp, FastMathFlagsAttr, MulfOp, MuliOp, NegfOp, OrIOp, RemSIOp, SubfOp, SubiOp
from xdsl.dialects.builtin import BoolAttr, Builtin, FloatAttr, FunctionType, IndexType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.memref import CastOp as MemrefCastOp
from xdsl.dialects.scf import ForOp, IfOp, YieldOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Attribute, Block, BlockArgument, OpResult, Region, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict
from xdsl_exo.dialects.exo import AllocOp, AssignOp, Exo, ExternOp, FreeOp, InstrOp, IntervalOp, ReadOp, ReduceOp, WindowOp
from xdsl_exo.dialects.extra import CastsOp, Index, LLVMIntrinsics
from xdsl_exo.platforms.avx2 import InlineAVX2Pass
from xdsl_exo.platforms.blas import InlineBLASAllocPass, InlineBLASPass
from xdsl_exo.rewrites.add_prefix import AddPrefixPass
from xdsl_exo.rewrites.convert_memref_to_llvm import ConvertMemRefToLLVM
from xdsl_exo.rewrites.convert_scalar_ref import ConvertScalarRefPass
from xdsl_exo.rewrites.inline_memory_space import InlineMemorySpacePass
from xdsl_exo.rewrites.reconcile_index_casts import ReconcileIndexCastsPass


class IRGenerator:
    module: ModuleOp
    builder: Builder

    symbol_table: ScopedDict[str, SSAValue] | None = None
    type_table: ScopedDict[str, Attribute] | None = None

    seen_procs: set[str] = set()

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))

    def _with_empty_scope(self):
        self.symbol_table = ScopedDict()
        self.type_table = ScopedDict()
        return self

    def _with_test_op(self, sym: Sym, type):
        assert self.symbol_table is not None
        op = TestOp(result_types=[self._get_type(type)])
        self.builder.insert(op)
        self.symbol_table[sym.__repr__()] = op.res[0]
        if self.type_table is not None:
            self.type_table[sym.__repr__()] = type
        return self

    #
    # symbol table
    #

    def _declare_arg(self, sym: Sym, arg: BlockArgument) -> BlockArgument:
        assert self.symbol_table is not None
        self._declare_value(sym, arg)
        return arg

    def _declare_value(self, sym: Sym, value: SSAValue) -> SSAValue:
        assert self.symbol_table is not None
        self.symbol_table[sym.__repr__()] = value
        return value

    def _get_sym(self, sym: Sym) -> SSAValue:
        assert self.symbol_table is not None
        assert sym.__repr__() in self.symbol_table, f"unknown symbol {sym.__repr__()}"
        return self.symbol_table[sym.__repr__()]

    def _declare_sym_exo_type(self, sym: Sym, type):
        assert self.type_table is not None
        self.type_table[sym.__repr__()] = type
        return type

    def _get_sym_exo_type(self, sym: Sym):
        assert self.type_table is not None
        assert sym.__repr__() in self.type_table, f"unknown symbol {sym.__repr__()}"
        return self.type_table[sym.__repr__()]

    #
    # type helpers
    #

    def _get_type(self, t, mem_space=StringAttr("DRAM")) -> Attribute:
        memref_element_types = {f16, f32, f64, i8, i16, i32}

        match t:
            case SSAValue():
                return t.type
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
                inner = self._get_type(t.type)
                assert inner in memref_element_types, f"unknown tensor inner type '{inner}'"
                shape = self._get_static_shape(t)
                return MemRefType(inner, shape, NoneAttr(), mem_space)
            case _:
                assert False, f"unknown type '{t}'"

    def _get_shape(self, type) -> tuple[list[IntegerAttr], list[SSAValue]]:
        assert isinstance(type, T.Tensor)

        dynamic_shapes = []

        def attr_from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return IntAttr(expr.val)
                case LoopIR.Read():
                    if self.symbol_table is not None:
                        dynamic_shapes.append(self._get_sym(expr.name))
                    return IntAttr(-1)
                case LoopIR.BinOp():
                    if self.symbol_table is not None:
                        dynamic_shapes.append(self._binop_expr(expr))
                    return IntAttr(-1)
                case _:
                    assert False, f"invalid shape argument {expr}"

        return ([attr_from_expr(expr) for expr in type.shape()], dynamic_shapes)

    def _get_static_shape(self, type) -> list[int]:
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return expr.val
                case LoopIR.Read() | LoopIR.BinOp():
                    return -1
                case _:
                    assert False, f"invalid shape argument {expr}"

        return [attr_from_expr(expr) for expr in type.shape()]

    def _get_dynamic_shape(self, type) -> list[SSAValue[Attribute] | int]:
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return expr.val
                case LoopIR.Read():
                    return self._get_sym(expr.name)
                case LoopIR.BinOp():
                    return self._binop_expr(expr)
                case _:
                    assert False, f"invalid shape argument {expr}"

        return [attr_from_expr(expr) for expr in type.shape()]

    def _cast_to_index(self, value: SSAValue) -> SSAValue:
        if isinstance(value.type, IndexType):
            return value
        cast = CastsOp(value, IndexType())
        self.builder.insert(cast)
        return cast.result

    def _cast_to(self, value: SSAValue, type: Attribute) -> SSAValue:
        if value.type == type:
            return value

        if isinstance(type, IndexType) ^ isinstance(value.type, IndexType):
            cast = CastsOp(value, type)
            result = cast.result

        elif isinstance(type, MemRefType) and isinstance(value.type, MemRefType):
            assert type.element_type == value.type.element_type, f"cannot cast from {value.type} to {type} as inner types do not match"

            cast = MemrefCastOp.get(value, type)
            result = cast.results[0]
        else:
            assert False, f"unknown cast from {value.type} to {type}"

        self.builder.insert(cast)
        return result

    #
    # expression generation
    #

    def _const_expr(self, const):
        type = self._get_type(const.type)

        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(const.val), type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            assert False, f"unknown type {type} passed to Const"

        const = ConstantOp(attr, self._get_type(const.type))
        self.builder.insert(const)
        return const.result

    def _read_expr(self, read):
        idx = self._expr_list(read.idx)

        operand = self._get_sym(read.name)

        exo_type = self._get_sym_exo_type(read.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self._get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(op := ReadOp(operand, idx, sizes, result_type=self._get_type(read.type)))

        return op.result

    def _usub_expr(self, usub):
        expr = self._expr(usub.arg)

        if self._get_type(usub.type) in [f16, f32, f64]:
            usub = NegfOp(expr)
        elif self._get_type(usub.type) in [i8, i16, i32, i64]:
            zero = ConstantOp(IntegerAttr(0, self._get_type(usub.type)))
            usub = SubiOp(zero.result, expr, result_type=self._get_type(usub.type))
            self.builder.insert(zero)
        else:
            assert False, f"bad type {type} passed to USub"

        self.builder.insert(usub)
        return usub.result

    def _binop_expr(self, binop):
        type = self._get_type(binop.type)
        if type == i1:
            return self._binop_expr_cmp(binop)

        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        float_ops = {"+": AddfOp, "-": SubfOp, "*": MulfOp, "/": DivfOp}
        int_ops = {"+": AddiOp, "-": SubiOp, "*": MuliOp, "/": DivSIOp, "%": RemSIOp}

        if type in [f16, f32, f64]:
            op_cls = float_ops[binop.op]
            op = op_cls(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif type in [i8, i16, i32, i64]:
            op_cls = int_ops[binop.op]
            op = op_cls(lhs, rhs, result_type=type)
        else:
            assert False, f"unknown type '{type.name}'"

        self.builder.insert(op)
        return op.result

    def _binop_expr_cmp(self, binop):
        integer_cmp_table = {
            "==": "eq",
            "!=": "ne",
            "<": "slt",
            "<=": "sle",
            ">": "sgt",
            ">=": "sge",
        }
        float_cmp_table = {
            "==": "oeq",
            "!=": "one",
            "<": "olt",
            "<=": "ole",
            ">": "ogt",
            ">=": "oge",
        }

        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        assert lhs.type == rhs.type, f"cannot compare {lhs.type} and {rhs.type} with operator '{binop.op}'"

        if lhs.type == i1:
            if binop.op == "and":
                binop = AndIOp(lhs, rhs)
            elif binop.op == "or":
                binop = OrIOp(lhs, rhs)
            else:
                assert False, f"unknown boolean operator '{binop.op}'"
        elif lhs.type in [i8, i16, i32, i64]:
            op = integer_cmp_table[binop.op]
            if op is None:
                assert False, f"unknown integer comparison operator '{binop.op}'"

            binop = CmpiOp(lhs, rhs, op)
        else:
            op = float_cmp_table[binop.op]
            if op is None:
                assert False, f"unknown float comparison operator '{binop.op}'"

            binop = CmpfOp(lhs, rhs, op)

        self.builder.insert(binop)
        return binop.result

    def _window_expr(self, window):
        idx = [self._w_access(w_access) for w_access in window.idx]

        input = self._get_sym(window.name)
        dest_type = self._get_type(window.type.as_tensor, input.type.memory_space)

        input_sizes = self._get_dynamic_shape(self._get_sym_exo_type(window.name))
        output_sizes = self._get_dynamic_shape(window.type.as_tensor)

        self.builder.insert(op := WindowOp(self._get_sym(window.name), idx, input_sizes, output_sizes, dest_type))

        return op.result

    def _w_access(self, w_access):
        match w_access:
            case LoopIR.Point():
                return self._expr(w_access.pt)
            case LoopIR.Interval():
                lo = self._expr(w_access.lo)
                hi = self._expr(w_access.hi)
                self.builder.insert(op := IntervalOp(lo, hi))
                return op.result
            case _:
                assert False, f"unknown window access type '{type(w_access)}' for '{w_access}'"

    def _extern_expr(self, extern):
        output_type = self._get_type(extern.f.typecheck(extern.args))
        args = self._expr_list(extern.args)
        self.builder.insert(op := ExternOp(extern.f.name(), args, output_type))
        return op.result

    def _expr_list(self, exprs) -> list[OpResult | SSAValue]:
        return [self._expr(expr) for expr in exprs]

    def _expr(self, expr) -> OpResult | SSAValue:
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
                assert False, f"unknown expression type '{type(expr)}' for expression '{expr}'"

    #
    # statement generation
    #

    def _assign_stmt(self, assign):
        idx = self._expr_list(assign.idx)
        value = self._expr(assign.rhs)
        memref = self._get_sym(assign.name)

        exo_type = self._get_sym_exo_type(assign.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self._get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(AssignOp(value, memref, idx, sizes))

    def _reduce_stmt(self, reduce):
        memref = self._get_sym(reduce.name)
        idx = self._expr_list(reduce.idx)
        value = self._expr(reduce.rhs)

        exo_type = self._get_sym_exo_type(reduce.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self._get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(ReduceOp(value, memref, idx, sizes))

    def _if_stmt(self, if_stmt):
        cond = self._expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        self._stmt_list(if_stmt.body)
        self.builder.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        self._stmt_list(if_stmt.orelse)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.builder = parent_builder
        self.builder.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def _for_stmt(self, for_stmt):
        lo = self._expr(for_stmt.lo)
        hi = self._expr(for_stmt.hi)
        step = ConstantOp(IntegerAttr(1, i64))
        self.builder.insert(step)

        parent_builder = self.builder
        parent_scope = self.symbol_table

        # construct loop block
        loop_block = Block(
            # TODO: this should be inferred from lo and hi
            arg_types=[i64],
        )
        self.builder = Builder(insertion_point=InsertPoint.at_end(loop_block))
        self.symbol_table = ScopedDict(parent_scope)

        # add loop variable to symbol table
        self._declare_arg(for_stmt.iter, loop_block.args[0])
        self._declare_sym_exo_type(for_stmt.iter, T.Index)

        # generate loop body
        self._stmt_list(for_stmt.body)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.builder.insert(ForOp(lo, hi, step.result, [], Region(loop_block)))

    def _alloc_stmt(self, alloc):
        type = self._get_type(alloc.type, StringAttr(alloc.mem.name()))
        self.builder.insert(op := AllocOp(alloc.mem.name(), type))
        self._declare_value(alloc.name, op.results[0])
        self._declare_sym_exo_type(alloc.name, alloc.type)
        return op.result

    def _free_stmt(self, free):
        self.builder.insert(FreeOp(self._get_sym(free.name), free.mem.name()))

    def _call_stmt(self, call):
        args = [self._expr(arg) for arg in call.args]

        if call.f.instr is not None:
            self.builder.insert(InstrOp(call.f.name, args))
            return

        self._procedure(call.f)

        if len(call.args) != len(call.f.args):
            assert False, f"call to '{call.f.name}' has {len(call.args)} arguments, expected {len(call.f.args)}"

        self.builder.insert(CallOp(call.f.name, args, []))

    def _stmt_list(self, stmts):
        for stmt in stmts:
            self._stmt(stmt)

    def _stmt(self, stmt):
        match stmt:
            case LoopIR.Assign():
                self._assign_stmt(stmt)
            case LoopIR.Reduce():
                self._reduce_stmt(stmt)
            case LoopIR.WriteConfig():
                raise NotImplementedError("WriteConfig is not supported")
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
                assert False, "window statements are not supported"
            case _:
                assert False, f"unknown statement {stmt}"

    def _procedure(self, procedure):
        if procedure.name in self.seen_procs:
            return

        self.seen_procs.add(procedure.name)

        input_types = [self._get_type(arg.type) for arg in procedure.args]
        input_types = [(MemRefType(ty.element_type, ty.shape, ty.layout, StringAttr(arg.mem.name())) if isinstance(ty, MemRefType) else ty) for (ty, arg) in zip(input_types, procedure.args)]

        func_type = FunctionType.from_lists(input_types, [])

        # instantiate builder at module level
        parent_builder = self.builder
        module_builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))

        # generate private funcs for instruction procedures
        if procedure.instr is not None:
            return

        parent_symbol_table = self.symbol_table
        parent_type_table = self.type_table
        self.symbol_table = ScopedDict[str, SSAValue]()
        self.type_table = ScopedDict[str, Attribute]()

        # initialise function block
        block = Block(arg_types=input_types)
        self.builder = Builder(insertion_point=InsertPoint.at_end(block))

        # add arguments to symbol table
        for proc_arg, block_arg in zip(procedure.args, block.args):
            self._declare_arg(proc_arg.name, block_arg)
            self._declare_sym_exo_type(proc_arg.name, proc_arg.type)

        # generate function body
        self._stmt_list(procedure.body)
        self.builder.insert(ReturnOp())

        # cleanup
        self.symbol_table = parent_symbol_table
        self.type_table = parent_type_table
        self.builder = parent_builder

        # insert procedure into module
        module_builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate(self, procs) -> ModuleOp:
        for proc in procs:
            self._procedure(proc)

        self.module.verify()  # structural only, exo ops don't implement verify_()
        return self.module


@cache
def context() -> Context:
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(Exo)
    ctx.load_dialect(Index)
    ctx.load_dialect(LLVMIntrinsics)
    return ctx


def transform(analyzed_procs: list, target: str = "llvm", prefix: str | None = None) -> ModuleOp:
    ctx = context()
    module = IRGenerator().generate(analyzed_procs)  # exo LoopIR -> xdsl MLIR

    # lower exo dialect to standard mlir
    InlineMemorySpacePass().apply(ctx, module)
    ConvertScalarRefPass().apply(ctx, module)
    ReconcileIndexCastsPass().apply(ctx, module)
    module.verify()

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    if target == "exo":
        return module

    # optional function renaming
    if prefix is not None:
        AddPrefixPass(prefix).apply(ctx, module)
        module.verify()

    # lower to llvm
    InlineBLASAllocPass().apply(ctx, module)
    ConvertMemRefToLLVM().apply(ctx, module)
    InlineAVX2Pass().apply(ctx, module)
    InlineBLASPass().apply(ctx, module)
    ConvertScfToCf().apply(ctx, module)
    ReconcileUnrealizedCastsPass().apply(ctx, module)
    module.verify()

    # optimize
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    module.verify()

    return module


def compile_procs(
    library: Sequence[Procedure],  # list of exo funcs decorated with @proc
    target: str = "llvm",
    prefix: str | None = None,
) -> ModuleOp:
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
    return transform(analyzed_procs, target, prefix)


def main():
    parser = ArgumentParser(description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, help="Source file to compile")
    parser.add_argument("-o", "--output", help="Output file. Defaults to stdout.")
    parser.add_argument("--target", default="llvm", choices=["llvm", "exo", "builtin", "lowered", "scf"])
    parser.add_argument("--prefix", help="Prefix to prepend to all procedure names.")
    args = parser.parse_args()

    src = Path(args.source)
    assert src.is_file() and src.suffix == ".py"

    library = get_procs_from_module(load_user_code(src))
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_procs(library, args.target, args.prefix)

    dst = None
    if args.output and args.output != "-":
        dst = Path(args.output)

    if not dst:
        print(module)
        return
    os.makedirs(dst.parent, exist_ok=True)
    dst.write_text(str(module))
