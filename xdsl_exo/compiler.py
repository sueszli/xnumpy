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
    seen_externs: set[str] = set()

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))

    def with_empty_scope(self):
        # return this IRGenerator with an empty symbol table.
        self.symbol_table = ScopedDict()
        self.type_table = ScopedDict()
        return self

    def declare_arg(self, sym: Sym, arg: BlockArgument) -> BlockArgument:
        # declare a symbol in the symbol table.
        assert self.symbol_table is not None
        self.declare_value(sym, arg)
        return arg

    def declare_value(self, sym: Sym, value: SSAValue) -> SSAValue:
        # declare a value in the symbol table.
        assert self.symbol_table is not None
        self.symbol_table[sym.__repr__()] = value
        return value

    def _with_test_op(self, sym: Sym, type):
        assert self.symbol_table is not None
        op = TestOp(result_types=[self.get_type(type)])
        self.builder.insert(op)
        self.symbol_table[sym.__repr__()] = op.res[0]
        if self.type_table is not None:
            self.type_table[sym.__repr__()] = type
        return self

    def get_sym(self, sym: Sym) -> SSAValue:
        # get the SSAValue for a symbol.
        assert self.symbol_table is not None

        assert sym.__repr__() in self.symbol_table, f"unknown symbol {sym.__repr__()}"
        return self.symbol_table[sym.__repr__()]

    def declare_sym_exo_type(self, sym: Sym, type):
        # declare a type for a symbol in the type table.
        assert self.type_table is not None
        self.type_table[sym.__repr__()] = type
        return type

    def get_sym_exo_type(self, sym: Sym):
        # get the type for a symbol.
        assert self.type_table is not None

        assert sym.__repr__() in self.type_table, f"unknown symbol {sym.__repr__()}"
        return self.type_table[sym.__repr__()]

    def cast_to_index(self, value: SSAValue) -> SSAValue:
        # must not cast if already an index
        if isinstance(value.type, IndexType):
            return value
        cast = CastsOp(value, IndexType())
        self.builder.insert(cast)
        return cast.result

    def cast_to(self, value: SSAValue, type: Attribute) -> SSAValue:
        # no need to cast if types match
        if value.type == type:
            return value

        if isinstance(type, IndexType) ^ isinstance(value.type, IndexType):
            cast = CastsOp(value, type)
            result = cast.result

        elif isinstance(type, MemRefType) and isinstance(value.type, MemRefType):
            # check inner types are equal
            assert type.element_type == value.type.element_type, f"cannot cast from {value.type} to {type} as inner types do not match"

            cast = MemrefCastOp.get(value, type)
            result = cast.results[0]
        else:
            assert False, f"unknown cast from {value.type} to {type}"

        self.builder.insert(cast)
        return result

    def generate(self, procs) -> ModuleOp:
        # generate the MLIR module from the given procedures and verify it.
        for proc in procs:
            self.generate_procedure(proc)

        # verify module
        # TODO: none of the operations actually implement verify_()
        try:
            self.module.verify()
        except Exception as e:
            print("module verification failed: ", e)
            raise

        return self.module

    def generate_procedure(self, procedure):
        # generate a procedure.

        if procedure.name in self.seen_procs:
            return

        self.seen_procs.add(procedure.name)

        input_types = [self.get_type(arg.type) for arg in procedure.args]
        input_types = [
            (
                MemRefType(
                    ty.element_type,
                    ty.shape,
                    ty.layout,
                    StringAttr(arg.mem.name()),
                )
                if isinstance(ty, MemRefType)
                else ty
            )
            for (ty, arg) in zip(input_types, procedure.args)
        ]

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
            self.declare_arg(proc_arg.name, block_arg)
            self.declare_sym_exo_type(proc_arg.name, proc_arg.type)

        # generate function body
        self.generate_stmt_list(procedure.body)
        self.builder.insert(ReturnOp())

        # cleanup
        self.symbol_table = parent_symbol_table
        self.type_table = parent_type_table
        self.builder = parent_builder

        # insert procedure into module
        module_builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    def generate_stmt_list(self, stmts):
        # generate a list of statements.
        for stmt in stmts:
            self.generate_stmt(stmt)

    def generate_stmt(self, stmt):
        match stmt:
            case LoopIR.Assign():
                self.generate_assign_stmt(stmt)
            case LoopIR.Reduce():
                self.generate_reduce_stmt(stmt)
            case LoopIR.WriteConfig():
                self.generate_write_config_stmt(stmt)
            case LoopIR.Pass():
                pass
            case LoopIR.If():
                self.generate_if_stmt(stmt)
            case LoopIR.For():
                self.generate_for_stmt(stmt)
            case LoopIR.Alloc():
                self.generate_alloc_stmt(stmt)
            case LoopIR.Free():
                self.generate_free_stmt(stmt)
            case LoopIR.Call():
                self.generate_call_stmt(stmt)
            case LoopIR.Window():
                assert False, "window statements are not supported"
            case _:
                assert False, f"unknown statement {stmt}"

    def generate_assign_stmt(self, assign):
        idx = self.generate_expr_list(assign.idx)
        value = self.generate_expr(assign.rhs)
        memref = self.get_sym(assign.name)

        exo_type = self.get_sym_exo_type(assign.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(AssignOp(value, memref, idx, sizes))

    def generate_reduce_stmt(self, reduce):
        memref = self.get_sym(reduce.name)
        idx = self.generate_expr_list(reduce.idx)
        value = self.generate_expr(reduce.rhs)

        exo_type = self.get_sym_exo_type(reduce.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(ReduceOp(value, memref, idx, sizes))

    def generate_write_config_stmt(self, write_config):
        # rhs = self.generate_expr(write_config.rhs)
        # self.builder.insert(WriteConfigOp(write_config.name, write_config.field, rhs))
        raise NotImplementedError

    def generate_if_stmt(self, if_stmt):
        cond = self.generate_expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        self.generate_stmt_list(if_stmt.body)
        self.builder.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        self.generate_stmt_list(if_stmt.orelse)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.builder = parent_builder
        self.builder.insert(IfOp(cond, [], Region(true_block), Region(false_block)))

    def generate_for_stmt(self, for_stmt):
        lo = self.generate_expr(for_stmt.lo)
        hi = self.generate_expr(for_stmt.hi)
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
        self.declare_arg(for_stmt.iter, loop_block.args[0])
        self.declare_sym_exo_type(for_stmt.iter, T.Index)

        # generate loop body
        self.generate_stmt_list(for_stmt.body)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.builder.insert(ForOp(lo, hi, step.result, [], Region(loop_block)))

    def generate_alloc_stmt(self, alloc):
        type = self.get_type(alloc.type, StringAttr(alloc.mem.name()))
        self.builder.insert(op := AllocOp(alloc.mem.name(), type))
        self.declare_value(alloc.name, op.results[0])
        self.declare_sym_exo_type(alloc.name, alloc.type)
        return op.result

    def generate_free_stmt(self, free):
        self.builder.insert(FreeOp(self.get_sym(free.name), free.mem.name()))

    def generate_call_stmt(self, call):
        # build arguments
        args = [self.generate_expr(arg) for arg in call.args]

        if call.f.instr is not None:
            self.builder.insert(InstrOp(call.f.name, args))
            return

        self.generate_procedure(call.f)

        # ensure arg lengths match
        if len(call.args) != len(call.f.args):
            assert False, f"call to '{call.f.name}' has {len(call.args)} arguments, expected {len(call.f.args)}"

        self.builder.insert(CallOp(call.f.name, args, []))

    # def generate_window_stmt(self, window):
    #     rhs = self.generate_expr(window.rhs)
    #     self.builder.insert(WindowStmtOp(self.symbol(window.name), rhs))

    def generate_expr_list(self, exprs) -> list[OpResult | SSAValue]:
        return [self.generate_expr(expr) for expr in exprs]

    def generate_expr(self, expr) -> OpResult | SSAValue:
        match expr:
            case LoopIR.Read():
                return self.generate_read_expr(expr)
            case LoopIR.Const():
                return self.generate_const_expr(expr)
            case LoopIR.USub():
                return self.generate_usub_expr(expr)
            case LoopIR.BinOp():
                return self.generate_binop_expr(expr)
            case LoopIR.WindowExpr():
                return self.generate_window_expr(expr)
            case LoopIR.Extern():
                return self.generate_extern_expr(expr)
            case _:
                assert False, f"unknown expression type '{type(expr)}' for expression '{expr}'"

    def generate_read_expr(self, read):
        idx = self.generate_expr_list(read.idx)

        operand = self.get_sym(read.name)

        exo_type = self.get_sym_exo_type(read.name)
        if isinstance(exo_type, T.Tensor):
            sizes = self.get_dynamic_shape(exo_type)
        else:
            sizes = []

        self.builder.insert(op := ReadOp(operand, idx, sizes, result_type=self.get_type(read.type)))

        return op.result

    def generate_const_expr(self, const):
        type = self.get_type(const.type)

        # construct attribute depending on type
        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(const.val), type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            assert False, f"unknown type {type} passed to Const"

        const = ConstantOp(attr, self.get_type(const.type))
        self.builder.insert(const)
        return const.result

    def generate_usub_expr(self, usub):
        # generate a unary negation expression.

        expr = self.generate_expr(usub.arg)
        # float case
        if self.get_type(usub.type) in [f16, f32, f64]:
            usub = NegfOp(expr)
        # integer case
        elif self.get_type(usub.type) in [i8, i16, i32, i64]:
            zero = ConstantOp(IntegerAttr(0, self.get_type(usub.type)))
            usub = SubiOp(zero.result, expr, result_type=self.get_type(usub.type))
            self.builder.insert(zero)
        else:
            assert False, f"bad type {type} passed to USub"

        self.builder.insert(usub)
        return usub.result

    def generate_binop_expr(self, binop):
        # generate a binary operation expression.

        type = self.get_type(binop.type)

        if type in [f16, f32, f64]:
            return self.generate_binop_expr_float(binop)
        elif type in [i8, i16, i32, i64]:
            return self.generate_binop_expr_int(binop)
        elif type == i1:
            return self.generate_binop_expr_cmp(binop)
        else:
            assert False, f"unknown type '{type.name}'"

    def generate_binop_expr_float(self, binop):
        # generate a floating point binary operation expression.

        type = self.get_type(binop.type)
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        if binop.op == "+":
            binop = AddfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "-":
            binop = SubfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "*":
            binop = MulfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        elif binop.op == "/":
            binop = DivfOp(lhs, rhs, result_type=type, flags=FastMathFlagsAttr("none"))
        else:
            assert False, f"unknown binop {binop.op}"

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_int(self, binop):
        # generate an integer binary operation expression.

        type = self.get_type(binop.type)
        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        if binop.op == "+":
            binop = AddiOp(lhs, rhs, result_type=type)
        elif binop.op == "-":
            binop = SubiOp(lhs, rhs, result_type=type)
        elif binop.op == "*":
            binop = MuliOp(lhs, rhs, result_type=type)
        elif binop.op == "/":
            binop = DivSIOp(lhs, rhs, result_type=type)
        elif binop.op == "%":
            binop = RemSIOp(lhs, rhs, result_type=type)
        else:
            assert False, f"unknown binop {binop.op}"

        self.builder.insert(binop)
        return binop.result

    def generate_binop_expr_cmp(self, binop):
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

        lhs = self.generate_expr(binop.lhs)
        rhs = self.generate_expr(binop.rhs)

        assert lhs.type == rhs.type, f"cannot compare {lhs.type} and {rhs.type} with operator '{binop.op}'"

        # boolean operations
        if lhs.type == i1:
            if binop.op == "and":
                binop = AndIOp(lhs, rhs)
            elif binop.op == "or":
                binop = OrIOp(lhs, rhs)
            else:
                assert False, f"unknown boolean operator '{binop.op}'"
        # cmpi
        elif lhs.type in [i8, i16, i32, i64]:
            op = integer_cmp_table[binop.op]
            if op is None:
                assert False, f"unknown integer comparison operator '{binop.op}'"

            binop = CmpiOp(lhs, rhs, op)
        # cmpf
        else:
            op = float_cmp_table[binop.op]
            if op is None:
                assert False, f"unknown float comparison operator '{binop.op}'"

            binop = CmpfOp(lhs, rhs, op)

        self.builder.insert(binop)
        return binop.result

    def generate_window_expr(self, window):
        # compute indices and result type
        idx = [self.generate_w_access(w_access) for w_access in window.idx]

        input = self.get_sym(window.name)
        dest_type = self.get_type(window.type.as_tensor, input.type.memory_space)

        input_sizes = self.get_dynamic_shape(self.get_sym_exo_type(window.name))
        output_sizes = self.get_dynamic_shape(window.type.as_tensor)

        self.builder.insert(op := WindowOp(self.get_sym(window.name), idx, input_sizes, output_sizes, dest_type))

        return op.result

    def generate_w_access(self, w_access):
        match w_access:
            case LoopIR.Point():
                return self.generate_expr(w_access.pt)
            case LoopIR.Interval():
                lo = self.generate_expr(w_access.lo)
                hi = self.generate_expr(w_access.hi)
                self.builder.insert(op := IntervalOp(lo, hi))
                return op.result
            case _:
                assert False, f"unknown window access type '{type(w_access)}' for '{w_access}'"

    def generate_stride_expr(self, stride):
        raise NotImplementedError("stride expressions are not yet supported")

    def generate_extern_expr(self, extern):
        # query exo for the type of the result
        output_type = self.get_type(extern.f.typecheck(extern.args))
        args = self.generate_expr_list(extern.args)
        self.builder.insert(op := ExternOp(extern.f.name(), args, output_type))
        return op.result

    def generate_read_config_expr(self, read_config):
        raise NotImplementedError()

    def get_type(self, t, mem_space=StringAttr("DRAM")) -> Attribute:
        # get the type of a LoopIR type as an MLIR type.

        _MEMREF_ELEMENT_TYPES = {f16, f32, f64, i8, i16, i32}

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
                inner = self.get_type(t.type)
                assert inner in _MEMREF_ELEMENT_TYPES, f"unknown tensor inner type '{inner}'"
                shape = self.get_static_shape(t)
                return MemRefType(inner, shape, NoneAttr(), mem_space)
            case _:
                assert False, f"unknown type '{t}'"

    def get_shape(self, type) -> tuple[list[IntegerAttr], list[SSAValue]]:
        # get the shape of a tensor type as a list of integer attributes.
        assert isinstance(type, T.Tensor)

        dynamic_shapes = []

        def attr_from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return IntAttr(expr.val)
                case LoopIR.Read():
                    if self.symbol_table is not None:
                        dynamic_shapes.append(self.get_sym(expr.name))
                    return IntAttr(-1)
                case LoopIR.BinOp():
                    if self.symbol_table is not None:
                        dynamic_shapes.append(self.generate_binop_expr(expr))
                    return IntAttr(-1)
                case _:
                    assert False, f"invalid shape argument {expr}"

        return ([attr_from_expr(expr) for expr in type.shape()], dynamic_shapes)

    def get_static_shape(self, type) -> list[int]:
        # get the shape of a tensor type as a list of integer attributes.
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

    def get_dynamic_shape(self, type) -> list[SSAValue[Attribute] | int]:
        # get the shape of a tensor type as a list of integer attributes.
        assert isinstance(type, T.Tensor)

        def attr_from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return expr.val
                case LoopIR.Read():
                    return self.get_sym(expr.name)
                case LoopIR.BinOp():
                    return self.generate_binop_expr(expr)
                case _:
                    assert False, f"invalid shape argument {expr}"

        return [attr_from_expr(expr) for expr in type.shape()]


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

    # run exo analysis passes
    def analyze(proc):
        assert isinstance(proc, LoopIR.proc)
        proc = ParallelAnalysis().run(proc)
        proc = PrecisionAnalysis().run(proc)
        proc = WindowAnalysis().apply_proc(proc)
        return MemoryAnalysis().run(proc)

    analyzed_procs = [analyze(proc) for proc in unique_procs]
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
