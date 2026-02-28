from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Sequence
from functools import cache
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
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.arith import AddfOp, AddiOp, AndIOp, CmpfOp, CmpiOp, ConstantOp, DivfOp, DivSIOp, FastMathFlagsAttr, MulfOp, MuliOp, NegfOp, OrIOp, RemSIOp, SubfOp, SubiOp
from xdsl.dialects.builtin import BoolAttr, Builtin, FloatAttr, FunctionType, IntAttr, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StringAttr, f16, f32, f64, i1, i8, i16, i32, i64
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.dialects.scf import ForOp, IfOp, YieldOp
from xdsl.ir import Attribute, Block, OpResult, Region, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.utils.scoped_dict import ScopedDict

from xdsl_exo.dialects.exo import AllocOp, AssignOp, Exo, ExternOp, FreeOp, InstrOp, IntervalOp, ReadOp, ReduceOp, WindowOp
from xdsl_exo.dialects.index import Index
from xdsl_exo.dialects.llvm import LLVMIntrinsics
from xdsl_exo.platforms.avx2 import InlineAVX2Pass
from xdsl_exo.platforms.blas import InlineBLASAllocPass, InlineBLASPass
from xdsl_exo.rewrites.convert_memref_to_llvm import ConvertMemRefToLLVM
from xdsl_exo.rewrites.convert_scalar_ref import ConvertScalarRefPass
from xdsl_exo.rewrites.inline_memory_space import InlineMemorySpacePass
from xdsl_exo.rewrites.reconcile_index_casts import ReconcileIndexCastsPass


class IRGenerator:
    module: ModuleOp  # container for IR
    builder: Builder  # string builder
    symbol_table: ScopedDict[str, SSAValue] | None
    type_table: ScopedDict[str, Attribute] | None
    seen_procs: set[str]  # avoid duplicates

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        self.symbol_table = None
        self.type_table = None
        self.seen_procs = set()

    #
    # helpers
    #

    def _type(self, t, mem_space: StringAttr | None = None) -> Attribute:
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
                assert mem_space is not None
                inner = self._type(t.type)
                assert inner in {f16, f32, f64, i8, i16, i32}
                shape = self._shape(t)
                return MemRefType(inner, shape, NoneAttr(), mem_space)
            case _:
                assert False

    def _shape(self, type, *, dynamic=False) -> list:
        assert isinstance(type, T.Tensor)

        def from_expr(expr):
            match expr:
                case LoopIR.Const():
                    return expr.val
                case LoopIR.Read():
                    return self.symbol_table[repr(expr.name)] if dynamic else -1
                case LoopIR.BinOp():
                    return self._binop_expr(expr) if dynamic else -1
                case _:
                    assert False

        return [from_expr(expr) for expr in type.shape()]

    def _sizes_for(self, name) -> list:
        exo_type = self.type_table[repr(name)]
        if isinstance(exo_type, T.Tensor):
            return self._shape(exo_type, dynamic=True)
        return []

    #
    # expression generation
    #

    def _const_expr(self, const):
        type = self._type(const.type)

        if type in [f16, f32, f64]:
            attr = FloatAttr(const.val, type)
        elif type in [i8, i16, i32, i64]:
            attr = IntegerAttr(IntAttr(int(const.val)), type)
        elif type == i1:
            attr = BoolAttr(const.val, i1)
        else:
            assert False

        const = ConstantOp(attr, type)
        self.builder.insert(const)
        return const.result

    def _read_expr(self, read):
        idx = [self._expr(e) for e in read.idx]
        operand = self.symbol_table[repr(read.name)]
        sizes = self._sizes_for(read.name)

        self.builder.insert(op := ReadOp(operand, idx, sizes, result_type=self._type(read.type)))

        return op.result

    def _usub_expr(self, usub):
        expr = self._expr(usub.arg)
        type = self._type(usub.type)

        if type in [f16, f32, f64]:
            usub = NegfOp(expr)
        elif type in [i8, i16, i32, i64]:
            zero = ConstantOp(IntegerAttr(0, type))
            self.builder.insert(zero)
            usub = SubiOp(zero.result, expr, result_type=type)
        else:
            assert False

        self.builder.insert(usub)
        return usub.result

    def _binop_expr(self, binop):
        type = self._type(binop.type)
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
            assert False

        self.builder.insert(op)
        return op.result

    def _binop_expr_cmp(self, binop):
        integer_cmp_table = {"==": "eq", "!=": "ne", "<": "slt", "<=": "sle", ">": "sgt", ">=": "sge"}
        float_cmp_table = {"==": "oeq", "!=": "one", "<": "olt", "<=": "ole", ">": "ogt", ">=": "oge"}

        lhs = self._expr(binop.lhs)
        rhs = self._expr(binop.rhs)

        assert lhs.type == rhs.type, f"cannot compare {lhs.type} and {rhs.type} with operator '{binop.op}'"

        if lhs.type == i1:
            if binop.op == "and":
                binop = AndIOp(lhs, rhs)
            elif binop.op == "or":
                binop = OrIOp(lhs, rhs)
            else:
                assert False
        elif lhs.type in [i8, i16, i32, i64]:
            binop = CmpiOp(lhs, rhs, integer_cmp_table[binop.op])
        else:
            binop = CmpfOp(lhs, rhs, float_cmp_table[binop.op])

        self.builder.insert(binop)
        return binop.result

    def _window_expr(self, window):
        idx = [self._w_access(w_access) for w_access in window.idx]

        input = self.symbol_table[repr(window.name)]
        dest_type = self._type(window.type.as_tensor, input.type.memory_space)

        input_sizes = self._shape(self.type_table[repr(window.name)], dynamic=True)
        output_sizes = self._shape(window.type.as_tensor, dynamic=True)

        self.builder.insert(op := WindowOp(input, idx, input_sizes, output_sizes, dest_type))

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
                assert False

    def _extern_expr(self, extern):
        output_type = self._type(extern.f.typecheck(extern.args))
        args = [self._expr(e) for e in extern.args]
        self.builder.insert(op := ExternOp(extern.f.name(), args, output_type))
        return op.result

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
                assert False

    #
    # statement generation
    #

    def _store_stmt(self, stmt, op_cls):
        idx = [self._expr(e) for e in stmt.idx]
        value = self._expr(stmt.rhs)
        memref = self.symbol_table[repr(stmt.name)]
        sizes = self._sizes_for(stmt.name)
        self.builder.insert(op_cls(value, memref, idx, sizes))

    def _if_stmt(self, if_stmt):
        cond = self._expr(if_stmt.cond)

        parent_builder = self.builder

        # construct true_block
        true_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(true_block))
        for s in if_stmt.body:
            self._stmt(s)
        self.builder.insert(YieldOp())

        # construct false_block
        false_block = Block()
        self.builder = Builder(insertion_point=InsertPoint.at_end(false_block))
        for s in if_stmt.orelse:
            self._stmt(s)
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
        self.symbol_table[repr(for_stmt.iter)] = loop_block.args[0]
        self.type_table[repr(for_stmt.iter)] = T.Index

        # generate loop body
        for s in for_stmt.body:
            self._stmt(s)
        self.builder.insert(YieldOp())

        # cleanup and construct
        self.symbol_table = parent_scope
        self.builder = parent_builder

        self.builder.insert(ForOp(lo, hi, step.result, [], Region(loop_block)))

    def _alloc_stmt(self, alloc):
        type = self._type(alloc.type, StringAttr(alloc.mem.name()))
        self.builder.insert(op := AllocOp(alloc.mem.name(), type))
        self.symbol_table[repr(alloc.name)] = op.results[0]
        self.type_table[repr(alloc.name)] = alloc.type
        return op.result

    def _free_stmt(self, free):
        self.builder.insert(FreeOp(self.symbol_table[repr(free.name)], free.mem.name()))

    def _call_stmt(self, call):
        args = [self._expr(arg) for arg in call.args]

        if call.f.instr is not None:
            self.builder.insert(InstrOp(call.f.name, args))
            return

        self._procedure(call.f)
        assert len(call.args) == len(call.f.args)

        self.builder.insert(CallOp(call.f.name, args, []))

    def _stmt(self, stmt):
        match stmt:
            case LoopIR.Assign():
                self._store_stmt(stmt, AssignOp)
            case LoopIR.Reduce():
                self._store_stmt(stmt, ReduceOp)
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

    def _procedure(self, procedure):
        if procedure.name in self.seen_procs:
            return
        self.seen_procs.add(procedure.name)

        if procedure.instr is not None:
            raise NotImplementedError()

        input_types = []
        for arg in procedure.args:
            if hasattr(arg, "mem"):
                input_types.append(self._type(arg.type, StringAttr(arg.mem.name())))
            else:
                input_types.append(self._type(arg.type))
        func_type = FunctionType.from_lists(input_types, [])

        parent_builder = self.builder
        parent_symbol_table = self.symbol_table
        parent_type_table = self.type_table
        self.symbol_table = ScopedDict[str, SSAValue]()
        self.type_table = ScopedDict[str, Attribute]()

        # initialise function block
        block = Block(arg_types=input_types)
        self.builder = Builder(insertion_point=InsertPoint.at_end(block))

        # add arguments to symbol table
        for proc_arg, block_arg in zip(procedure.args, block.args):
            self.symbol_table[repr(proc_arg.name)] = block_arg
            self.type_table[repr(proc_arg.name)] = proc_arg.type

        # generate function body
        for s in procedure.body:
            self._stmt(s)
        self.builder.insert(ReturnOp())

        # cleanup and insert procedure into module
        self.symbol_table = parent_symbol_table
        self.type_table = parent_type_table
        self.builder = parent_builder

        module_builder = Builder(insertion_point=InsertPoint.at_end(self.module.body.blocks[0]))
        module_builder.insert(FuncOp(procedure.name, func_type, Region(block)))

    #
    # entry point
    #

    def generate(self, procs) -> ModuleOp:
        for proc in procs:
            self._procedure(proc)

        self.module.verify()
        return self.module


@cache
def _context() -> Context:
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


def _transform(analyzed_procs: list, target: str = "llvm") -> ModuleOp:
    ctx = _context()

    # exo LoopIR -> raw exo IR
    module = IRGenerator().generate(analyzed_procs)

    # partial lowering: convert memory spaces, scalar refs, and index casts to standard mlir
    # (exo memory ops like exo.read, exo.assign, exo.reduce are preserved)
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

    # full lowering to llvm dialect
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
    return _transform(analyzed_procs, target)


def main():
    parser = ArgumentParser(description="Compile an Exo library to MLIR.")
    parser.add_argument("source", type=str, help="Source file to compile")
    parser.add_argument("-o", "--output", help="Output file. Defaults to stdout.")
    parser.add_argument("--target", default="llvm", choices=["llvm", "exo"])
    args = parser.parse_args()

    src = Path(args.source)
    assert src.is_file() and src.suffix == ".py"

    library = get_procs_from_module(load_user_code(src))
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_procs(library, args.target)

    dst = None
    if args.output and args.output != "-":
        dst = Path(args.output)

    if not dst:
        print(module)
        return
    os.makedirs(dst.parent, exist_ok=True)
    dst.write_text(str(module))
