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
from exo.core.LoopIR import LoopIR
from exo.main import get_procs_from_module, load_user_code

from xdsl.context import Context
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import Builtin, ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl_exo.dialects import extra
from xdsl_exo.dialects.exo import Exo
from xdsl_exo.dialects.extra import Index, LLVMIntrinsics
from xdsl_exo.generator import IRGenerator
from xdsl_exo.platforms.avx2 import InlineAVX2Pass
from xdsl_exo.platforms.blas import InlineBLASAllocPass, InlineBLASPass
from xdsl_exo.rewrites.convert_memref_to_llvm import ConvertMemRefToLLVM
from xdsl_exo.rewrites.convert_scalar_ref import ConvertScalarRefPass
from xdsl_exo.rewrites.inline_memory_space import InlineMemorySpacePass


class ReconcileIndexCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: extra.CastsOp, rewriter: PatternRewriter):
        if len(op.result.uses) == 0:
            rewriter.erase_matched_op()
            return

        # replace x -> y -> x cast with x
        if not isinstance(op.input.owner, extra.CastsOp):
            return

        if op.input.owner.input.type != op.result.type:
            return

        rewriter.replace_matched_op((), (op.input.owner.input,))


class ReconcileIndexCastsPass(ModulePass):
    name = "reconcile-index-casts"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ReconcileIndexCasts()]),
            walk_reverse=True,
        ).rewrite_module(m)


class ConvertFuncOp(RewritePattern):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.name.startswith(self.prefix):
            return

        op.sym_name = StringAttr(f"{self.prefix}_{op.sym_name.data}")


class AddPrefixPass(ModulePass):
    name = "add-prefix"

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertFuncOp(self.prefix),
                ]
            ),
        ).rewrite_module(m)


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
    module = IRGenerator().generate(analyzed_procs)

    InlineMemorySpacePass().apply(ctx, module)
    module.verify()

    ConvertScalarRefPass().apply(ctx, module)
    module.verify()

    ReconcileIndexCastsPass().apply(ctx, module)
    module.verify()

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)

    module.verify()

    if target == "exo":
        return module

    if prefix is not None:
        AddPrefixPass(prefix).apply(ctx, module)
        module.verify()

    InlineBLASAllocPass().apply(ctx, module)
    module.verify()

    ConvertMemRefToLLVM().apply(ctx, module)
    module.verify()
    InlineAVX2Pass().apply(ctx, module)
    module.verify()
    InlineBLASPass().apply(ctx, module)
    module.verify()

    ConvertScfToCf().apply(ctx, module)
    ReconcileUnrealizedCastsPass().apply(ctx, module)
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
