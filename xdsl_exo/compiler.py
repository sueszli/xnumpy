import contextlib
import os
import sys
from collections.abc import Sequence
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
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.convert_scf_to_cf import ConvertScfToCf
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl_exo.dialects.exo import Exo
from xdsl_exo.dialects.index import Index
from xdsl_exo.dialects.llvm_intrinsics import LLVMIntrinsics
from xdsl_exo.generator import IRGenerator
from xdsl_exo.platforms.avx2 import InlineAVX2Pass
from xdsl_exo.platforms.blas import InlineBLASAllocPass, InlineBLASPass
from xdsl_exo.rewrites.add_prefix import AddPrefixPass
from xdsl_exo.rewrites.convert_memref_to_llvm import ConvertMemRefToLLVM
from xdsl_exo.rewrites.convert_scalar_ref import ConvertScalarRefPass
from xdsl_exo.rewrites.inline_memory_space import InlineMemorySpacePass
from xdsl_exo.rewrites.reconcile_index_casts import ReconcileIndexCastsPass


class CompilerOptions:
    """
    Compiler options for exo-mlir.
    """

    def __init__(self):
        self.target = "llvm"
        self.prefix = None
        self.verbose = False


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


def analyze(p):
    """
    Perform the default Exo analysis on a procedure.
    """

    assert isinstance(p, LoopIR.proc)

    p = ParallelAnalysis().run(p)
    p = PrecisionAnalysis().run(p)
    p = WindowAnalysis().apply_proc(p)
    return MemoryAnalysis().run(p)


def compile_one(proc: Procedure, opts: CompilerOptions = CompilerOptions()) -> ModuleOp:
    """
    Compile a single procedure. This is an alias for `compile_many([proc])`.
    """
    if proc.is_instr():
        raise TypeError("Cannot compile an instr procedure.")
    return compile_many([proc], opts)


def compile_many(
    library: Sequence[Procedure],
    opts: CompilerOptions = CompilerOptions(),
) -> ModuleOp:
    """
    Compile a list of procedures into a single MLIR module..
    """
    input_procedures = list(
        sorted(
            find_all_subprocs([proc._loopir_proc for proc in library if not proc.is_instr()]),
            key=lambda x: x.name,
        )
    )

    # ensure no duplicate procedures
    seen_procs = set()
    for proc in input_procedures:
        if proc.name in seen_procs:
            raise TypeError(f"multiple procs named {proc.name}")
        seen_procs.add(proc.name)

    # analyze procedures
    analyzed_procedures = [analyze(proc) for proc in input_procedures]

    # generate MLIR
    return transform(context(), IRGenerator().generate(analyzed_procedures), opts)


def compile_path(
    src: Path,
    dest: Path | None = None,
    opts: CompilerOptions = CompilerOptions(),
):
    """
    Compile all procedures in a Python source file to a single MLIR module, and write it to a file.
    """
    assert src.exists(), f"{src} does not exist."
    assert src.is_file() and src.suffix == ".py", f"{src} is not a Python source file."

    if opts.verbose:
        print(f"INFO: Compile[{src}] Destination: {dest}", file=sys.stderr)

    # load user code and get procedures from exo
    # procedures tend to do a lot of printing, so we suppress stdout temporarily
    with contextlib.redirect_stdout(None):
        library = get_procs_from_module(load_user_code(src))  # type: list[Procedure]

    if opts.verbose:
        print(f"INFO: Compile[{src}] Loaded {len(library)} procedure(s) from source", file=sys.stderr)

    # invoke exo analysis
    assert isinstance(library, list)
    assert all(isinstance(proc, Procedure) for proc in library)

    module = compile_many(library, opts)

    # print to stdout if no dest
    if not dest:
        print(module)
        return

    # write MLIR to file
    os.makedirs(dest.parent, exist_ok=True)
    dest.write_text(str(module))


def transform(ctx: Context, module: ModuleOp, opts: CompilerOptions = CompilerOptions()) -> ModuleOp:
    """
    Apply transformations to an MLIR module.
    """

    InlineMemorySpacePass().apply(ctx, module)
    module.verify()

    ConvertScalarRefPass().apply(ctx, module)
    module.verify()

    ReconcileIndexCastsPass().apply(ctx, module)
    module.verify()

    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)

    module.verify()

    if opts.target == "exo":
        return module

    if opts.prefix is not None:
        AddPrefixPass(opts.prefix).apply(ctx, module)
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
