from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from enum import Enum, auto
from functools import cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.core.LoopIR import LoopIR

from xnumpy.main import compile_procs as xdsl_compile_procs
from xnumpy.patches_llvmlite import jit_compile


class Backend(Enum):
    EXO_C = auto()  # exo's native C codegen -> clang -> .so (reference)
    MLIR = auto()  # xdsl -> mlir-translate + clang -> .so
    JIT = auto()  # xdsl -> llvmlite JIT (in-memory)


_TYPES: dict[str, tuple[type, type]] = {
    "f16": (np.float16, ctypes.c_uint16),
    "f32": (np.float32, ctypes.c_float),
    "f64": (np.float64, ctypes.c_double),
    "i8": (np.int8, ctypes.c_int8),
    "ui8": (np.uint8, ctypes.c_uint8),
    "i16": (np.int16, ctypes.c_int16),
    "ui16": (np.uint16, ctypes.c_uint16),
    "i32": (np.int32, ctypes.c_int32),
}


@cache
def _find_llvm_bin() -> Path:
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    prefix = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()
    return Path(prefix) / "bin"


def _compile_exo_c(procs: list[Procedure]) -> ctypes.CDLL:
    d = Path(tempfile.mkdtemp())
    exo_compile_procs(procs, d, "o.c", "o.h")
    subprocess.run(["clang", "-shared", "-fPIC", "-O0", "-I", str(d), "-o", str(d / "lib.so"), str(d / "o.c")], check=True)
    return ctypes.CDLL(str(d / "lib.so"))


def _compile_xdsl_mlir(procs: list[Procedure]) -> ctypes.CDLL:
    mlir_text = str(xdsl_compile_procs(procs))
    d = Path(tempfile.mkdtemp())
    mlir, so = d / "o.mlir", d / "lib.so"
    mlir.write_text(mlir_text)
    subprocess.run(f"{_find_llvm_bin()}/mlir-translate --mlir-to-llvmir {mlir} | clang -shared -x ir -o {so} -", shell=True, check=True)
    return ctypes.CDLL(str(so))


def _call(lib: ctypes.CDLL, proc_ir: Any, kwargs: dict[str, Any], *, has_ctxt: bool) -> dict[str, np.ndarray]:
    fn = getattr(lib, proc_ir.name)
    argtypes: list = []
    args: list = []
    bufs: dict[str, np.ndarray] = {}

    if has_ctxt:
        argtypes += [ctypes.c_void_p]
        args += [None]

    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]

        match arg.type:
            case LoopIR.Size() | LoopIR.Index():
                argtypes += [ctypes.c_long]
                args += [int(val)]
            case LoopIR.Tensor():
                np_dtype, c_type = _TYPES[str(arg.type.basetype())]
                arr = np.array(val, dtype=np_dtype)
                bufs[name] = arr
                argtypes += [ctypes.POINTER(c_type)]
                args += [arr.ctypes.data_as(ctypes.POINTER(c_type))]

    fn.argtypes, fn.restype = argtypes, None
    fn(*args)
    return bufs


def _call_jit(fns: dict, proc_ir: Any, kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    fn = fns[proc_ir.name]
    args: list = []
    bufs: dict[str, np.ndarray] = {}

    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]

        match arg.type:
            case LoopIR.Size() | LoopIR.Index():
                args += [int(val)]
            case LoopIR.Tensor():
                np_dtype, _ = _TYPES[str(arg.type.basetype())]
                arr = np.array(val, dtype=np_dtype)
                bufs[name] = arr
                args += [arr.ctypes.data]

    fn(*args)
    return bufs


def compile_and_load(proc: Procedure, backend: Backend) -> Callable[..., dict[str, np.ndarray]]:
    # compile a procedure and return a callable.
    # fn(**kwargs) -> {buffer_name: np.ndarray} with mutated output buffers.
    ir = proc._loopir_proc

    match backend:
        case Backend.EXO_C:
            lib = _compile_exo_c([proc])
            return lambda **kwargs: _call(lib, ir, deepcopy(kwargs), has_ctxt=True)

        case Backend.MLIR:
            lib = _compile_xdsl_mlir([proc])
            return lambda **kwargs: _call(lib, ir, deepcopy(kwargs), has_ctxt=False)

        case Backend.JIT:
            fns = jit_compile(xdsl_compile_procs(proc))
            return lambda **kwargs: _call_jit(fns, ir, deepcopy(kwargs))

        case _:
            assert False
