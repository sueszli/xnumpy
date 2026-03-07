from __future__ import annotations

import ctypes
import fcntl
import functools
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.core.LoopIR import LoopIR

from xdsl_exo.main import compile_procs as xdsl_compile_procs
from xdsl_exo.patches_llvmlite import jit_compile

#
# benchmarking
#


_BENCH_DIR = Path(tempfile.gettempdir()) / "xdsl_exo_bench"
_BENCH_JSONL = _BENCH_DIR / "timings.jsonl"


def _timed(fn, *args, **kw):
    t0 = time.perf_counter()
    result = fn(*args, **kw)
    return result, time.perf_counter() - t0


def _record_timing(kernel: str, exo_c: float, xdsl_mlir: float, jit: float) -> None:
    _BENCH_DIR.mkdir(exist_ok=True)
    row = json.dumps({"kernel": kernel, "exo_c": exo_c, "xdsl_mlir": xdsl_mlir, "jit": jit})
    with open(_BENCH_JSONL, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(row + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


#
# testing
#


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


@functools.cache
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

        if isinstance(arg.type, (LoopIR.Size, LoopIR.Index)):
            argtypes += [ctypes.c_long]
            args += [int(val)]
        elif isinstance(arg.type, LoopIR.Tensor):
            np_dtype, c_type = _TYPES[str(arg.type.basetype())]
            arr = np.array(val, dtype=np_dtype)
            bufs[name] = arr
            argtypes += [ctypes.POINTER(c_type)]
            args += [arr.ctypes.data_as(ctypes.POINTER(c_type))]

    fn.argtypes, fn.restype = argtypes, None
    fn(*args)
    return bufs


def _call_jit(engine, proc_ir: Any, kwargs: dict[str, Any]) -> dict[str, np.ndarray]:
    addr = engine.get_function_address(proc_ir.name)
    argtypes: list = []
    args: list = []
    bufs: dict[str, np.ndarray] = {}

    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]

        if isinstance(arg.type, (LoopIR.Size, LoopIR.Index)):
            argtypes += [ctypes.c_int64]
            args += [int(val)]
        elif isinstance(arg.type, LoopIR.Tensor):
            np_dtype, _ = _TYPES[str(arg.type.basetype())]
            arr = np.array(val, dtype=np_dtype)
            bufs[name] = arr
            argtypes += [ctypes.c_void_p]
            args += [arr.ctypes.data]

    cfunc = ctypes.CFUNCTYPE(None, *argtypes)(addr)
    cfunc(*args)
    return bufs


def assert_match(proc: Procedure, **kwargs: Any) -> None:
    ir = proc._loopir_proc

    exo_lib = _compile_exo_c([proc])
    exo_bufs, t_exo = _timed(_call, exo_lib, ir, deepcopy(kwargs), has_ctxt=True)

    xdsl_lib = _compile_xdsl_mlir([proc])
    xdsl_bufs, t_xdsl = _timed(_call, xdsl_lib, ir, deepcopy(kwargs), has_ctxt=False)

    jit_engine = jit_compile(xdsl_compile_procs([proc]))
    jit_bufs, t_jit = _timed(_call_jit, jit_engine, ir, deepcopy(kwargs))

    _record_timing(ir.name, t_exo, t_xdsl, t_jit)

    for name in exo_bufs:
        e, x, j = exo_bufs[name], xdsl_bufs[name], jit_bufs[name]
        np.testing.assert_allclose(x, e, atol=1e-6, err_msg=f"xdsl mismatch on buffer '{name}'")
        np.testing.assert_allclose(j, e, atol=1e-6, err_msg=f"jit mismatch on buffer '{name}'")
