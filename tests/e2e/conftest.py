from __future__ import annotations

import ctypes
import functools
import os
import re
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
from exo import compile_procs as exo_compile_procs
from exo.core.LoopIR import LoopIR

from xdsl_exo.main import compile_procs as xdsl_compile_procs

# exo basetype string -> (numpy dtype, ctypes type)
_TYPES = {
    "f16": (np.float16, ctypes.c_uint16),  # no c_half, pass as raw bits
    "f32": (np.float32, ctypes.c_float),
    "f64": (np.float64, ctypes.c_double),
    "i8": (np.int8, ctypes.c_int8),
    "ui8": (np.uint8, ctypes.c_uint8),
    "i16": (np.int16, ctypes.c_int16),
    "ui16": (np.uint16, ctypes.c_uint16),
    "i32": (np.int32, ctypes.c_int32),
}

_MLIR_SCRIPT = """\
set -euo pipefail
{llvm}/mlir-opt \
  --convert-arith-to-llvm --convert-cf-to-llvm \
  --convert-func-to-llvm --reconcile-unrealized-casts {mlir} \
| {llvm}/mlir-translate --mlir-to-llvmir \
| {llvm}/llc -filetype=obj -o {obj}
clang -shared -o {so} {obj}
"""


@functools.cache
def _find_llvm_bin():
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    prefix = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()
    return Path(prefix) / "bin"


def _compile_exo_c(procs):
    d = Path(tempfile.mkdtemp())
    exo_compile_procs(procs, d, "o.c", "o.h")
    subprocess.run(["clang", "-shared", "-fPIC", "-O2", "-I", str(d), "-o", str(d / "lib.so"), str(d / "o.c")], check=True)
    return ctypes.CDLL(str(d / "lib.so"))


def _compile_xdsl_mlir(procs):
    d = Path(tempfile.mkdtemp())
    (d / "o.mlir").write_text(str(xdsl_compile_procs(procs)))
    subprocess.run(["bash", "-c", _MLIR_SCRIPT.format(llvm=_find_llvm_bin(), mlir=d / "o.mlir", obj=d / "o.o", so=d / "lib.so")], check=True)
    return ctypes.CDLL(str(d / "lib.so"))


def _call(lib, proc_ir, kwargs, *, has_ctxt):
    fn = getattr(lib, proc_ir.name)
    argtypes, args, bufs = [], [], {}

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


def assert_match(proc, **kwargs):
    # compile proc via both exo (C) and xdsl (MLIR), run with kwargs, assert outputs match
    ir = proc._loopir_proc
    exo_bufs = _call(_compile_exo_c([proc]), ir, deepcopy(kwargs), has_ctxt=True)
    xdsl_bufs = _call(_compile_xdsl_mlir([proc]), ir, deepcopy(kwargs), has_ctxt=False)

    for name in exo_bufs:
        e, x = exo_bufs[name], xdsl_bufs[name]
        np.testing.assert_allclose(x, e, atol=1e-6, err_msg=f"mismatch on buffer '{name}'")
