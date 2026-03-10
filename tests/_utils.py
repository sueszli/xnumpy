from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from exo import compile_procs as exo_compile_procs
from exo.API import Procedure
from exo.core.LoopIR import LoopIR
from xdsl.dialects.builtin import ModuleOp

_DTYPES: dict[str, tuple[type, type]] = {
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
def llvm_bin_path() -> Path:
    # resolve llvm bin dir from env, path, or homebrew
    if env := os.environ.get("LLVM_BIN"):
        return Path(env)
    if mlir_opt := shutil.which("mlir-opt"):
        return Path(mlir_opt).parent
    return Path(subprocess.run(["brew", "--prefix", "llvm"], capture_output=True, text=True, check=True).stdout.strip()) / "bin"


def _disk_cache(fn: Callable[..., Path]) -> Callable[..., Path]:
    # like @cache but persists .so to disk across forked processes
    # first arg must be source text (used as cache key)
    cache_dir = Path(tempfile.gettempdir()) / "so_cache"

    def wrapper(source: str, *args, **kwargs) -> Path:
        cache_dir.mkdir(exist_ok=True)
        h = hashlib.sha256(source.encode()).hexdigest()[:16]
        cached = cache_dir / f"{h}.so"
        if cached.exists():
            return cached
        so_path = fn(source, *args, **kwargs)
        tmp = cache_dir / f"{h}.{os.getpid()}.tmp"
        shutil.copy2(so_path, tmp)
        os.replace(str(tmp), str(cached))  # atomic on posix
        return cached

    return wrapper


@_disk_cache
def _build_exo_so(_source: str, build_dir: Path) -> Path:
    subprocess.run(["clang", "-shared", "-fPIC", "-O0", "-I", str(build_dir), "-o", str(build_dir / "lib.so"), str(build_dir / "o.c")], check=True)
    return build_dir / "lib.so"


@_disk_cache
def _build_mlir_so(source: str) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "o.mlir").write_text(source)
    subprocess.run(f"{llvm_bin_path()}/mlir-translate --mlir-to-llvmir {d / 'o.mlir'} | clang -shared -x ir -o {d / 'lib.so'} -", shell=True, check=True)
    return d / "lib.so"


def _exo_bin_path(proc: Procedure) -> Path:
    d = Path(tempfile.mkdtemp())
    exo_compile_procs([proc], d, "o.c", "o.h")
    return _build_exo_so((d / "o.c").read_text(), d)


def _mlir_bin_path(module: ModuleOp) -> Path:
    return _build_mlir_so(str(module))


def _call(fn: Callable, proc_ir: Any, kwargs: dict[str, Any], *, ctx: bool = False) -> dict[str, np.ndarray]:
    # marshal args and invoke a ctypes function (argtypes must be set at load time)
    args: list = [None] if ctx else []
    bufs: dict[str, np.ndarray] = {}
    for arg in proc_ir.args:
        name = re.sub(r"_\d+$", "", str(arg.name))
        val = kwargs[name]
        match arg.type:
            case LoopIR.Size() | LoopIR.Index():
                args.append(int(val))
            case LoopIR.Tensor():
                np_dtype, _ = _DTYPES[str(arg.type.basetype())]
                arr = np.array(val, dtype=np_dtype)
                bufs[name] = arr
                args.append(arr.ctypes.data)
    fn(*args)
    return bufs


def compile_exo(proc: Procedure) -> Callable[..., dict[str, np.ndarray]]:
    # proc -> exo c shared lib -> callable
    proc_ir = proc._loopir_proc
    so_path = _exo_bin_path(proc)
    lib_fn = getattr(ctypes.CDLL(str(so_path)), proc_ir.name)
    lib_fn.argtypes = [ctypes.c_void_p] * (len(proc_ir.args) + 1)  # +1 for exo ctx
    lib_fn.restype = None
    return lambda **kw: _call(lib_fn, proc_ir, deepcopy(kw), ctx=True)


def compile_mlir(proc: Procedure, module: ModuleOp) -> Callable[..., dict[str, np.ndarray]]:
    # proc + mlir module -> shared lib -> callable
    proc_ir = proc._loopir_proc
    so_path = _mlir_bin_path(module)
    lib_fn = getattr(ctypes.CDLL(str(so_path)), proc_ir.name)
    lib_fn.argtypes = [ctypes.c_void_p] * len(proc_ir.args)
    lib_fn.restype = None
    return lambda **kw: _call(lib_fn, proc_ir, deepcopy(kw))
