from __future__ import annotations

import gc as _gc
from copy import deepcopy
from typing import Any

import numpy as np
from _utils import _call, compile_exo, compile_mlir
from exo.API import Procedure

from xnumpy.main import compile_jit, to_mlir

# xdsl irdl holds raw ctypes pointers. gc finalizer ordering -> dangling ptr -> segfault
_gc.disable()
_gc.set_threshold(0)
_gc.enable = lambda: None
_gc.collect = lambda *a, **kw: 0


def assert_match(proc: Procedure, **kwargs: Any) -> None:
    # compile proc on all backends, verify outputs match exo_c (reference)
    ir = proc._loopir_proc
    module = to_mlir(proc)
    jit_fn = compile_jit(proc)[ir.name]
    results = {
        "exo_c": compile_exo(proc)(**kwargs),
        "xdsl_mlir": compile_mlir(proc, module)(**kwargs),
        "jit": _call(jit_fn, ir, deepcopy(kwargs)),
    }

    ref = results["exo_c"]
    for key in ("xdsl_mlir", "jit"):
        for name in ref:
            np.testing.assert_allclose(results[key][name], ref[name], atol=1e-6, err_msg=f"{key} mismatch on buffer '{name}'")
