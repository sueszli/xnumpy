from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import fission, rename, simplify

from xnumpy.main import compile_jit


@proc
def _matvec(M: size, K: size, y: f32[M] @ DRAM, W: f32[M, K] @ DRAM, x: f32[K] @ DRAM):
    for j in seq(0, M):
        y[j] = 0.0
        for i in seq(0, K):
            y[j] += W[j, i] * x[i]


@cache
def matvec_exo(m: int, k: int) -> Callable[..., None]:
    p = _matvec.partial_eval(M=m, K=k)
    p = fission(p, p.find("for i in _: _").before(), n_lifts=1)
    p = simplify(p)
    name = f"_matvec_{m}_{k}"
    return compile_jit(rename(p, name))[name]
