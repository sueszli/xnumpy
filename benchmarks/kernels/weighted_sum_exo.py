from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit


@proc
def _weighted_sum(T: size, D: size, out: f32[D] @ DRAM, weights: f32[T] @ DRAM, V: f32[T, D] @ DRAM):
    for j in seq(0, D):
        out[j] = 0.0
    for t in seq(0, T):
        for j in seq(0, D):
            out[j] += weights[t] * V[t, j]


@cache
def weighted_sum_exo(t: int, d: int) -> Callable[..., None]:
    p = _weighted_sum.partial_eval(T=t, D=d)
    p = simplify(p)
    name = f"_weighted_sum_{t}_{d}"
    return compile_jit(rename(p, name))[name]
