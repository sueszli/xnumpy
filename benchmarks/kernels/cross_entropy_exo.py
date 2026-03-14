from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit
from xnumpy.patches_exo import Stack


@proc
def _sum_exp(N: size, result: f32[1], inp: f32[N], mx: f32[1]):
    sum_val: f32 @ Stack
    t: f32 @ Stack
    y: f32 @ Stack
    e5: f32 @ Stack
    e4: f32 @ Stack
    e3: f32 @ Stack
    e2: f32 @ Stack
    e1: f32 @ Stack
    s1: f32 @ Stack
    s2: f32 @ Stack
    s3: f32 @ Stack
    s4: f32 @ Stack
    s5: f32 @ Stack

    sum_val = 0.0
    for j in seq(0, N):
        t = inp[j] - mx[0]
        y = t * 0.03125
        e5 = y * 0.008333333 + 0.041666667
        e4 = e5 * y + 0.166666667
        e3 = e4 * y + 0.5
        e2 = e3 * y + 1.0
        e1 = e2 * y + 1.0
        s1 = e1 * e1
        s2 = s1 * s1
        s3 = s2 * s2
        s4 = s3 * s3
        s5 = s4 * s4
        sum_val += s5

    result[0] = sum_val


@cache
def _jit_sum_exp(n: int) -> Callable[..., None]:
    p = _sum_exp.partial_eval(N=n)
    p = simplify(p)
    name = f"_sum_exp_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def cross_entropy_exo(n: int) -> tuple[Callable[..., None], Callable[..., None]]:
    from kernels.softmax_exo import _jit_max, _jit_max_neon

    max_fn = _jit_max_neon(n) if n % 8 == 0 else _jit_max(n)
    return max_fn, _jit_sum_exp(n)
