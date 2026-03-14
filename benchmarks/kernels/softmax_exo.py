from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import select
from exo.stdlib.scheduling import rename, simplify
from kernels.softmax_neon import neon_loadu_f32x4, neon_storeu_f32x4

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON, Stack


@instr("neon_fmax_acc_f32x4({acc_data}, {src_data});")
def neon_fmax_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = select(acc[i], src[i], src[i], acc[i])


@cache
def _jit_max_neon(n: int) -> Callable[..., None]:
    assert n % 8 == 0
    n8 = n // 8

    @proc
    def _find_max_neon(result: f32[1] @ DRAM, inp: f32[n] @ DRAM):
        acc0: f32[4] @ NEON
        acc1: f32[4] @ NEON
        neon_loadu_f32x4(acc0, inp[0:4])
        neon_loadu_f32x4(acc1, inp[4:8])

        for i in seq(0, n8):
            c0: f32[4] @ NEON
            c1: f32[4] @ NEON
            neon_loadu_f32x4(c0, inp[8 * i : 8 * i + 4])
            neon_loadu_f32x4(c1, inp[8 * i + 4 : 8 * i + 8])
            neon_fmax_acc_f32x4(acc0, c0)
            neon_fmax_acc_f32x4(acc1, c1)

        neon_fmax_acc_f32x4(acc0, acc1)

        buf: f32[4] @ DRAM
        neon_storeu_f32x4(buf[0:4], acc0)
        m0: f32 @ Stack
        m1: f32 @ Stack
        m0 = select(buf[0], buf[1], buf[1], buf[0])
        m1 = select(buf[2], buf[3], buf[3], buf[2])
        result[0] = select(m0, m1, m1, m0)

    name = f"_find_max_neon_{n}"
    p = rename(_find_max_neon, name)
    return compile_jit(p)[name]


@proc
def _find_max(N: size, result: f32[1], inp: f32[N]):
    acc: f32 @ Stack
    acc = inp[0]
    for i in seq(0, N):
        acc = select(acc, inp[i], inp[i], acc)
    result[0] = acc


@cache
def _jit_max(n: int) -> Callable[..., None]:
    p = _find_max.partial_eval(N=n)
    p = simplify(p)
    name = f"_find_max_{n}"
    return compile_jit(rename(p, name))[name]


@proc
def _softmax_core(N: size, out: f32[N], inp: f32[N], mx: f32[1]):
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
        out[j] = s5
        sum_val += s5

    for k in seq(0, N):
        out[k] = out[k] / sum_val


@cache
def _jit_core(n: int) -> Callable[..., None]:
    p = _softmax_core.partial_eval(N=n)
    p = simplify(p)
    name = f"_softmax_core_{n}"
    return compile_jit(rename(p, name))[name]


@cache
def softmax_exo(n: int) -> tuple[Callable[..., None], Callable[..., None]]:
    max_fn = _jit_max_neon(n) if n % 4 == 0 else _jit_max(n)
    return max_fn, _jit_core(n)
