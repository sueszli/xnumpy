from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_add_acc_f32x4({acc_data}, {src_data});")
def neon_add_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] += src[i]


@cache
def weighted_sum_neon(t: int, d: int) -> Callable[..., None]:
    assert d % 4 == 0
    assert t % 4 == 0

    d4 = d // 4
    t4 = t // 4

    @proc
    def _ws_neon(out: f32[d] @ DRAM, weights: f32[t] @ DRAM, V: f32[t, d] @ DRAM):
        for j in seq(0, d):
            out[j] = 0.0

        for jo in seq(0, d4):
            acc0: f32[4] @ NEON
            acc1: f32[4] @ NEON
            acc2: f32[4] @ NEON
            acc3: f32[4] @ NEON
            neon_loadu_f32x4(acc0, out[4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(acc1, out[4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(acc2, out[4 * jo : 4 * jo + 4])
            neon_loadu_f32x4(acc3, out[4 * jo : 4 * jo + 4])
            for to in seq(0, t4):
                w0: f32[4] @ NEON
                w1: f32[4] @ NEON
                w2: f32[4] @ NEON
                w3: f32[4] @ NEON
                v0: f32[4] @ NEON
                v1: f32[4] @ NEON
                v2: f32[4] @ NEON
                v3: f32[4] @ NEON
                neon_broadcast_f32x4(w0, weights[4 * to + 0 : 4 * to + 1])
                neon_broadcast_f32x4(w1, weights[4 * to + 1 : 4 * to + 2])
                neon_broadcast_f32x4(w2, weights[4 * to + 2 : 4 * to + 3])
                neon_broadcast_f32x4(w3, weights[4 * to + 3 : 4 * to + 4])
                neon_loadu_f32x4(v0, V[4 * to + 0, 4 * jo : 4 * jo + 4])
                neon_loadu_f32x4(v1, V[4 * to + 1, 4 * jo : 4 * jo + 4])
                neon_loadu_f32x4(v2, V[4 * to + 2, 4 * jo : 4 * jo + 4])
                neon_loadu_f32x4(v3, V[4 * to + 3, 4 * jo : 4 * jo + 4])
                neon_fmadd_f32x4(acc0, w0, v0)
                neon_fmadd_f32x4(acc1, w1, v1)
                neon_fmadd_f32x4(acc2, w2, v2)
                neon_fmadd_f32x4(acc3, w3, v3)
            neon_add_acc_f32x4(acc0, acc1)
            neon_add_acc_f32x4(acc0, acc2)
            neon_add_acc_f32x4(acc0, acc3)
            neon_storeu_f32x4(out[4 * jo : 4 * jo + 4], acc0)

    name = f"_ws_neon_{t}_{d}"
    p = rename(_ws_neon, name)
    return compile_jit(p)[name]
