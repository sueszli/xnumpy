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


@instr("neon_sub_f32x4({dst_data}, {a_data}, {b_data});")
def neon_sub_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] - b[i]


@instr("neon_mul_f32x4({dst_data}, {a_data}, {b_data});")
def neon_mul_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("neon_square_f32x4({dst_data}, {src_data});")
def neon_square_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i] * src[i]


@instr("neon_add_acc_f32x4({acc_data}, {src_data});")
def neon_add_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] += src[i]


@cache
def softmax_neon(n: int) -> Callable[..., None]:
    assert n % 4 == 0
    n4 = n // 4

    @proc
    def _softmax_neon(out: f32[n] @ DRAM, inp: f32[n] @ DRAM, mx: f32[1] @ DRAM):
        C: f32[7] @ DRAM
        C[0] = 0.03125
        C[1] = 0.008333333
        C[2] = 0.041666667
        C[3] = 0.166666667
        C[4] = 0.5
        C[5] = 1.0
        C[6] = 1.0

        inv32_v: f32[4] @ NEON
        c5_v: f32[4] @ NEON
        c4_v: f32[4] @ NEON
        c3_v: f32[4] @ NEON
        c2_v: f32[4] @ NEON
        c1_v: f32[4] @ NEON
        c0_v: f32[4] @ NEON
        max_v: f32[4] @ NEON
        neon_broadcast_f32x4(inv32_v, C[0:1])
        neon_broadcast_f32x4(c5_v, C[1:2])
        neon_broadcast_f32x4(c4_v, C[2:3])
        neon_broadcast_f32x4(c3_v, C[3:4])
        neon_broadcast_f32x4(c2_v, C[4:5])
        neon_broadcast_f32x4(c1_v, C[5:6])
        neon_broadcast_f32x4(c0_v, C[6:7])
        neon_broadcast_f32x4(max_v, mx[0:1])

        sum_buf: f32[4] @ DRAM
        sum_buf[0] = 0.0
        sum_buf[1] = 0.0
        sum_buf[2] = 0.0
        sum_buf[3] = 0.0
        sum_v: f32[4] @ NEON
        neon_loadu_f32x4(sum_v, sum_buf[0:4])

        for i in seq(0, n4):
            x: f32[4] @ NEON
            neon_loadu_f32x4(x, inp[4 * i : 4 * i + 4])

            t: f32[4] @ NEON
            neon_sub_f32x4(t, x, max_v)

            y: f32[4] @ NEON
            neon_mul_f32x4(y, t, inv32_v)

            h: f32[4] @ NEON
            neon_mul_f32x4(h, c5_v, y)
            neon_add_acc_f32x4(h, c4_v)

            g: f32[4] @ NEON
            neon_mul_f32x4(g, h, y)
            neon_add_acc_f32x4(g, c3_v)

            neon_mul_f32x4(h, g, y)
            neon_add_acc_f32x4(h, c2_v)

            neon_mul_f32x4(g, h, y)
            neon_add_acc_f32x4(g, c1_v)

            neon_mul_f32x4(h, g, y)
            neon_add_acc_f32x4(h, c0_v)

            sq1: f32[4] @ NEON
            neon_square_f32x4(sq1, h)
            sq2: f32[4] @ NEON
            neon_square_f32x4(sq2, sq1)
            sq3: f32[4] @ NEON
            neon_square_f32x4(sq3, sq2)
            sq4: f32[4] @ NEON
            neon_square_f32x4(sq4, sq3)
            sq5: f32[4] @ NEON
            neon_square_f32x4(sq5, sq4)

            neon_storeu_f32x4(out[4 * i : 4 * i + 4], sq5)
            neon_add_acc_f32x4(sum_v, sq5)

        neon_storeu_f32x4(sum_buf[0:4], sum_v)
        total: f32 @ DRAM
        total = sum_buf[0]
        total += sum_buf[1]
        total += sum_buf[2]
        total += sum_buf[3]

        inv_s: f32[1] @ DRAM
        inv_s[0] = 1.0 / total
        inv_v: f32[4] @ NEON
        neon_broadcast_f32x4(inv_v, inv_s[0:1])

        for j in seq(0, n4):
            v: f32[4] @ NEON
            neon_loadu_f32x4(v, out[4 * j : 4 * j + 4])
            r: f32[4] @ NEON
            neon_mul_f32x4(r, v, inv_v)
            neon_storeu_f32x4(out[4 * j : 4 * j + 4], r)

    name = f"_softmax_neon_{n}"
    p = rename(_softmax_neon, name)
    return compile_jit(p)[name]
