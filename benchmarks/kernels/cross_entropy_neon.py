from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename
from kernels.softmax_neon import neon_add_acc_f32x4, neon_broadcast_f32x4, neon_fmadd_f32x4, neon_loadu_f32x4, neon_mul_f32x4, neon_square_f32x4, neon_storeu_f32x4, neon_sub_f32x4

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


@cache
def cross_entropy_neon(n: int) -> Callable[..., None]:
    assert n % 4 == 0
    n4 = n // 4

    @proc
    def _sum_exp_neon(result: f32[1] @ DRAM, inp: f32[n] @ DRAM, mx: f32[1] @ DRAM):
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
        max_v: f32[4] @ NEON
        neon_broadcast_f32x4(inv32_v, C[0:1])
        neon_broadcast_f32x4(c5_v, C[1:2])
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
            neon_broadcast_f32x4(h, C[2:3])
            neon_fmadd_f32x4(h, c5_v, y)

            g: f32[4] @ NEON
            neon_broadcast_f32x4(g, C[3:4])
            neon_fmadd_f32x4(g, h, y)

            neon_broadcast_f32x4(h, C[4:5])
            neon_fmadd_f32x4(h, g, y)

            neon_broadcast_f32x4(g, C[5:6])
            neon_fmadd_f32x4(g, h, y)

            neon_broadcast_f32x4(h, C[6:7])
            neon_fmadd_f32x4(h, g, y)

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

            neon_add_acc_f32x4(sum_v, sq5)

        neon_storeu_f32x4(sum_buf[0:4], sum_v)
        result[0] = sum_buf[0]
        result[0] += sum_buf[1]
        result[0] += sum_buf[2]
        result[0] += sum_buf[3]

    name = f"_sum_exp_neon_{n}"
    p = rename(_sum_exp_neon, name)
    return compile_jit(p)[name]
