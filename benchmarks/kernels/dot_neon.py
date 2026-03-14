from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.stdlib.scheduling import rename
from kernels.softmax_neon import neon_add_acc_f32x4, neon_fmadd_f32x4, neon_loadu_f32x4, neon_storeu_f32x4

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


@cache
def dot_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    n16 = n // 16

    @proc
    def _dot_neon(result: f32[1] @ DRAM, q: f32[n] @ DRAM, k: f32[n] @ DRAM):
        zero_buf: f32[4] @ DRAM
        zero_buf[0] = 0.0
        zero_buf[1] = 0.0
        zero_buf[2] = 0.0
        zero_buf[3] = 0.0

        acc0: f32[4] @ NEON
        acc1: f32[4] @ NEON
        acc2: f32[4] @ NEON
        acc3: f32[4] @ NEON
        neon_loadu_f32x4(acc0, zero_buf[0:4])
        neon_loadu_f32x4(acc1, zero_buf[0:4])
        neon_loadu_f32x4(acc2, zero_buf[0:4])
        neon_loadu_f32x4(acc3, zero_buf[0:4])

        for i in seq(0, n16):
            q0: f32[4] @ NEON
            q1: f32[4] @ NEON
            q2: f32[4] @ NEON
            q3: f32[4] @ NEON
            k0: f32[4] @ NEON
            k1: f32[4] @ NEON
            k2: f32[4] @ NEON
            k3: f32[4] @ NEON
            neon_loadu_f32x4(q0, q[16 * i : 16 * i + 4])
            neon_loadu_f32x4(q1, q[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(q2, q[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(q3, q[16 * i + 12 : 16 * i + 16])
            neon_loadu_f32x4(k0, k[16 * i : 16 * i + 4])
            neon_loadu_f32x4(k1, k[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(k2, k[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(k3, k[16 * i + 12 : 16 * i + 16])
            neon_fmadd_f32x4(acc0, q0, k0)
            neon_fmadd_f32x4(acc1, q1, k1)
            neon_fmadd_f32x4(acc2, q2, k2)
            neon_fmadd_f32x4(acc3, q3, k3)

        neon_add_acc_f32x4(acc0, acc1)
        neon_add_acc_f32x4(acc2, acc3)
        neon_add_acc_f32x4(acc0, acc2)

        sum_buf: f32[4] @ DRAM
        neon_storeu_f32x4(sum_buf[0:4], acc0)
        total: f32 @ DRAM
        total = sum_buf[0]
        total += sum_buf[1]
        total += sum_buf[2]
        total += sum_buf[3]
        result[0] = total

    name = f"_dot_neon_{n}"
    p = rename(_dot_neon, name)
    return compile_jit(p)[name]
