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


@cache
def matmul_neon(m: int, k: int, n: int) -> Callable[..., None]:
    assert n % 4 == 0
    assert m % 4 == 0, "required for register blocking"

    n4 = n // 4
    m4 = m // 4
    bk = min(k, 64)
    assert k % bk == 0, f"K={k} must be divisible by BK={bk}"
    n_k_tiles = k // bk

    @proc
    def _mm_neon(C: f32[m, n] @ DRAM, A: f32[m, k] @ DRAM, B: f32[k, n] @ DRAM):
        # zero-init C
        for i in seq(0, m):
            for jo in seq(0, n4):
                C[i, 4 * jo + 0] = 0.0
                C[i, 4 * jo + 1] = 0.0
                C[i, 4 * jo + 2] = 0.0
                C[i, 4 * jo + 3] = 0.0

        # 4-row register-blocked, k-tiled NEON matmul
        for ko in seq(0, n_k_tiles):
            for io in seq(0, m4):
                for jo in seq(0, n4):
                    # 4 row accumulators in NEON registers
                    c0: f32[4] @ NEON
                    c1: f32[4] @ NEON
                    c2: f32[4] @ NEON
                    c3: f32[4] @ NEON
                    neon_loadu_f32x4(c0, C[4 * io + 0, 4 * jo : 4 * jo + 4])
                    neon_loadu_f32x4(c1, C[4 * io + 1, 4 * jo : 4 * jo + 4])
                    neon_loadu_f32x4(c2, C[4 * io + 2, 4 * jo : 4 * jo + 4])
                    neon_loadu_f32x4(c3, C[4 * io + 3, 4 * jo : 4 * jo + 4])
                    for ki in seq(0, bk):
                        # load B row once, reuse across 4 i-rows
                        b_vec: f32[4] @ NEON
                        neon_loadu_f32x4(b_vec, B[bk * ko + ki, 4 * jo : 4 * jo + 4])
                        a0: f32[4] @ NEON
                        a1: f32[4] @ NEON
                        a2: f32[4] @ NEON
                        a3: f32[4] @ NEON
                        neon_broadcast_f32x4(a0, A[4 * io + 0, bk * ko + ki : bk * ko + ki + 1])
                        neon_broadcast_f32x4(a1, A[4 * io + 1, bk * ko + ki : bk * ko + ki + 1])
                        neon_broadcast_f32x4(a2, A[4 * io + 2, bk * ko + ki : bk * ko + ki + 1])
                        neon_broadcast_f32x4(a3, A[4 * io + 3, bk * ko + ki : bk * ko + ki + 1])
                        neon_fmadd_f32x4(c0, a0, b_vec)
                        neon_fmadd_f32x4(c1, a1, b_vec)
                        neon_fmadd_f32x4(c2, a2, b_vec)
                        neon_fmadd_f32x4(c3, a3, b_vec)
                    neon_storeu_f32x4(C[4 * io + 0, 4 * jo : 4 * jo + 4], c0)
                    neon_storeu_f32x4(C[4 * io + 1, 4 * jo : 4 * jo + 4], c1)
                    neon_storeu_f32x4(C[4 * io + 2, 4 * jo : 4 * jo + 4], c2)
                    neon_storeu_f32x4(C[4 * io + 3, 4 * jo : 4 * jo + 4], c3)

    name = f"_mm_neon_{m}_{k}_{n}"
    p = rename(_mm_neon, name)
    return compile_jit(p)[name]
