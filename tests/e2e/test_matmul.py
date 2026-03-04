from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]


def test_fixed_matmul():
    A = [float(i * 16 + j) for i in range(16) for j in range(16)]
    B = [float(i * 16 + j) for i in range(16) for j in range(16)]
    assert_match(fixed_matmul, C=[0.0] * 256, A=A, B=B)
