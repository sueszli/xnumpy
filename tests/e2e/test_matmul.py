from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def matmul_4x4(C: f32[4, 4] @ DRAM, A: f32[4, 4] @ DRAM, B: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            C[i, j] = 0.0
            for k in seq(0, 4):
                C[i, j] += A[i, k] * B[k, j]


@proc
def matmul_16x16(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]


def test_matmul_identity():
    eye = [0.0] * 16
    for i in range(4):
        eye[i * 4 + i] = 1.0
    A = [float(i * 4 + j + 1) for i in range(4) for j in range(4)]
    assert_match(matmul_4x4, C=[0.0] * 16, A=A, B=eye)


def test_matmul_4x4():
    A = [float(i * 4 + j + 1) for i in range(4) for j in range(4)]
    B = [float((i + j) % 4 + 1) for i in range(4) for j in range(4)]
    assert_match(matmul_4x4, C=[0.0] * 16, A=A, B=B)


def test_matmul_16x16():
    A = [float(i * 16 + j) for i in range(16) for j in range(16)]
    B = [float(i * 16 + j) for i in range(16) for j in range(16)]
    assert_match(matmul_16x16, C=[0.0] * 256, A=A, B=B)
