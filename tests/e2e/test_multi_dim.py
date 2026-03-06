from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def copy_3d(dst: f32[2, 3, 4] @ DRAM, src: f32[2, 3, 4] @ DRAM):
    for i in seq(0, 2):
        for j in seq(0, 3):
            for k in seq(0, 4):
                dst[i, j, k] = src[i, j, k]


@proc
def sum_3d(out: f32[1] @ DRAM, src: f32[2, 3, 4] @ DRAM):
    for i in seq(0, 2):
        for j in seq(0, 3):
            for k in seq(0, 4):
                out[0] += src[i, j, k]


def test_copy_3d():
    src = [float(x) for x in range(24)]
    assert_match(copy_3d, dst=[0.0] * 24, src=src)


def test_sum_3d():
    src = [float(x) for x in range(24)]
    assert_match(sum_3d, out=[0.0], src=src)


@proc
def col_sum(out: f32[4] @ DRAM, src: f32[4, 4] @ DRAM):
    for j in seq(0, 4):
        for i in seq(0, 4):
            out[j] += src[i, j]


def test_col_sum():
    src = [float(i * 4 + j) for i in range(4) for j in range(4)]
    assert_match(col_sum, out=[0.0] * 4, src=src)


@proc
def scale_3d(dst: f32[2, 3, 4] @ DRAM, src: f32[2, 3, 4] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, 2):
        for j in seq(0, 3):
            for k in seq(0, 4):
                dst[i, j, k] = src[i, j, k] * s[0]


def test_scale_3d():
    src = [float(x) for x in range(24)]
    assert_match(scale_3d, dst=[0.0] * 24, src=src, s=[3.0])


@proc
def dynamic_2d_copy(M: size, N: size, dst: f32[M, N] @ DRAM, src: f32[M, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            dst[i, j] = src[i, j]


def test_dynamic_2d_3x4():
    src = [float(x) for x in range(12)]
    assert_match(dynamic_2d_copy, M=3, N=4, dst=[0.0] * 12, src=src)


def test_dynamic_2d_1x1():
    assert_match(dynamic_2d_copy, M=1, N=1, dst=[0.0], src=[42.0])


def test_dynamic_2d_1x8():
    src = [float(x) for x in range(8)]
    assert_match(dynamic_2d_copy, M=1, N=8, dst=[0.0] * 8, src=src)


def test_dynamic_2d_5x7():
    src = [float(x) for x in range(35)]
    assert_match(dynamic_2d_copy, M=5, N=7, dst=[0.0] * 35, src=src)


@proc
def dynamic_2d_scale(M: size, N: size, dst: f32[M, N] @ DRAM, src: f32[M, N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            dst[i, j] = src[i, j] * s[0]


def test_dynamic_2d_scale():
    src = [float(x) for x in range(12)]
    assert_match(dynamic_2d_scale, M=3, N=4, dst=[0.0] * 12, src=src, s=[2.0])
