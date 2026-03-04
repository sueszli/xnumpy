from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def copy_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]


@proc
def transpose_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[j, i] = src[i, j]


@proc
def add_2d(dst: f32[4, 4] @ DRAM, a: f32[4, 4] @ DRAM, b: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = a[i, j] + b[i, j]


@proc
def scale_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j] * s[0]


@proc
def row_sum(out: f32[4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            out[i] += src[i, j]


def test_copy_2d():
    src = [float(x) for x in range(16)]
    assert_match(copy_2d, dst=[0.0] * 16, src=src)


def test_transpose_2d():
    src = [float(x) for x in range(16)]
    assert_match(transpose_2d, dst=[0.0] * 16, src=src)


def test_add_2d():
    a = [float(x) for x in range(16)]
    b = [float(16 - x) for x in range(16)]
    assert_match(add_2d, dst=[0.0] * 16, a=a, b=b)


def test_scale_2d():
    src = [float(x) for x in range(16)]
    assert_match(scale_2d, dst=[0.0] * 16, src=src, s=[2.5])


def test_row_sum():
    src = [float(i * 4 + j) for i in range(4) for j in range(4)]
    assert_match(row_sum, out=[0.0] * 4, src=src)
