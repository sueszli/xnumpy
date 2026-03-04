from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def even_odd(out: f32[8] @ DRAM):
    for i in seq(0, 8):
        if i % 2 == 0:
            out[i] = 1.0
        else:
            out[i] = 0.0


def test_even_odd():
    assert_match(even_odd, out=[0.0] * 8)


@proc
def mod3_pattern(out: f32[12] @ DRAM):
    for i in seq(0, 12):
        if i % 3 == 0:
            out[i] = 3.0
        else:
            if i % 3 == 1:
                out[i] = 1.0
            else:
                out[i] = 2.0


def test_mod3_pattern():
    assert_match(mod3_pattern, out=[0.0] * 12)


@proc
def stride_access(src: f32[16] @ DRAM, dst: f32[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = src[i * 4]


def test_stride_access():
    src = [float(x) for x in range(16)]
    assert_match(stride_access, src=src, dst=[0.0] * 4)


@proc
def checkerboard(out: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            if (i + j) % 2 == 0:
                out[i, j] = 1.0
            else:
                out[i, j] = 0.0


def test_checkerboard():
    assert_match(checkerboard, out=[0.0] * 16)


@proc
def mod5_flag(out: f32[20] @ DRAM):
    for i in seq(0, 20):
        if i % 5 == 0:
            out[i] = 1.0
        else:
            out[i] = 0.0


def test_mod5_flag():
    assert_match(mod5_flag, out=[0.0] * 20)
