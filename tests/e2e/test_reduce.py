from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def reduce_float(x: f32[8] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


@proc
def reduce_int(x: i32[8] @ DRAM, y: i32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


def test_reduce_float():
    assert_match(reduce_float, x=[1, 2, 3, 4, 5, 6, 7, 8], y=[0.0])


def test_reduce_int():
    assert_match(reduce_int, x=[1, 2, 3, 4, 5, 6, 7, 8], y=[0])
