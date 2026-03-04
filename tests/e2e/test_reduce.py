from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def reduce_f32(x: f32[8] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


@proc
def reduce_i32(x: i32[8] @ DRAM, y: i32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


@proc
def reduce_f64(x: f64[8] @ DRAM, y: f64[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


def test_reduce_f32():
    assert_match(reduce_f32, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], y=[0.0])


def test_reduce_f32_with_negatives():
    assert_match(reduce_f32, x=[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], y=[0.0])


def test_reduce_f32_nonzero_init():
    assert_match(reduce_f32, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], y=[100.0])


def test_reduce_i32():
    assert_match(reduce_i32, x=[1, 2, 3, 4, 5, 6, 7, 8], y=[0])


def test_reduce_f64():
    assert_match(reduce_f64, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], y=[0.0])


@proc
def dot_f32(N: size, out: f32[1] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[0] += a[i] * b[i]


def test_dot_f32():
    assert_match(dot_f32, N=4, out=[0.0], a=[1.0, 2.0, 3.0, 4.0], b=[5.0, 6.0, 7.0, 8.0])


def test_dot_f32_large():
    N = 64
    a = [float(i) for i in range(N)]
    b = [float(N - i) for i in range(N)]
    assert_match(dot_f32, N=N, out=[0.0], a=a, b=b)


@proc
def reduce_dynamic(N: size, x: f32[N] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, N):
        y[0] += x[i]


def test_reduce_dynamic_small():
    assert_match(reduce_dynamic, N=1, x=[42.0], y=[0.0])


def test_reduce_dynamic_large():
    N = 128
    assert_match(reduce_dynamic, N=N, x=[1.0] * N, y=[0.0])
