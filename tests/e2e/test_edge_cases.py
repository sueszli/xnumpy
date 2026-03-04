from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def add_one_dynamic(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        x[i] = x[i] + 1.0


def test_dynamic_n1():
    assert_match(add_one_dynamic, N=1, x=[5.0])


def test_dynamic_n256():
    N = 256
    assert_match(add_one_dynamic, N=N, x=[float(i) for i in range(N)])


def test_dynamic_n16():
    assert_match(add_one_dynamic, N=16, x=[float(i * 0.5) for i in range(16)])


@proc
def in_place_double(x: f32[8] @ DRAM):
    for i in seq(0, 8):
        x[i] = x[i] + x[i]


def test_in_place_double():
    assert_match(in_place_double, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def in_place_square(x: f32[4] @ DRAM):
    for i in seq(0, 4):
        x[i] = x[i] * x[i]


def test_in_place_square():
    assert_match(in_place_square, x=[2.0, -3.0, 0.0, 0.5])


@proc
def zero_init(out: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = 0.0


def test_zero_init():
    assert_match(zero_init, out=[99.0] * 8)


@proc
def fill_constant(out: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = 42.5


def test_fill_constant():
    assert_match(fill_constant, out=[0.0] * 8)


@proc
def square(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = x[i] * x[i]


def test_negative_float_square():
    assert_match(square, out=[0.0] * 4, x=[-3.0, -1.5, 0.0, 2.5])


@proc
def mixed_arith(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] * b[i] - a[i] + b[i]


def test_mixed_arithmetic():
    assert_match(mixed_arith, out=[0.0] * 4, a=[2.0, -3.0, 0.0, 1.5], b=[4.0, -2.0, 5.0, -1.0])


@proc
def alloc_roundtrip(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp


def test_alloc_roundtrip():
    assert_match(alloc_roundtrip, N=4, x=[1.0, 2.0, 3.0, 4.0])


@proc
def scale_large(N: size, x: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        x[i] = x[i] * s[0]


def test_scale_large():
    N = 512
    assert_match(scale_large, N=N, x=[float(i) for i in range(N)], s=[0.5])


@proc
def split_copy(N: size, even: f32[N] @ DRAM, odd: f32[N] @ DRAM, x: f32[N] @ DRAM):
    for i in seq(0, N):
        if i % 2 == 0:
            even[i] = x[i]
            odd[i] = 0.0
        else:
            even[i] = 0.0
            odd[i] = x[i]


def test_split_copy():
    N = 8
    x = [float(i + 1) for i in range(N)]
    assert_match(split_copy, N=N, even=[0.0] * N, odd=[0.0] * N, x=x)


@proc
def pipeline_3stage(x: f32[8] @ DRAM):
    for i in seq(0, 8):
        x[i] = x[i] + 10.0
    for i in seq(0, 8):
        x[i] = x[i] * 2.0
    for i in seq(0, 8):
        x[i] = x[i] - 1.0


def test_pipeline_3stage():
    assert_match(pipeline_3stage, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
