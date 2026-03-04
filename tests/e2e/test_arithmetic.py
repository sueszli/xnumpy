from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


@proc
def int_arithmetic(out: i32[1] @ DRAM, a: i32[1] @ DRAM, b: i32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]


def test_float_arithmetic():
    assert_match(float_arithmetic, out=[0.0], a=[10.0], b=[3.0])


def test_f64_arithmetic():
    assert_match(f64_arithmetic, out=[0.0], a=[10.0], b=[3.0])


def test_int_arithmetic():
    assert_match(int_arithmetic, out=[0], a=[10], b=[3])
