from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def f32_add(out: f32[8] @ DRAM, a: f32[8] @ DRAM, b: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = a[i] + b[i]


@proc
def f32_sub(out: f32[8] @ DRAM, a: f32[8] @ DRAM, b: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = a[i] - b[i]


@proc
def f32_mul(out: f32[8] @ DRAM, a: f32[8] @ DRAM, b: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = a[i] * b[i]


@proc
def f32_div(out: f32[8] @ DRAM, a: f32[8] @ DRAM, b: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = a[i] / b[i]


@proc
def f32_neg(out: f32[4] @ DRAM, a: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = -a[i]


def test_f32_add():
    assert_match(f32_add, out=[0.0] * 8, a=[1.0, -2.0, 0.0, 3.5, 1e6, -1e6, 0.001, 100.0], b=[10.0, 20.0, 0.0, -3.5, 1.0, 1e6, -0.001, -100.0])


def test_f32_sub():
    assert_match(f32_sub, out=[0.0] * 8, a=[10.0, 0.0, -5.0, 3.5, 1e6, 1.0, 0.0, 42.0], b=[3.0, 0.0, -5.0, 7.0, 1e6, -1.0, 100.0, 42.0])


def test_f32_mul():
    assert_match(f32_mul, out=[0.0] * 8, a=[2.0, -3.0, 0.0, 1.5, 1000.0, -1.0, 0.5, -0.5], b=[4.0, -2.0, 999.0, -1.0, 1000.0, -1.0, 0.5, -0.5])


def test_f32_div():
    assert_match(f32_div, out=[0.0] * 8, a=[10.0, -6.0, 0.0, 1.0, 100.0, -100.0, 7.0, 1.0], b=[2.0, 3.0, 1.0, 3.0, 10.0, -10.0, 7.0, -1.0])


def test_f32_neg():
    assert_match(f32_neg, out=[0.0] * 4, a=[42.0, -42.0, 0.0, -0.0])


@proc
def f64_add(out: f64[4] @ DRAM, a: f64[4] @ DRAM, b: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] + b[i]


@proc
def f64_sub(out: f64[4] @ DRAM, a: f64[4] @ DRAM, b: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] - b[i]


@proc
def f64_mul(out: f64[4] @ DRAM, a: f64[4] @ DRAM, b: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] * b[i]


@proc
def f64_div(out: f64[4] @ DRAM, a: f64[4] @ DRAM, b: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] / b[i]


@proc
def f64_neg(out: f64[4] @ DRAM, a: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = -a[i]


def test_f64_add():
    assert_match(f64_add, out=[0.0] * 4, a=[1.0, -1e15, 0.0, 3.141592653589793], b=[2.0, 1e15, 0.0, -3.141592653589793])


def test_f64_sub():
    assert_match(f64_sub, out=[0.0] * 4, a=[10.0, 1e15, 0.0, 1.0], b=[3.0, 1e15, 0.0, 0.3])


def test_f64_mul():
    assert_match(f64_mul, out=[0.0] * 4, a=[2.0, -3.0, 0.0, 1e8], b=[4.0, -2.0, 1e10, 1e8])


def test_f64_div():
    assert_match(f64_div, out=[0.0] * 4, a=[10.0, -6.0, 0.0, 1.0], b=[3.0, 3.0, 1.0, 7.0])


def test_f64_neg():
    assert_match(f64_neg, out=[0.0] * 4, a=[1.0, -1.0, 0.0, 1e100])


@proc
def i32_add(out: i32[4] @ DRAM, a: i32[4] @ DRAM, b: i32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] + b[i]


@proc
def i32_sub(out: i32[4] @ DRAM, a: i32[4] @ DRAM, b: i32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] - b[i]


@proc
def i32_mul(out: i32[4] @ DRAM, a: i32[4] @ DRAM, b: i32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] * b[i]


@proc
def i32_div(out: i32[4] @ DRAM, a: i32[4] @ DRAM, b: i32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] / b[i]


@proc
def i32_neg(out: i32[4] @ DRAM, a: i32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = -a[i]


def test_i32_add():
    assert_match(i32_add, out=[0] * 4, a=[10, -10, 0, 2147483], b=[3, 10, 0, -2147483])


def test_i32_sub():
    assert_match(i32_sub, out=[0] * 4, a=[10, 0, -5, 100], b=[3, 0, -5, 100])


def test_i32_mul():
    assert_match(i32_mul, out=[0] * 4, a=[3, -4, 0, 100], b=[7, -5, 999, -1])


def test_i32_div():
    assert_match(i32_div, out=[0] * 4, a=[10, -7, 0, 100], b=[3, 2, 1, -3])


def test_i32_neg():
    assert_match(i32_neg, out=[0] * 4, a=[1, -1, 0, 42])


@proc
def f32_fma_pattern(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM, c: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = a[i] * b[i] + c[i]


@proc
def f32_complex_expr(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = (a[i] + b[i]) * (a[i] - b[i])


def test_f32_fma():
    assert_match(f32_fma_pattern, out=[0.0] * 4, a=[1.0, 2.0, 3.0, 0.0], b=[4.0, 5.0, 6.0, 10.0], c=[7.0, 8.0, 9.0, -1.0])


def test_f32_difference_of_squares():
    assert_match(f32_complex_expr, out=[0.0] * 4, a=[5.0, 3.0, 10.0, 0.0], b=[3.0, 3.0, 1.0, 7.0])
