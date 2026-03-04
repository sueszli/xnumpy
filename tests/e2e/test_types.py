from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def f64_copy(dst: f64[8] @ DRAM, src: f64[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


@proc
def f64_reduce(x: f64[8] @ DRAM, y: f64[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]


@proc
def f64_scale(out: f64[4] @ DRAM, x: f64[4] @ DRAM, s: f64[1] @ DRAM):
    for i in seq(0, 4):
        out[i] = x[i] * s[0]


def test_f64_copy():
    assert_match(f64_copy, dst=[0.0] * 8, src=[1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8])


def test_f64_reduce():
    assert_match(f64_reduce, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], y=[0.0])


def test_f64_scale():
    assert_match(f64_scale, out=[0.0] * 4, x=[1.0, -2.0, 0.0, 100.0], s=[3.0])


@proc
def ui8_copy(dst: ui8[8] @ DRAM, src: ui8[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


@proc
def ui8_add(dst: ui8[4] @ DRAM, a: ui8[4] @ DRAM, b: ui8[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


def test_ui8_copy():
    assert_match(ui8_copy, dst=[0] * 8, src=[0, 1, 127, 128, 200, 254, 255, 42])


def test_ui8_add():
    assert_match(ui8_add, dst=[0] * 4, a=[1, 10, 100, 0], b=[2, 20, 50, 0])


@proc
def ui16_copy(dst: ui16[8] @ DRAM, src: ui16[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


@proc
def ui16_add(dst: ui16[4] @ DRAM, a: ui16[4] @ DRAM, b: ui16[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


def test_ui16_copy():
    assert_match(ui16_copy, dst=[0] * 8, src=[0, 1, 256, 1000, 30000, 50000, 65534, 65535])


def test_ui16_add():
    assert_match(ui16_add, dst=[0] * 4, a=[100, 1000, 0, 30000], b=[200, 2000, 0, 1000])


@proc
def i8_copy(dst: i8[8] @ DRAM, src: i8[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


@proc
def i8_add(dst: i8[4] @ DRAM, a: i8[4] @ DRAM, b: i8[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@proc
def i8_negate(dst: i8[4] @ DRAM, src: i8[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = -src[i]


@proc
def i8_mul(dst: i8[4] @ DRAM, a: i8[4] @ DRAM, b: i8[4] @ DRAM):
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


def test_i8_copy():
    assert_match(i8_copy, dst=[0] * 8, src=[-128, -1, 0, 1, 42, 100, 126, 127])


def test_i8_add():
    assert_match(i8_add, dst=[0] * 4, a=[1, -1, 10, 0], b=[2, 1, -10, 0])


def test_i8_negate():
    assert_match(i8_negate, dst=[0] * 4, src=[1, -1, 42, 0])


def test_i8_mul():
    assert_match(i8_mul, dst=[0] * 4, a=[2, -3, 0, 7], b=[3, -4, 99, 1])
