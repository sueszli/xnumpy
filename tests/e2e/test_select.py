from __future__ import annotations

from conftest import assert_match
from exo import *
from exo.libs.externs import select


@proc
def threshold_f32(out: f32[8] @ DRAM, x: f32[8] @ DRAM, t: f32[1] @ DRAM):
    for i in seq(0, 8):
        out[i] = select(x[i], t[0], 1.0, 0.0)


def test_threshold_mixed():
    assert_match(threshold_f32, out=[0.0] * 8, x=[1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0], t=[5.0])


def test_threshold_all_below():
    assert_match(threshold_f32, out=[0.0] * 8, x=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], t=[100.0])


def test_threshold_all_above():
    assert_match(threshold_f32, out=[0.0] * 8, x=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], t=[1.0])


@proc
def select_min_f32(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = select(a[i], b[i], a[i], b[i])


@proc
def select_max_f32(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = select(b[i], a[i], b[i], a[i])


def test_select_min():
    assert_match(select_min_f32, out=[0.0] * 4, a=[1.0, 5.0, 3.0, 7.0], b=[4.0, 2.0, 6.0, 1.0])


def test_select_max():
    assert_match(select_max_f32, out=[0.0] * 4, a=[1.0, 5.0, 3.0, 7.0], b=[4.0, 2.0, 6.0, 1.0])


@proc
def select_f64(out: f64[4] @ DRAM, a: f64[4] @ DRAM, b: f64[4] @ DRAM):
    for i in seq(0, 4):
        out[i] = select(a[i], b[i], a[i], b[i])


def test_select_f64():
    assert_match(select_f64, out=[0.0] * 4, a=[1.0, 5.0, 3.0, 7.0], b=[4.0, 2.0, 6.0, 1.0])


@proc
def clamp_positive(out: f32[8] @ DRAM, x: f32[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = select(x[i], 0.0, 0.0, x[i])


def test_clamp_positive():
    assert_match(clamp_positive, out=[0.0] * 8, x=[-3.0, -1.0, 0.0, 0.5, 1.0, -0.1, 2.5, -10.0])
