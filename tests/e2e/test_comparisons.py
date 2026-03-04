from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def cmp_lt(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 0.0


@proc
def cmp_le(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a <= b:
        out[0] = 1.0
    else:
        out[0] = 0.0


@proc
def cmp_gt(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a > b:
        out[0] = 1.0
    else:
        out[0] = 0.0


@proc
def cmp_ge(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a >= b:
        out[0] = 1.0
    else:
        out[0] = 0.0


@proc
def cmp_eq(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b:
        out[0] = 1.0
    else:
        out[0] = 0.0


def test_lt_true():
    assert_match(cmp_lt, out=[0.0], a=3, b=5)


def test_lt_equal():
    assert_match(cmp_lt, out=[0.0], a=5, b=5)


def test_lt_false():
    assert_match(cmp_lt, out=[0.0], a=7, b=5)


def test_le_true():
    assert_match(cmp_le, out=[0.0], a=3, b=5)


def test_le_equal():
    assert_match(cmp_le, out=[0.0], a=5, b=5)


def test_le_false():
    assert_match(cmp_le, out=[0.0], a=7, b=5)


def test_gt_true():
    assert_match(cmp_gt, out=[0.0], a=7, b=5)


def test_gt_equal():
    assert_match(cmp_gt, out=[0.0], a=5, b=5)


def test_gt_false():
    assert_match(cmp_gt, out=[0.0], a=3, b=5)


def test_ge_true():
    assert_match(cmp_ge, out=[0.0], a=7, b=5)


def test_ge_equal():
    assert_match(cmp_ge, out=[0.0], a=5, b=5)


def test_ge_false():
    assert_match(cmp_ge, out=[0.0], a=3, b=5)


def test_eq_true():
    assert_match(cmp_eq, out=[0.0], a=5, b=5)


def test_eq_false():
    assert_match(cmp_eq, out=[0.0], a=3, b=5)


def test_lt_zero():
    assert_match(cmp_lt, out=[0.0], a=0, b=1)


def test_eq_zero():
    assert_match(cmp_eq, out=[0.0], a=0, b=0)
