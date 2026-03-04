from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def logic_and(out: f32[1] @ DRAM, a: index, b: index, c: index):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    if a < b and b < c:
        out[0] = 1.0
    else:
        out[0] = 0.0


@proc
def logic_or(out: f32[1] @ DRAM, a: index, b: index, c: index):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    if a < b or b < c:
        out[0] = 1.0
    else:
        out[0] = 0.0


def test_and_tt():
    assert_match(logic_and, out=[0.0], a=1, b=2, c=3)


def test_and_tf():
    assert_match(logic_and, out=[0.0], a=1, b=5, c=3)


def test_and_ft():
    assert_match(logic_and, out=[0.0], a=5, b=2, c=3)


def test_and_ff():
    assert_match(logic_and, out=[0.0], a=5, b=2, c=1)


def test_or_tt():
    assert_match(logic_or, out=[0.0], a=1, b=2, c=3)


def test_or_tf():
    assert_match(logic_or, out=[0.0], a=1, b=2, c=1)


def test_or_ft():
    assert_match(logic_or, out=[0.0], a=5, b=2, c=3)


def test_or_ff():
    assert_match(logic_or, out=[0.0], a=5, b=2, c=1)


@proc
def and_boundary(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b and a < 10:
        out[0] = 1.0
    else:
        out[0] = 0.0


def test_and_boundary_both_true():
    assert_match(and_boundary, out=[0.0], a=5, b=5)


def test_and_boundary_eq_but_ge10():
    assert_match(and_boundary, out=[0.0], a=10, b=10)


def test_and_boundary_ne():
    assert_match(and_boundary, out=[0.0], a=3, b=5)
