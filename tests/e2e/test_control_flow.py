from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def nested_if(out: f32[1] @ DRAM, a: index, b: index, c: index):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    if a < b:
        if b < c:
            out[0] = 1.0
        else:
            out[0] = 2.0
    else:
        out[0] = 3.0


def test_nested_if_inner_true():
    assert_match(nested_if, out=[0.0], a=1, b=5, c=10)


def test_nested_if_inner_false():
    assert_match(nested_if, out=[0.0], a=1, b=5, c=3)


def test_nested_if_outer_false():
    assert_match(nested_if, out=[0.0], a=10, b=5, c=20)


@proc
def multi_loop(x: f32[8] @ DRAM):
    for i in seq(0, 8):
        x[i] = x[i] + 1.0
    for i in seq(0, 8):
        x[i] = x[i] * 2.0


def test_multi_loop():
    assert_match(multi_loop, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def sum_first_half(out: f32[1] @ DRAM, N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        if i < N / 2:
            out[0] += x[i]


def test_sum_first_half():
    assert_match(sum_first_half, out=[0.0], N=8, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_sum_first_half_odd():
    assert_match(sum_first_half, out=[0.0], N=6, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@proc
def triple_nested(out: f32[1] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            for k in seq(0, 4):
                out[0] += 1.0


def test_triple_nested():
    assert_match(triple_nested, out=[0.0])


@proc
def diagonal_flag(out: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            if i == j:
                out[i, j] = 1.0
            else:
                out[i, j] = 0.0


def test_diagonal_flag():
    assert_match(diagonal_flag, out=[0.0] * 16)


@proc
def conditional_negate(out: f32[8] @ DRAM, x: f32[8] @ DRAM):
    for i in seq(0, 8):
        if i < 4:
            out[i] = -x[i]
        else:
            out[i] = x[i]


def test_conditional_negate():
    assert_match(conditional_negate, out=[0.0] * 8, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def upper_triangle(out: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            if i <= j:
                out[i, j] = 1.0
            else:
                out[i, j] = 0.0


def test_upper_triangle():
    assert_match(upper_triangle, out=[0.0] * 16)
