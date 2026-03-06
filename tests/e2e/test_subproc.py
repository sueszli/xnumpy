from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def add_one_scalar(x: f32[1] @ DRAM):
    x[0] = x[0] + 1.0


@proc
def call_add_one(x: f32[1] @ DRAM):
    add_one_scalar(x)


def test_call_scalar_subproc():
    assert_match(call_add_one, x=[5.0])


@proc
def double_elements(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        x[i] = x[i] * 2.0


@proc
def call_double(x: f32[8] @ DRAM):
    double_elements(8, x)


def test_call_array_subproc():
    assert_match(call_double, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def increment(x: f32[1] @ DRAM):
    x[0] = x[0] + 1.0


@proc
def double_val(x: f32[1] @ DRAM):
    x[0] = x[0] * 2.0


@proc
def inc_then_double(x: f32[1] @ DRAM):
    increment(x)
    double_val(x)


def test_chained_subprocs():
    assert_match(inc_then_double, x=[5.0])


@proc
def add_buffers(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + b[i]


@proc
def call_add_buffers(out: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    add_buffers(4, out, a, b)


def test_call_multi_buffer_subproc():
    assert_match(call_add_buffers, out=[0.0] * 4, a=[1.0, 2.0, 3.0, 4.0], b=[10.0, 20.0, 30.0, 40.0])
