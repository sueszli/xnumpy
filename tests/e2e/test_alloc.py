from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def alloc_copy_1d(x: f32[8] @ DRAM, y: f32[8] @ DRAM):
    tmp: f32[8]
    for i in seq(0, 8):
        tmp[i] = x[i] * 2.0
    for i in seq(0, 8):
        y[i] = tmp[i]


def test_alloc_copy_1d():
    assert_match(alloc_copy_1d, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], y=[0.0] * 8)


@proc
def multi_alloc(x: f32[4] @ DRAM, y: f32[4] @ DRAM, out: f32[4] @ DRAM):
    a: f32[4]
    b: f32[4]
    for i in seq(0, 4):
        a[i] = x[i] + 1.0
    for i in seq(0, 4):
        b[i] = y[i] * 2.0
    for i in seq(0, 4):
        out[i] = a[i] + b[i]


def test_multi_alloc():
    assert_match(multi_alloc, x=[1.0, 2.0, 3.0, 4.0], y=[10.0, 20.0, 30.0, 40.0], out=[0.0] * 4)


@proc
def alloc_in_loop(x: f32[4] @ DRAM, out: f32[4] @ DRAM):
    for i in seq(0, 4):
        tmp: f32
        tmp = x[i] * x[i]
        out[i] = tmp


def test_alloc_in_loop():
    assert_match(alloc_in_loop, x=[2.0, 3.0, 4.0, 5.0], out=[0.0] * 4)


@proc
def alloc_2d(src: f32[4, 4] @ DRAM, dst: f32[4, 4] @ DRAM):
    tmp: f32[4, 4]
    for i in seq(0, 4):
        for j in seq(0, 4):
            tmp[i, j] = src[i, j] + 1.0
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = tmp[i, j]


def test_alloc_2d():
    src = [float(i) for i in range(16)]
    assert_match(alloc_2d, src=src, dst=[0.0] * 16)


@proc
def alloc_accumulator(x: f32[8] @ DRAM, out: f32[1] @ DRAM):
    acc: f32
    acc = 0.0
    for i in seq(0, 8):
        acc = acc + x[i]
    out[0] = acc


def test_alloc_accumulator():
    assert_match(alloc_accumulator, x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], out=[0.0])


@proc
def alloc_i32(x: i32[4] @ DRAM, out: i32[4] @ DRAM):
    tmp: i32[4]
    for i in seq(0, 4):
        tmp[i] = x[i] * x[i]
    for i in seq(0, 4):
        out[i] = tmp[i]


def test_alloc_i32():
    assert_match(alloc_i32, x=[2, 3, 4, 5], out=[0] * 4)
