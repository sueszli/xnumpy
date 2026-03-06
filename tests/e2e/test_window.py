from __future__ import annotations

from conftest import assert_match
from exo import *


@proc
def fill_val(x: [f32][4] @ DRAM):
    for i in seq(0, 4):
        x[i] = 99.0


@proc
def window_fill(x: f32[8] @ DRAM):
    fill_val(x[4:8])


def test_window_fill_second_half():
    assert_match(window_fill, x=[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])


@proc
def window_fill_first(x: f32[8] @ DRAM):
    fill_val(x[0:4])


def test_window_fill_first_half():
    assert_match(window_fill_first, x=[0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0])


@proc
def sum_slice(out: f32[1] @ DRAM, x: [f32][4] @ DRAM):
    for i in seq(0, 4):
        out[0] += x[i]


@proc
def sum_second_half(out: f32[1] @ DRAM, x: f32[8] @ DRAM):
    sum_slice(out, x[4:8])


def test_window_read():
    assert_match(sum_second_half, out=[0.0], x=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@proc
def scale_row(row: [f32][4] @ DRAM, factor: f32[1] @ DRAM):
    for j in seq(0, 4):
        row[j] = row[j] * factor[0]


@proc
def scale_row_2(M: f32[4, 4] @ DRAM, factor: f32[1] @ DRAM):
    scale_row(M[2, :], factor)


def test_window_2d_row():
    M = [float(i * 4 + j) for i in range(4) for j in range(4)]
    assert_match(scale_row_2, M=M, factor=[10.0])
