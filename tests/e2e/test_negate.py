from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def negate_float(out: f32[1] @ DRAM, a: f32[1] @ DRAM):
    out[0] = -a[0]


@proc
def negate_int(out: i32[1] @ DRAM, a: i32[1] @ DRAM):
    out[0] = -a[0]


def test_negate_float():
    assert_match(negate_float, out=[0.0], a=[42.0])


def test_negate_int():
    assert_match(negate_int, out=[0], a=[7])
