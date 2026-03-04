from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0


def test_if_else_true_branch():
    assert_match(if_else, out=[0.0], a=1, b=5)


def test_if_else_false_branch():
    assert_match(if_else, out=[0.0], a=5, b=1)
