from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def assign_2d(dst: f32[4, 4] @ DRAM, src: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]


def test_assign_2d():
    src = list(range(16))
    assert_match(assign_2d, dst=[0] * 16, src=src)
