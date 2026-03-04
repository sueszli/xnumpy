from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def alloc_copy(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp


def test_alloc_copy():
    assert_match(alloc_copy, N=4, x=[1.0, 2.0, 3.0, 4.0])
