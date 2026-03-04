from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def vec_add(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + b[i]


def test_vec_add():
    assert_match(vec_add, N=5, out=[0] * 5, a=[1, 2, 3, 4, 5], b=[10, 20, 30, 40, 50])
