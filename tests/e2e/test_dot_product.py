from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def dot_product(N: size, out: f32[1] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[0] += a[i] * b[i]


def test_dot_product():
    assert_match(dot_product, N=4, out=[0], a=[1, 2, 3, 4], b=[5, 6, 7, 8])
