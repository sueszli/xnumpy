from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def scalar_double(x: f32[1] @ DRAM):
    x[0] = x[0] + x[0]


def test_scalar_double():
    assert_match(scalar_double, x=[21.0])
