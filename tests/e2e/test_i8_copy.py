from __future__ import annotations

from exo import *

from conftest import assert_match


@proc
def i8_copy(dst: i8[8] @ DRAM, src: i8[8] @ DRAM):
    for i in seq(0, 8):
        dst[i] = src[i]


def test_i8_copy():
    assert_match(i8_copy, dst=[0] * 8, src=[10, 20, 30, 40, 50, 60, 70, 80])
