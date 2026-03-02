# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_comparisons(%offset_pointer : !llvm.ptr, %0 : i64, %1 : i64) {
# CHECK-NEXT:   %2 = arith.cmpi eq, %0, %1 : i64
# CHECK-NEXT:   cf.cond_br %2, ^bb0, ^bb1
# CHECK:        %6 = arith.cmpi slt, %0, %1 : i64
# CHECK-NEXT:   cf.cond_br %6, ^bb2, ^bb3
# CHECK:        %10 = arith.cmpi sgt, %0, %1 : i64
# CHECK-NEXT:   cf.cond_br %10, ^bb4, ^bb5
@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
