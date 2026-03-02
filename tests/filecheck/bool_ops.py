# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @bool_ops(%offset_pointer : !llvm.ptr, %0 : i64, %1 : i64, %2 : i64) {
# CHECK-NEXT:   %3 = arith.cmpi slt, %0, %1 : i64
# CHECK-NEXT:   %4 = arith.cmpi slt, %1, %2 : i64
# CHECK-NEXT:   %5 = arith.andi %3, %4 : i1
# CHECK-NEXT:   cf.cond_br %5, ^bb0, ^bb1
# CHECK:        %9 = arith.ori %3, %4 : i1
# CHECK-NEXT:   cf.cond_br %9, ^bb2, ^bb3
@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    if a < b and b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
