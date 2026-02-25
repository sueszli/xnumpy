# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @preserves_args(%0 : memref<16xf32, "DRAM">, %1 : i64) {
# CHECK-NEXT:   %2 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:   exo.assign %2, %0[%1], sizes : [], {static_sizes = array<i64: 16>} : f32, memref<16xf32, "DRAM">
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
