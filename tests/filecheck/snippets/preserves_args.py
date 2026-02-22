# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @preserves_args(%0 : memref<16xf32, strided<[1]>>, %1 : i32) {
# CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
# CHECK-NEXT:   %3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:   memref.store %3, %0[%2] : memref<16xf32, strided<[1]>>
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
