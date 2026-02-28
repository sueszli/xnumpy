# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @alloc_free(%0 : i64, %1 : memref<-1xf32, "DRAM">) {
# CHECK-NEXT:   %c0 = arith.constant 0 : i64
# CHECK-NEXT:   %2 = arith.constant 1 : i64
# CHECK-NEXT:   scf.for %3 = %c0 to %0 step %2  : i64 {
# CHECK-NEXT:     %4 = exo.alloc "DRAM" : memref<1xf32, "DRAM">
# CHECK-NEXT:     %5 = exo.read %1[%3] -> f32
# CHECK-NEXT:     exo.assign %5, %4[%c0], sizes : [], {static_sizes = array<i64: 1>} : f32, memref<1xf32, "DRAM">
# CHECK-NEXT:     %6 = exo.read %4[%c0] -> f32
# CHECK-NEXT:     exo.assign %6, %1[%3], sizes : [%0], {static_sizes = array<i64: -9223372036854775808>} : f32, memref<-1xf32, "DRAM">
# CHECK-NEXT:     memref.dealloc %4 : memref<1xf32, "DRAM">
# CHECK-NEXT:   }
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: }
@proc
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
