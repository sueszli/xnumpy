# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import DRAM, proc


# CHECK: builtin.module {
# CHECK-NEXT:   func.func @fixed_matmul(%0 : memref<16x16xf32, "DRAM">, %1 : memref<16x16xf32, "DRAM">, %2 : memref<16x16xf32, "DRAM">) {
# CHECK-NEXT:     %3 = arith.constant 0 : i64
# CHECK-NEXT:     %4 = arith.constant 16 : i64
# CHECK-NEXT:     %5 = arith.constant 1 : i64
# CHECK-NEXT:     %6 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     scf.for %7 = %3 to %4 step %5  : i64 {
# CHECK-NEXT:       scf.for %8 = %3 to %4 step %5  : i64 {
# CHECK-NEXT:         exo.assign %6, %0[%7, %8], sizes : [], {static_sizes = array<i64: 16, 16>} : f32, memref<16x16xf32, "DRAM">
# CHECK-NEXT:         scf.for %9 = %3 to %4 step %5  : i64 {
# CHECK-NEXT:           %10 = exo.read %1[%7, %9] -> f32
# CHECK-NEXT:           %11 = exo.read %2[%9, %8] -> f32
# CHECK-NEXT:           %12 = arith.mulf %10, %11 : f32
# CHECK-NEXT:           exo.reduce %12, %0[%7, %8], sizes : [], {static_sizes = array<i64: 16, 16>} : f32, memref<16x16xf32, "DRAM">
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT: }
@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]
