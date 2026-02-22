# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import DRAM, proc


# CHECK: builtin.module {
# CHECK-NEXT: func.func @fixed_matmul(%0 : memref<16x16xf32, strided<[1, 1]>>, %1 : memref<16x16xf32, strided<[1, 1]>>, %2 : memref<16x16xf32, strided<[1, 1]>>) {
@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    # CHECK-NEXT: %3 = arith.constant 0 : i32
    # CHECK-NEXT: %4 = arith.index_cast %3 : i32 to index
    # CHECK-NEXT: %5 = arith.constant 16 : i32
    # CHECK-NEXT: %6 = arith.index_cast %5 : i32 to index
    # CHECK-NEXT: %7 = arith.constant 1 : index
    # CHECK-NEXT: %8 = arith.constant 0.000000e+00 : f32
    # CHECK-NEXT: scf.for %9 = %4 to %6 step %7 {
    for i in seq(0, 16):
        # CHECK-NEXT: scf.for %10 = %4 to %6 step %7 {
        for j in seq(0, 16):
            # CHECK-NEXT:memref.store %8, %0[%9, %10] : memref<16x16xf32, strided<[1, 1]>>
            C[i, j] = 0.0
            # CHECK-NEXT: scf.for %11 = %4 to %6 step %7 {
            for k in seq(0, 16):
                # CHECK-NEXT: %12 = memref.load %1[%9, %11] : memref<16x16xf32, strided<[1, 1]>>
                # CHECK-NEXT: %13 = memref.load %2[%11, %10] : memref<16x16xf32, strided<[1, 1]>>
                # CHECK-NEXT: %14 = arith.mulf %12, %13 : f32
                # CHECK-NEXT: %15 = memref.load %0[%9, %10] : memref<16x16xf32, strided<[1, 1]>>
                # CHECK-NEXT: %16 = arith.addf %15, %14 : f32
                # CHECK-NEXT: memref.store %16, %0[%9, %10] : memref<16x16xf32, strided<[1, 1]>>
                C[i, j] += A[i, k] * B[k, j]
            # CHECK-NEXT: }
        # CHECK-NEXT: }
    # CHECK-NEXT: }
    # CHECK-NEXT: func.return
    # CHECK-NEXT: }


# CHECK-NEXT: }
