# RUN: FILECHECK-EXO

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @conv1d(%0 : i64, %1 : i64, %2 : i64, %3 : i64, %4 : memref<?x?xi32, "DRAM">, %5 : memref<?x?x?xi32, "DRAM">, %6 : memref<?x?xi32, "DRAM">) {
# CHECK-NEXT:   %c0 = arith.constant 0 : i64
# CHECK-NEXT:   %7 = arith.constant 1 : i64
# CHECK-NEXT:   %8 = arith.constant 0 : i32
# CHECK-NEXT:   scf.for %9 = %c0 to %1 step %7  : i64 {
# CHECK-NEXT:     scf.for %10 = %c0 to %2 step %7  : i64 {
# CHECK-NEXT:       exo.assign %8, %6[%9, %10]
# CHECK-NEXT:       scf.for %11 = %c0 to %0 step %7  : i64 {
# CHECK-NEXT:         scf.for %12 = %c0 to %3 step %7  : i64 {
# CHECK-NEXT:           %13 = exo.alloc "DRAM" : memref<1xi32, "DRAM">
# CHECK-NEXT:           %14 = arith.addi %10, %12 : i64
# CHECK-NEXT:           %15 = arith.cmpi slt, %14, %2 : i64
# CHECK-NEXT:           scf.if %15 {
# CHECK-NEXT:             %16 = exo.read %4[%11, %14] -> i32
# CHECK-NEXT:             exo.assign %16, %13[%c0]
# CHECK-NEXT:           } else {
# CHECK-NEXT:             exo.assign %8, %13[%c0]
# CHECK-NEXT:           }
# CHECK-NEXT:           %17 = exo.read %5[%9, %11, %12] -> i32
# CHECK-NEXT:           %18 = exo.read %13[%c0] -> i32
# CHECK-NEXT:           %19 = arith.muli %17, %18 : i32
# CHECK-NEXT:           exo.reduce %19, %6[%9, %10]
# CHECK-NEXT:           exo.free %13 "DRAM" : memref<1xi32, "DRAM">
# CHECK-NEXT:         }
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: }
@proc
def conv1d(
    IC: size,
    OC: size,
    N: size,
    W: size,
    data: i32[IC, N],
    kernels: i32[OC, IC, W],
    out: i32[OC, N],
):
    for i in seq(0, OC):
        for j in seq(0, N):
            out[i, j] = 0
            for c in seq(0, IC):
                for r in seq(0, W):
                    y: i32
                    if j + r < N:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
