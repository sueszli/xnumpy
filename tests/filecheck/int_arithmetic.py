# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_arithmetic(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK:        %2 = "llvm.load"(%offset_pointer_6) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK:        %3 = "llvm.load"(%offset_pointer_9) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   %4 = arith.addi %2, %3 : i32
# CHECK:        %7 = arith.subi %5, %6 : i32
# CHECK:        %10 = arith.muli %8, %9 : i32
# CHECK:        %13 = arith.divsi %11, %12 : i32
@proc
def int_arithmetic(out: i32[1] @ DRAM, a: i32[1] @ DRAM, b: i32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
