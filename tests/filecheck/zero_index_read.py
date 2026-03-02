# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.read with zero constant index, arithmetic on loaded value, exo.assign
# Edge case: index 0 produces a zero-offset load

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @zero_index_read(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 0 : i64
# CHECK-NEXT:   %1 = arith.index_cast %0 : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %1, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_2 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_3 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_4 = arith.addi %offset_pointer_3, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_5 = "llvm.inttoptr"(%offset_pointer_4) : (i64) -> !llvm.ptr
# CHECK-NEXT:   %2 = "llvm.load"(%offset_pointer_5) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %3 = arith.constant 1.000000e+00 : f32
# CHECK-NEXT:   %4 = arith.addf %2, %3 : f32
# CHECK-NEXT:   %offset_pointer_6 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_7 = arith.addi %offset_pointer_6, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_8 = "llvm.inttoptr"(%offset_pointer_7) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%4, %offset_pointer_8) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def zero_index_read(x: f32[1] @ DRAM, y: f32[1] @ DRAM):
    y[0] = x[0] + 1.0
