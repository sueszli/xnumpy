# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.assign (multi-dimensional tensor, 2D indexing)
# Lowering: 2D indices → stride computation + offset + llvm.store

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @assign_2d(%offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 0 : i64
# CHECK-NEXT:   %1 = arith.constant 4 : i64
# CHECK-NEXT:   %2 = arith.constant 1 : i64
# CHECK-NEXT:   %3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:   cf.br ^bb0(%0 : i64)
# CHECK-NEXT: ^bb0(%4 : i64):
# CHECK-NEXT:   %5 = arith.cmpi slt, %4, %1 : i64
# CHECK-NEXT:   cf.cond_br %5, ^bb1(%0 : i64), ^bb2
# CHECK-NEXT: ^bb1(%6 : i64):
# CHECK-NEXT:   %7 = arith.cmpi slt, %6, %1 : i64
# CHECK-NEXT:   cf.cond_br %7, ^bb3, ^bb4
# CHECK-NEXT: ^bb3:
# CHECK-NEXT:   %8 = arith.index_cast %4 : i64 to index
# CHECK-NEXT:   %9 = arith.index_cast %6 : i64 to index
# CHECK-NEXT:   %pointer_dim_stride = arith.constant 4 : index
# CHECK-NEXT:   %pointer_dim_offset = arith.muli %8, %pointer_dim_stride : index
# CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %9 : index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_1 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_2 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_3 = arith.addi %offset_pointer_2, %offset_pointer_1 : i64
# CHECK-NEXT:   %offset_pointer_4 = "llvm.inttoptr"(%offset_pointer_3) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%3, %offset_pointer_4) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   %10 = arith.addi %6, %2 : i64
# CHECK-NEXT:   cf.br ^bb1(%10 : i64)
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   %11 = arith.addi %4, %2 : i64
# CHECK-NEXT:   cf.br ^bb0(%11 : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def assign_2d(x: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            x[i, j] = 0.0
