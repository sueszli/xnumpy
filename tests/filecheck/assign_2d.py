# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @assign_2d(%offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   cf.br ^bb0({{.*}} : i64)
# CHECK-NEXT: ^bb0({{.*}} : i64):
# CHECK-NEXT:   {{.*}} = arith.cmpi slt, {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb1({{.*}} : i64), ^bb2
# CHECK-NEXT: ^bb1({{.*}} : i64):
# CHECK-NEXT:   {{.*}} = arith.cmpi slt, {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb3, ^bb4
# CHECK-NEXT: ^bb3:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = arith.index_cast {{.*}} : i64 to index
# CHECK-NEXT:   {{.*}} = arith.index_cast {{.*}} : i64 to index
# CHECK-NEXT:   %pointer_dim_stride = arith.constant 4 : index
# CHECK-NEXT:   %pointer_dim_offset = arith.muli {{.*}}, %pointer_dim_stride : index
# CHECK-NEXT:   %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, {{.*}} : index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_1 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_2 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_3 = arith.addi %offset_pointer_2, %offset_pointer_1 : i64
# CHECK-NEXT:   %offset_pointer_4 = "llvm.inttoptr"(%offset_pointer_3) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, %offset_pointer_4) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   {{.*}} = arith.addi {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.br ^bb1({{.*}} : i64)
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   {{.*}} = arith.addi {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.br ^bb0({{.*}} : i64)
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
