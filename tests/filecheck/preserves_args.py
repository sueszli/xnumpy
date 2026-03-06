# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @preserves_args(%offset_pointer : !llvm.ptr, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = arith.index_cast {{.*}} : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli {{.*}}, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_1 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_2 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_3 = arith.addi %offset_pointer_2, %offset_pointer_1 : i64
# CHECK-NEXT:   %offset_pointer_4 = "llvm.inttoptr"(%offset_pointer_3) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, %offset_pointer_4) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
