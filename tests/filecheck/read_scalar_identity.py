# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.read (scalar identity / memref[1] path), exo.assign (scalar memref path)
# Lowering: scalar memref → memref[1], exo.read/assign → indexed load/store with zero index

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @read_scalar_identity(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 0 : i64
# CHECK-NEXT:   %1 = arith.index_cast %0 : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %1, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_2 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_3 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_4 = arith.addi %offset_pointer_3, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_5 = "llvm.inttoptr"(%offset_pointer_4) : (i64) -> !llvm.ptr
# CHECK-NEXT:   %2 = "llvm.load"(%offset_pointer_5) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %offset_pointer_6 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_7 = arith.addi %offset_pointer_6, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_8 = "llvm.inttoptr"(%offset_pointer_7) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%2, %offset_pointer_8) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def read_scalar_identity(x: f32[1] @ DRAM, y: f32[1] @ DRAM):
    y[0] = x[0]
