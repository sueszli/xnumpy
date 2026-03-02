# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.alloc (DRAM path), exo.read (tensor indexed), exo.assign (tensor indexed)
# Lowering: exo.alloc → malloc, exo.read/assign → memref.load/store → llvm.load/store

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @alloc_dram(%offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 4 : i64
# CHECK-NEXT:   %offset_pointer_1 = "llvm.call"(%0) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %1 = arith.constant 0 : i64
# CHECK-NEXT:   %2 = arith.index_cast %1 : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %2, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_2 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_3 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_4 = arith.addi %offset_pointer_3, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_5 = "llvm.inttoptr"(%offset_pointer_4) : (i64) -> !llvm.ptr
# CHECK-NEXT:   %3 = "llvm.load"(%offset_pointer_5) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %offset_pointer_6 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_7 = arith.addi %offset_pointer_6, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_8 = "llvm.inttoptr"(%offset_pointer_7) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%3, %offset_pointer_8) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   %4 = "llvm.load"(%offset_pointer_8) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   "llvm.store"(%4, %offset_pointer_5) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   "llvm.call"(%offset_pointer_1) <{callee = @free, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def alloc_dram(x: f32[8] @ DRAM):
    tmp: f32[4]
    tmp[0] = x[0]
    x[0] = tmp[0]
