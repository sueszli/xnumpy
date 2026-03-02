# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @alloc_free(%0 : i64, %offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   %1 = arith.constant 0 : i64
# CHECK-NEXT:   %2 = arith.constant 1 : i64
# CHECK-NEXT:   cf.br ^bb0(%1 : i64)
# CHECK-NEXT: ^bb0(%3 : i64):
# CHECK-NEXT:   %4 = arith.cmpi slt, %3, %0 : i64
# CHECK-NEXT:   cf.cond_br %4, ^bb1, ^bb2
# CHECK-NEXT: ^bb1:
# CHECK-NEXT:   %5 = arith.constant 1 : i64
# CHECK-NEXT:   %offset_pointer_1 = "llvm.call"(%5) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %6 = arith.index_cast %3 : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %6, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_2 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_3 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_4 = arith.addi %offset_pointer_3, %offset_pointer_2 : i64
# CHECK-NEXT:   %offset_pointer_5 = "llvm.inttoptr"(%offset_pointer_4) : (i64) -> !llvm.ptr
# CHECK-NEXT:   %7 = "llvm.load"(%offset_pointer_5) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %8 = arith.index_cast %1 : i64 to index
# CHECK-NEXT:   %bytes_per_element_1 = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset_1 = arith.muli %8, %bytes_per_element_1 : index
# CHECK-NEXT:   %offset_pointer_6 = arith.index_cast %scaled_pointer_offset_1 : index to i64
# CHECK-NEXT:   %offset_pointer_7 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_8 = arith.addi %offset_pointer_7, %offset_pointer_6 : i64
# CHECK-NEXT:   %offset_pointer_9 = "llvm.inttoptr"(%offset_pointer_8) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%7, %offset_pointer_9) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   %bytes_per_element_2 = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset_2 = arith.muli %8, %bytes_per_element_2 : index
# CHECK-NEXT:   %offset_pointer_10 = arith.index_cast %scaled_pointer_offset_2 : index to i64
# CHECK-NEXT:   %offset_pointer_11 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_12 = arith.addi %offset_pointer_11, %offset_pointer_10 : i64
# CHECK-NEXT:   %offset_pointer_13 = "llvm.inttoptr"(%offset_pointer_12) : (i64) -> !llvm.ptr
# CHECK-NEXT:   %9 = "llvm.load"(%offset_pointer_13) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %bytes_per_element_3 = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset_3 = arith.muli %6, %bytes_per_element_3 : index
# CHECK-NEXT:   %offset_pointer_14 = arith.index_cast %scaled_pointer_offset_3 : index to i64
# CHECK-NEXT:   %offset_pointer_15 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_16 = arith.addi %offset_pointer_15, %offset_pointer_14 : i64
# CHECK-NEXT:   %offset_pointer_17 = "llvm.inttoptr"(%offset_pointer_16) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%9, %offset_pointer_17) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   "llvm.call"(%offset_pointer_1) <{callee = @free, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   %10 = arith.addi %3, %2 : i64
# CHECK-NEXT:   cf.br ^bb0(%10 : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
