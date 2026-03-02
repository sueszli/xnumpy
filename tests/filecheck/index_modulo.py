# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @index_modulo(%offset_pointer : !llvm.ptr, %0 : i64) {
# CHECK-NEXT:   %1 = arith.constant 10 : i64
# CHECK-NEXT:   %2 = arith.remsi %0, %1 : i64
# CHECK-NEXT:   %3 = arith.constant 42 : i32
# CHECK-NEXT:   %4 = arith.index_cast %2 : i64 to index
# CHECK-NEXT:   %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:   %scaled_pointer_offset = arith.muli %4, %bytes_per_element : index
# CHECK-NEXT:   %offset_pointer_1 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:   %offset_pointer_2 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:   %offset_pointer_3 = arith.addi %offset_pointer_2, %offset_pointer_1 : i64
# CHECK-NEXT:   %offset_pointer_4 = "llvm.inttoptr"(%offset_pointer_3) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"(%3, %offset_pointer_4) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
@proc
def index_modulo(out: i32[10] @ DRAM, n: index):
    assert n >= 0
    assert n < 10
    out[n % 10] = 42
