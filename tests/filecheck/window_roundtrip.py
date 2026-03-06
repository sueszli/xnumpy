# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @set_element(%offset_pointer : !llvm.ptr) {
# CHECK:        arith.constant 4.200000e+01 : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK:      func.func @window_roundtrip(%offset_pointer : !llvm.ptr) {
# CHECK:        arith.constant 2 : i64
# CHECK:        arith.muli {{.*}}, {{.*}} : index
# CHECK:        func.call @set_element({{.*}}) : (!llvm.ptr) -> ()
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        func.return
# CHECK-NEXT: }
@proc
def set_element(row: [f32][4] @ DRAM):
    row[1] = 42.0


@proc
def window_roundtrip(A: f32[4, 4] @ DRAM):
    set_element(A[2, :])
    A[0, 0] = A[2, 1]
