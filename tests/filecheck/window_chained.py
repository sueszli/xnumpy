# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: chained window (subview of subview), multi-proc lowering
# Lowering: T[2, :, :] → 2D subview of 3D tensor, then M[1, :] → 1D subview of 2D

from __future__ import annotations

from exo import *


# CHECK:      func.func @set_first(%offset_pointer : !llvm.ptr) {
# CHECK:        %1 = arith.constant 1.000000e+00 : f32
# CHECK:        "llvm.store"(%1, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK:      func.func @inner(%offset_pointer : !llvm.ptr) {
# CHECK:        %0 = arith.constant 16 : i64
# CHECK:        func.call @set_first({{.*}}) : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK:      func.func @outer(%offset_pointer : !llvm.ptr) {
# CHECK:        %0 = arith.constant 96 : i64
# CHECK:        func.call @inner({{.*}}) : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def set_first(x: [f32][4] @ DRAM):
    x[0] = 1.0


@proc
def inner(M: [f32][4, 4] @ DRAM):
    set_first(M[1, :])


@proc
def outer(T: f32[3, 4, 4] @ DRAM):
    inner(T[2, :, :])
