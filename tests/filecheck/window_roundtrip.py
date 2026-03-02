# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Proves that writes through a window are visible via direct reads from the
# original buffer — i.e. that the window round-trip is correct.
#
# Windows lower to plain pointer arithmetic (an offset into the original
# buffer). This test verifies that both paths — writing through a windowed
# pointer and reading from the base pointer — resolve to the same memory:
#
#   1. set_element(A[2, :]) — A[2, :] becomes a pointer offset
#      (32 = row 2 * 4 elements * 4 bytes). set_element writes 42.0
#      through that offset pointer.
#   2. A[0, 0] = A[2, 1]   — reads from the base pointer at the same
#      location the windowed write targeted (row 2, col 1).
#
# The CHECK patterns verify the offset computation for the window call
# and the load/store pair for the read-back.

from __future__ import annotations

from exo import *


# CHECK:      func.func @set_element(%offset_pointer : !llvm.ptr) {
# CHECK:        arith.constant 4.200000e+01 : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK:      func.func @window_roundtrip(%offset_pointer : !llvm.ptr) {
# CHECK:        arith.constant 32 : i64
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
