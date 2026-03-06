# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @set_row(%offset_pointer : !llvm.ptr) {
# CHECK:        cf.br ^bb0(%0 : i64)
# CHECK:      ^bb0(%4 : i64):
# CHECK:        cf.cond_br %5, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        "llvm.store"(%3, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:      ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK:      func.func @window_row(%offset_pointer : !llvm.ptr) {
# CHECK:        cf.br ^bb0(%0 : i64)
# CHECK:      ^bb0({{.*}} : i64):
# CHECK:        cf.cond_br {{.*}}, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        arith.muli {{.*}}, {{.*}} : index
# CHECK:        func.call @set_row({{.*}}) : (!llvm.ptr) -> ()
# CHECK:      ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def set_row(row: [f32][4] @ DRAM):
    for j in seq(0, 4):
        row[j] = 0.0


@proc
def window_row(A: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        set_row(A[i, :])
