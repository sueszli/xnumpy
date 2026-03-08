# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      llvm.func @if_else({{.*}}) {
# CHECK:        llvm.cond_br {{.*}}, ^bb0, ^bb1
# CHECK:      ^bb0:
# CHECK:        llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        llvm.br ^bb2
# CHECK:      ^bb1:
# CHECK:        llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        llvm.br ^bb2
# CHECK:      ^bb2:
# CHECK-NEXT:   llvm.return
@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0
