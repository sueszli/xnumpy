# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @if_else({{.*}}) {
# CHECK:        cf.cond_br {{.*}}, ^bb0, ^bb1
# CHECK:      ^bb0:
# CHECK:        llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb2
# CHECK:      ^bb1:
# CHECK:        llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb2
# CHECK:      ^bb2:
# CHECK-NEXT:   func.return
@proc
def if_else(out: f32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a < b:
        out[0] = 1.0
    else:
        out[0] = 2.0
