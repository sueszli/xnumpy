# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @reduce_float(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK:        cf.br ^bb0({{.*}} : i64)
# CHECK:      ^bb0({{.*}} : i64):
# CHECK:        cf.cond_br {{.*}}, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = llvm.fadd {{.*}}, {{.*}} : f32
# CHECK:        "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:      ^bb2:
# CHECK-NEXT:   func.return
@proc
def reduce_float(x: f32[8] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]
