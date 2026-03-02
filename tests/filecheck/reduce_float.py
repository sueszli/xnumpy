# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: reduce (+=) which internally generates exo.read + arith.addf + exo.assign
# Lowering: read → llvm.load, add → arith.addf, assign → llvm.store

from __future__ import annotations

from exo import *


# CHECK:      func.func @reduce_float(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK:        cf.br ^bb0(%0 : i64)
# CHECK:      ^bb0(%3 : i64):
# CHECK:        cf.cond_br %4, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        %6 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        %8 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %9 = arith.addf %8, %6 : f32
# CHECK:        "llvm.store"(%9, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:      ^bb2:
# CHECK-NEXT:   func.return
@proc
def reduce_float(x: f32[8] @ DRAM, y: f32[1] @ DRAM):
    for i in seq(0, 8):
        y[0] += x[i]
