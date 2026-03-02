# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.read (tensor indexed), exo.assign (tensor indexed) in a loop
# Lowering: exo.read → arith.index_cast + memref.load → llvm.load
#           exo.assign → arith.index_cast + memref.store → llvm.store

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @read_tensor_indexed(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 0 : i64
# CHECK-NEXT:   %1 = arith.constant 8 : i64
# CHECK-NEXT:   %2 = arith.constant 1 : i64
# CHECK-NEXT:   cf.br ^bb0(%0 : i64)
# CHECK-NEXT: ^bb0(%3 : i64):
# CHECK-NEXT:   %4 = arith.cmpi slt, %3, %1 : i64
# CHECK-NEXT:   cf.cond_br %4, ^bb1, ^bb2
# CHECK-NEXT: ^bb1:
# CHECK:        %6 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        "llvm.store"(%6, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb0({{.*}} : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def read_tensor_indexed(x: f32[8] @ DRAM, y: f32[8] @ DRAM):
    for i in seq(0, 8):
        y[i] = x[i]
