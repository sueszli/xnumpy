# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.alloc (scalar), memref.store (scalar memref value → tensor),
#            memref.load (scalar memref load before tensor store)
# Lowering: scalar alloc → malloc(1), scalar value loaded before storing into tensor

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @assign_from_scalar_memref(%offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 1 : i64
# CHECK-NEXT:   %offset_pointer_1 = "llvm.call"(%0) <{callee = @malloc, {{.*}}}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %1 = arith.constant 4.200000e+01 : f32
# CHECK:        "llvm.store"(%1, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb0({{.*}} : i64)
# CHECK:      ^bb0(%6 : i64):
# CHECK:        cf.cond_br %7, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        %8 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        "llvm.store"(%8, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb0({{.*}} : i64)
# CHECK:      ^bb2:
# CHECK:        "llvm.call"(%offset_pointer_1) <{callee = @free, {{.*}}}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def assign_from_scalar_memref(x: f32[8] @ DRAM):
    tmp: f32
    tmp = 42.0
    for i in seq(0, 8):
        x[i] = tmp
