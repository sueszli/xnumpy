# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @assign_from_scalar_memref(%offset_pointer : !llvm.ptr) {
# CHECK-NEXT:   %0 = arith.constant 1 : i64
# CHECK-NEXT:   %offset_pointer_1 = "llvm.call"(%0) <{callee = @malloc, {{.*}}}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %1 = arith.constant 4.200000e+01 : f32
# CHECK:        "llvm.store"(%1, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        cf.br ^bb0({{.*}} : i64)
# CHECK:      ^bb0(%5 : i64):
# CHECK:        cf.cond_br %6, ^bb1, ^bb2
# CHECK:      ^bb1:
# CHECK:        %7 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        "llvm.store"(%7, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
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
