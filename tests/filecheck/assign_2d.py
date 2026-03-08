# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @assign_2d({{.*}} : !llvm.ptr) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   llvm.br ^bb0({{.*}} : i64)
# CHECK-NEXT: ^bb0({{.*}} : i64):
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb1, ^bb2
# CHECK-NEXT: ^bb1:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   llvm.br ^bb3({{.*}} : i64)
# CHECK-NEXT: ^bb3({{.*}} : i64):
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb4, ^bb5
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.br ^bb3({{.*}} : i64)
# CHECK-NEXT: ^bb5:
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.br ^bb0({{.*}} : i64)
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def assign_2d(x: f32[4, 4] @ DRAM):
    for i in seq(0, 4):
        for j in seq(0, 4):
            x[i, j] = 0.0
