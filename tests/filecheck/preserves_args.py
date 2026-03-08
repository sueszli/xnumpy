# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @preserves_args({{.*}} : !llvm.ptr, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def preserves_args(x: f32[16], idx: index):
    assert idx >= 0 and idx < 16
    x[idx] = 0.0
