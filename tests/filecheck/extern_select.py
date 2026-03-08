# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @uses_select({{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = llvm.fcmp "olt" {{.*}}, {{.*}} : f32
# CHECK-NEXT:   {{.*}} = llvm.select {{.*}}, {{.*}}, {{.*}} : i1, f32
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
from exo.platforms.x86 import *


@proc
def uses_select(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = select(0.0, a[0], a[0], b[0])
