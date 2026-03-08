# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @bool_ops({{.*}} : !llvm.ptr, {{.*}} : i64, {{.*}} : i64, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.and {{.*}}, {{.*}} : i1
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb0, ^bb1
# CHECK-NEXT: ^bb0:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.br ^bb2
# CHECK-NEXT: ^bb1:
# CHECK-NEXT:   llvm.br ^bb2
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.or {{.*}}, {{.*}} : i1
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb3, ^bb4
# CHECK-NEXT: ^bb3:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(2.000000e+00 : f32) : f32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.br ^bb5
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   llvm.br ^bb5
# CHECK-NEXT: ^bb5:
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    if a < b and b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
