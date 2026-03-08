# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @int_comparisons({{.*}} : !llvm.ptr, {{.*}} : i64, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.icmp "eq" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb0, ^bb1
# CHECK-NEXT: ^bb0:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1 : i32) : i32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.br ^bb2
# CHECK-NEXT: ^bb1:
# CHECK-NEXT:   llvm.br ^bb2
# CHECK-NEXT: ^bb2:
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb3, ^bb4
# CHECK-NEXT: ^bb3:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(2 : i32) : i32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.br ^bb5
# CHECK-NEXT: ^bb4:
# CHECK-NEXT:   llvm.br ^bb5
# CHECK-NEXT: ^bb5:
# CHECK-NEXT:   {{.*}} = llvm.icmp "sgt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   llvm.cond_br {{.*}}, ^bb6, ^bb7
# CHECK-NEXT: ^bb6:
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(3 : i32) : i32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.br ^bb8
# CHECK-NEXT: ^bb7:
# CHECK-NEXT:   llvm.br ^bb8
# CHECK-NEXT: ^bb8:
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
