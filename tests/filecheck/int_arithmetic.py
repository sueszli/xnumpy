# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @int_arithmetic({{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr) {
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i32
# CHECK-NEXT:   {{.*}} = "llvm.ptrtoint"({{.*}}) : (!llvm.ptr) -> i64
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = "llvm.inttoptr"({{.*}}) : (i64) -> !llvm.ptr
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.sub {{.*}}, {{.*}} : i32
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.mul {{.*}}, {{.*}} : i32
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.sdiv {{.*}}, {{.*}} : i32
# CHECK-NEXT:   "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }

from __future__ import annotations

from exo import *


@proc
def int_arithmetic(out: i32[1] @ DRAM, x: i32[1] @ DRAM, y: i32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
