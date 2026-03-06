# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @usub_float({{.*}}) {
# CHECK:        {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = llvm.fneg {{.*}} : f32
# CHECK:        "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
@proc
def usub_float(out: f32[1] @ DRAM, x: f32[1] @ DRAM):
    out[0] = -x[0]


# CHECK:      func.func @usub_int({{.*}}) {
# CHECK:        {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.mlir.constant(0 : i32) : i32
# CHECK-NEXT:   {{.*}} = llvm.sub {{.*}}, {{.*}} : i32
# CHECK:        "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
@proc
def usub_int(out: i32[1] @ DRAM, x: i32[1] @ DRAM):
    out[0] = -x[0]
