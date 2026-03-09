# RUN: uv run xnumpy --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @reduce_int({{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr) {
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(8) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}} : i64)
# CHECK-NEXT:   {{.*}}({{.*}} : i64):
# CHECK-NEXT:     {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}} : i32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}} : i64)
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def reduce_int(x: i32[8] @ DRAM, out: i32[1] @ DRAM):
    for i in seq(0, 8):
        out[0] += x[i]
