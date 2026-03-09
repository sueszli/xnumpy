# RUN: uv run xnumpy --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @int_comparisons({{.*}} : !llvm.ptr, {{.*}} : i64, {{.*}} : i64) {
# CHECK-NEXT:     {{.*}} = llvm.icmp "eq" {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1 : i32) : i32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(2 : i32) : i32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.icmp "sgt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(3 : i32) : i32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.br {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
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
