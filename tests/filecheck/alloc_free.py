# RUN: uv run xnumpy --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @alloc_free({{.*}} : i64, {{.*}} : !llvm.ptr) {
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}} : i64)
# CHECK-NEXT:   {{.*}}({{.*}} : i64):
# CHECK-NEXT:     {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     {{.*}} = "llvm.call"({{.*}}) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     "llvm.call"({{.*}}) <{callee = @free, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
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
def alloc_free(N: size, x: f32[N] @ DRAM):
    for i in seq(0, N):
        tmp: f32
        tmp = x[i]
        x[i] = tmp
