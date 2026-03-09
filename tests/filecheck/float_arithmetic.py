# RUN: uv run xnumpy --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @float_arithmetic({{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr, {{.*}} : !llvm.ptr) {
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}} : i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.fadd {{.*}}, {{.*}} {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.fsub {{.*}}, {{.*}} {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.fmul {{.*}}, {{.*}} {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     {{.*}} = llvm.fdiv {{.*}}, {{.*}} {fastmathFlags = #llvm.fastmath<fast>} : f32
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def float_arithmetic(out: f32[1] @ DRAM, x: f32[1] @ DRAM, y: f32[1] @ DRAM):
    out[0] = x[0] + y[0]
    out[0] = x[0] - y[0]
    out[0] = x[0] * y[0]
    out[0] = x[0] / y[0]
