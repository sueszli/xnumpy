# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @float_arithmetic(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_6) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_9) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   {{.*}} = llvm.fadd {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = llvm.fsub {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = llvm.fmul {{.*}}, {{.*}} : f32
# CHECK:        {{.*}} = llvm.fdiv {{.*}}, {{.*}} : f32
@proc
def float_arithmetic(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
