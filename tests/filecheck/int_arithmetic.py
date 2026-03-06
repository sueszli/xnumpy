# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_arithmetic(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_6) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK:        {{.*}} = "llvm.load"(%offset_pointer_9) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK-NEXT:   {{.*}} = llvm.add {{.*}}, {{.*}} : i32
# CHECK:        {{.*}} = llvm.sub {{.*}}, {{.*}} : i32
# CHECK:        {{.*}} = llvm.mul {{.*}}, {{.*}} : i32
# CHECK:        {{.*}} = llvm.sdiv {{.*}}, {{.*}} : i32
@proc
def int_arithmetic(out: i32[1] @ DRAM, a: i32[1] @ DRAM, b: i32[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] - b[0]
    out[0] = a[0] * b[0]
    out[0] = a[0] / b[0]
