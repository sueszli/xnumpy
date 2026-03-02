# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.alloc with multiple types (f32 and i32), scalar alloc + read + assign
# Lowering: two malloc calls, f32 and i32 stores/loads

from __future__ import annotations

from exo import *


# CHECK:      func.func @multi_type_alloc(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr) {
# CHECK:        %0 = arith.constant 1 : i64
# CHECK-NEXT:   %offset_pointer_2 = "llvm.call"(%0) <{callee = @malloc, {{.*}}}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %offset_pointer_3 = "llvm.call"(%0) <{callee = @malloc, {{.*}}}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %1 = arith.constant 3.140000e+00 : f32
# CHECK:        "llvm.store"(%1, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        %4 = arith.constant 42 : i32
# CHECK:        "llvm.store"(%4, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK:        %5 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK:        "llvm.store"(%5, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        "llvm.call"(%offset_pointer_2) <{callee = @free, {{.*}}}> : (!llvm.ptr) -> ()
# CHECK:        %6 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
# CHECK:        "llvm.store"(%6, {{.*}}) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
# CHECK:        "llvm.call"(%offset_pointer_3) <{callee = @free, {{.*}}}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def multi_type_alloc(out_f: f32[1] @ DRAM, out_i: i32[1] @ DRAM):
    tmp_f: f32
    tmp_i: i32
    tmp_f = 3.14
    tmp_i = 42
    out_f[0] = tmp_f
    out_i[0] = tmp_i
