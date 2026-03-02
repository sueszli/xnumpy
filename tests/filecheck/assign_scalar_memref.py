# RUN: uv run xdsl-exo -o - %s | filecheck %s

# Exercises: exo.alloc (scalar → memref[1]), memref.store (scalar memref),
#            memref.load (scalar memref), reduce (scalar += via load + add + store)
# Lowering: scalar alloc → malloc(1), scalar read/assign → indexed with zero index

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @assign_scalar_memref() {
# CHECK-NEXT:   %0 = arith.constant 1 : i64
# CHECK-NEXT:   %offset_pointer = "llvm.call"(%0) <{callee = @malloc, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (i64) -> !llvm.ptr
# CHECK-NEXT:   %1 = arith.constant 1.000000e+00 : f32
# CHECK-NEXT:   %2 = arith.constant 0 : i64
# CHECK:        "llvm.store"(%1, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        %4 = "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:   %5 = arith.addf %4, %1 : f32
# CHECK:        "llvm.store"(%5, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK:        "llvm.call"(%offset_pointer) <{callee = @free, {{.*}}}> : (!llvm.ptr) -> ()
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
# CHECK-NEXT: llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT: llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def assign_scalar_memref():
    x: f32
    x = 1.0
    x = x + 1.0
