# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      llvm.func @f64_arithmetic({{.*}}) {
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f64
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> f64
# CHECK:        llvm.fadd {{.*}} : f64
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
# CHECK:        llvm.fmul {{.*}} : f64
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (f64, !llvm.ptr) -> ()
@proc
def f64_arithmetic(out: f64[1] @ DRAM, a: f64[1] @ DRAM, b: f64[1] @ DRAM):
    out[0] = a[0] + b[0]
    out[0] = a[0] * b[0]
