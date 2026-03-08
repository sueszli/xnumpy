# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      llvm.func @i8_copy({{.*}}) {
# CHECK:        "llvm.load"({{.*}}) <{ordering = 0 : i64}> : (!llvm.ptr) -> i8
# CHECK:        "llvm.store"({{.*}}) <{ordering = 0 : i64}> : (i8, !llvm.ptr) -> ()
@proc
def i8_copy(out: i8[8] @ DRAM, x: i8[8] @ DRAM):
    for i in seq(0, 8):
        out[i] = x[i]
