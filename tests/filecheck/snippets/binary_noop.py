# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @binary_noop(%0 : memref<16xf32, "DRAM">, %1 : memref<16xf32, "DRAM">) {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def binary_noop(x: f32[16], y: f32[16]):
    pass
