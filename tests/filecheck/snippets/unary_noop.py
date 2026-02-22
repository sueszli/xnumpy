# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @unary_noop(%0 : memref<16xf32, "DRAM">) {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def unary_noop(x: f32[16]):
    pass
