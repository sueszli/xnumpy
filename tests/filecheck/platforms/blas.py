# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *
from exo.platforms.x86 import *


# CHECK: func.func @uses_select(%0 : !llvm.ptr, %1 : !llvm.ptr, %2 : !llvm.ptr) {
# CHECK:   %3 = arith.constant 0.000000e+00 : f32
# CHECK:   %10 = arith.cmpf olt, %3, %6 : f32
# CHECK:   %11 = arith.select %10, %7, %9 : f32
@proc
def uses_select(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = select(0.0, a[0], a[0], b[0])
