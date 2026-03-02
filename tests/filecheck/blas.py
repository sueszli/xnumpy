# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *
from exo.platforms.x86 import *


# CHECK: func.func @uses_select(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK:   %1 = arith.constant 0.000000e+00 : f32
# CHECK:   %5 = arith.cmpf olt, %1, %3 : f32
# CHECK:   %6 = arith.select %5, %3, %4 : f32
@proc
def uses_select(out: f32[1] @ DRAM, a: f32[1] @ DRAM, b: f32[1] @ DRAM):
    out[0] = select(0.0, a[0], a[0], b[0])
