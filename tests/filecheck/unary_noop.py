# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @unary_noop(%0 : !llvm.ptr) {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def unary_noop(x: f32[16]):
    pass
