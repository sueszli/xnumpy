# RUN: uv run xnumpy --mlir -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: llvm.func @noop() {
# CHECK-NEXT:   llvm.return
# CHECK-NEXT: }
@proc
def noop():
    pass
