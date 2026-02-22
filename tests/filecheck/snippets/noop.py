# RUN: FILECHECK-EXO
# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *


# CHECK: builtin.module {
# CHECK-NEXT: func.func @noop() {
# CHECK-NEXT:   func.return
# CHECK-NEXT: }
@proc
def noop():
    pass
