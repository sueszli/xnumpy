# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
#   "numpy",  # just for ffi
# ]
# ///

import numpy as np
from exo import *

from exojit.main import compile_jit


@proc
def matmul(C: f32[128, 128] @ DRAM, A: f32[128, 128] @ DRAM, B: f32[128, 128] @ DRAM):
    for i in seq(0, 128):
        for j in seq(0, 128):
            C[i, j] = 0.0
            for k in seq(0, 128):
                C[i, j] += A[i, k] * B[k, j]


A = np.random.randn(128, 128).astype(np.float32)
B = np.random.randn(128, 128).astype(np.float32)
C = np.zeros((128, 128), np.float32)
compile_jit(matmul)[matmul.name()](C, A, B)
print(C)
