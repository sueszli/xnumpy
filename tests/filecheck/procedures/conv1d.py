# RUN: FILECHECK-LLVM

from __future__ import annotations

from exo import *


# CHECK: builtin.module
@proc
def conv1d(
    IC: size,
    OC: size,
    N: size,
    W: size,
    data: i32[IC, N],
    kernels: i32[OC, IC, W],
    out: i32[OC, N],
):
    # do the convolution
    for i in seq(0, OC):
        for j in seq(0, N):
            # zero out the result memory
            out[i, j] = 0.0
            for c in seq(0, IC):
                for r in seq(0, W):
                    y: i32
                    if j + r < N:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
