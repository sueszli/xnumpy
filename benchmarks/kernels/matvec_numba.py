from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _matvec(y, W, x):
    M, N = W.shape
    for i in range(M):
        acc = np.float32(0.0)
        for j in range(N):
            acc += W[i, j] * x[j]
        y[i] = acc


def matvec_numba(M: int, N: int):
    dummy_W = np.zeros((M, N), dtype=np.float32)
    dummy_x = np.zeros(N, dtype=np.float32)
    dummy_y = np.zeros(M, dtype=np.float32)
    _matvec(dummy_y, dummy_W, dummy_x)
    return _matvec
