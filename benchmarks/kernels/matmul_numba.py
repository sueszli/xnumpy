from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _matmul(C, A, B):
    M, K = A.shape
    _, N = B.shape
    for i in range(M):
        for j in range(N):
            acc = np.float32(0.0)
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def matmul_numba(M: int, K: int, N: int):
    dummy_A = np.zeros((M, K), dtype=np.float32)
    dummy_B = np.zeros((K, N), dtype=np.float32)
    dummy_C = np.zeros((M, N), dtype=np.float32)
    _matmul(dummy_C, dummy_A, dummy_B)  # warm up
    return _matmul
