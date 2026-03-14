from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _weighted_sum(out, weights, V):
    T, D = V.shape
    for j in range(D):
        acc = np.float32(0.0)
        for t in range(T):
            acc += weights[t] * V[t, j]
        out[j] = acc


def weighted_sum_numba(T: int, D: int):
    dummy_w = np.zeros(T, dtype=np.float32)
    dummy_V = np.zeros((T, D), dtype=np.float32)
    dummy_out = np.zeros(D, dtype=np.float32)
    _weighted_sum(dummy_out, dummy_w, dummy_V)
    return _weighted_sum
