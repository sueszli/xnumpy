from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _dot(result, q, k):
    acc = np.float32(0.0)
    for i in range(q.shape[0]):
        acc += q[i] * k[i]
    result[0] = acc


def dot_numba(n: int):
    dummy_q = np.zeros(n, dtype=np.float32)
    dummy_k = np.zeros(n, dtype=np.float32)
    dummy_r = np.zeros(1, dtype=np.float32)
    _dot(dummy_r, dummy_q, dummy_k)
    return _dot
