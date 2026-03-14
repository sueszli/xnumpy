from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _relu(out, x):
    for i in range(x.shape[0]):
        out[i] = max(np.float32(0.0), x[i])


def relu_numba(n: int):
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_out = np.zeros(n, dtype=np.float32)
    _relu(dummy_out, dummy_x)
    return _relu
