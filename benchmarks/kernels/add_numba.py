from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _add(z, x, y):
    for i in range(x.shape[0]):
        z[i] = x[i] + y[i]


def add_numba(n: int):
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_y = np.zeros(n, dtype=np.float32)
    dummy_z = np.zeros(n, dtype=np.float32)
    _add(dummy_z, dummy_x, dummy_y)
    return _add
