from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _embedding(out, row):
    for i in range(row.shape[0]):
        out[i] = row[i]


def embedding_numba(d: int):
    dummy_out = np.zeros(d, dtype=np.float32)
    dummy_row = np.zeros(d, dtype=np.float32)
    _embedding(dummy_out, dummy_row)
    return _embedding
