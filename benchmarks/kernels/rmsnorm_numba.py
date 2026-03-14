from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _rmsnorm_sumsq(sumsq, x):
    s = np.float32(0.0)
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    sumsq[0] = s


@nb.njit(cache=True, fastmath=True)
def _rmsnorm_scale(out, x, scale):
    s = scale[0]
    for i in range(x.shape[0]):
        out[i] = x[i] * s


def rmsnorm_numba(n: int):
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_sq = np.zeros(1, dtype=np.float32)
    dummy_sc = np.zeros(1, dtype=np.float32)
    dummy_out = np.zeros(n, dtype=np.float32)
    _rmsnorm_sumsq(dummy_sq, dummy_x)
    _rmsnorm_scale(dummy_out, dummy_x, dummy_sc)
    return _rmsnorm_sumsq, _rmsnorm_scale
