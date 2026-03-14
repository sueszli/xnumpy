from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _ce_max(mx, x):
    m = x[0]
    for i in range(1, x.shape[0]):
        if x[i] > m:
            m = x[i]
    mx[0] = m


@nb.njit(cache=True, fastmath=True)
def _ce_sum_exp(sum_exp, x, mx):
    m = mx[0]
    s = np.float32(0.0)
    for i in range(x.shape[0]):
        s += math.exp(x[i] - m)
    sum_exp[0] = s


def cross_entropy_numba(n: int):
    dummy_x = np.zeros(n, dtype=np.float32)
    dummy_mx = np.zeros(1, dtype=np.float32)
    dummy_se = np.zeros(1, dtype=np.float32)
    _ce_max(dummy_mx, dummy_x)
    _ce_sum_exp(dummy_se, dummy_x, dummy_mx)
    return _ce_max, _ce_sum_exp
