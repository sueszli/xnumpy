from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, fastmath=True)
def _adam(param, grad, m, v, b1, b2, eps, lr, beta1_t, beta2_t):
    inv_b1 = np.float32(1.0) - b1[0]
    inv_b2 = np.float32(1.0) - b2[0]
    inv_beta1_t = np.float32(1.0) / beta1_t[0]
    inv_beta2_t = np.float32(1.0) / beta2_t[0]

    for i in range(param.shape[0]):
        g = grad[i]

        m_val = b1[0] * m[i] + inv_b1 * g
        m[i] = m_val

        v_val = b2[0] * v[i] + inv_b2 * g * g
        v[i] = v_val

        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t

        param[i] = param[i] - lr[0] * m_hat / (math.sqrt(v_hat) + eps[0])


def adam_numba(n: int):
    dummy_param = np.zeros(n, dtype=np.float32)
    dummy_grad = np.zeros(n, dtype=np.float32)
    dummy_m = np.zeros(n, dtype=np.float32)
    dummy_v = np.zeros(n, dtype=np.float32)
    dummy_b1 = np.array([0.9], dtype=np.float32)
    dummy_b2 = np.array([0.999], dtype=np.float32)
    dummy_eps = np.array([1e-8], dtype=np.float32)
    dummy_lr = np.array([0.001], dtype=np.float32)
    dummy_beta1_t = np.array([0.9], dtype=np.float32)
    dummy_beta2_t = np.array([0.999], dtype=np.float32)
    _adam(dummy_param, dummy_grad, dummy_m, dummy_v, dummy_b1, dummy_b2, dummy_eps, dummy_lr, dummy_beta1_t, dummy_beta2_t)
    return _adam
