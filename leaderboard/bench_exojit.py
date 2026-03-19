# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

import ctypes
import random
import time
from pathlib import Path

from exo import *
from exo.libs.externs import expf, select, sqrt
from exo.stdlib.scheduling import simplify
from utils import assert_weights_match, save_times

from exojit.main import jit
from exojit.patches_exo import Stack

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD
NUM_STEPS = 1000
INV_SCALE = 1.0 / HEAD_DIM**0.5
CAUSAL_MASK_VALUE = -1e10
LR_T = [0.01 * (1.0 - step / NUM_STEPS) for step in range(NUM_STEPS)]
BC1 = [1.0 - 0.85 ** (step + 1) for step in range(NUM_STEPS)]
BC2 = [1.0 - 0.99 ** (step + 1) for step in range(NUM_STEPS)]


class _Ptr:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def array_numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def _ctype(dtype):
    if dtype is float:
        return ctypes.c_double
    if dtype is int:
        return ctypes.c_int32
    raise TypeError(dtype)


class Tensor:
    __slots__ = ("shape", "dtype", "_ctype", "_buf", "_offset", "_size")

    def __init__(self, shape: tuple[int, ...], dtype=float, *, buffer=None, offset: int = 0, fill=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ctype = _ctype(dtype)
        self._offset = offset
        self._size = array_numel(self.shape)
        self._buf = (self._ctype * self._size)() if buffer is None else buffer
        if buffer is None and fill not in (None, 0, 0.0):
            for i in range(self._size):
                self._buf[i] = fill

    @property
    def ctypes(self):
        return _Ptr(ctypes.addressof(self._buf) + self._offset * ctypes.sizeof(self._ctype))

    def _flat_index(self, key) -> int:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != len(self.shape):
            raise IndexError(key)
        if len(key) == 1:
            return self._offset + key[0]
        if len(key) == 2:
            return self._offset + key[0] * self.shape[1] + key[1]
        if len(key) == 3:
            return self._offset + (key[0] * self.shape[1] + key[1]) * self.shape[2] + key[2]
        raise ValueError(self.shape)

    def __getitem__(self, key):
        return self._buf[self._flat_index(key)]

    def __setitem__(self, key, value):
        self._buf[self._flat_index(key)] = value


def empty_array(shape: tuple[int, ...], dtype=float) -> Tensor:
    return Tensor(shape, dtype=dtype)


def zeros_array(shape: tuple[int, ...], dtype=float) -> Tensor:
    return Tensor(shape, dtype=dtype, fill=0.0 if dtype is float else 0)


def scalar_array(value: float) -> Tensor:
    out = empty_array((1,), dtype=float)
    out[0] = value
    return out


def flat_view(x: Tensor, offset: int, shape: tuple[int, ...]) -> Tensor:
    return Tensor(shape, dtype=x.dtype, buffer=x._buf, offset=x._offset + offset)


def random_matrix(rows: int, cols: int, std: float = 0.08) -> Tensor:
    out = empty_array((rows, cols), dtype=float)
    for i in range(out._size):
        out._buf[i] = random.gauss(0.0, std)
    return out


RMS_INV_N = scalar_array(1.0 / N_EMBED)
RMS_EPS = scalar_array(1e-5)
ADAM_B1 = scalar_array(0.85)
ADAM_B2 = scalar_array(0.99)
ADAM_EPS = scalar_array(1e-8)


@proc
def _attn_fwd_fused(
    out: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM,
    out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    x: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    wq: f64[N_EMBED, N_EMBED] @ DRAM,
    wk: f64[N_EMBED, N_EMBED] @ DRAM,
    wv: f64[N_EMBED, N_EMBED] @ DRAM,
    wo: f64[N_EMBED, N_EMBED] @ DRAM,
):
    for i in seq(0, BLOCK_SIZE):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N_EMBED):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * (1.0 / N_EMBED) + 1e-5)
        rms[i, 0] = scale
        for j in seq(0, N_EMBED):
            xn[i, j] = x[i, j] * scale

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc_q: f64 @ Stack
                acc_k: f64 @ Stack
                acc_v: f64 @ Stack
                acc_q = 0.0
                acc_k = 0.0
                acc_v = 0.0
                for e in seq(0, N_EMBED):
                    acc_q += xn[t, e] * wq[h * HEAD_DIM + d, e]
                    acc_k += xn[t, e] * wk[h * HEAD_DIM + d, e]
                    acc_v += xn[t, e] * wv[h * HEAD_DIM + d, e]
                q[h, t, d] = acc_q
                k[h, t, d] = acc_k
                v[h, t, d] = acc_v

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            mx: f64 @ Stack
            sum_val: f64 @ Stack
            logit: f64 @ Stack
            t0: f64 @ Stack

            mx = CAUSAL_MASK_VALUE
            for s in seq(0, BLOCK_SIZE):
                if s > t:
                    logit = CAUSAL_MASK_VALUE
                else:
                    logit = 0.0
                    for d in seq(0, HEAD_DIM):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * INV_SCALE
                attn_w[h, t, s] = logit
                mx = select(mx, attn_w[h, t, s], attn_w[h, t, s], mx)

            sum_val = 0.0
            for s in seq(0, BLOCK_SIZE):
                t0 = attn_w[h, t, s] - mx
                attn_w[h, t, s] = expf(t0)
                sum_val += attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                attn_w[h, t, s] = attn_w[h, t, s] / sum_val

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out_flat[t, h * HEAD_DIM + d] = acc

    for t in seq(0, BLOCK_SIZE):
        for j in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for e in seq(0, N_EMBED):
                acc += out_flat[t, e] * wo[j, e]
            out[t, j] = acc + x[t, j]


@proc
def _mlp_fwd_fused(
    out: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM,
    h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM,
    x: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM,
    fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM,
    inv_n: f64[1] @ DRAM,
    eps: f64[1] @ DRAM,
):
    for i in seq(0, BLOCK_SIZE):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N_EMBED):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N_EMBED):
            xn[i, j] = x[i, j] * scale

    for t in seq(0, BLOCK_SIZE):
        for j in seq(0, 4 * N_EMBED):
            acc0: f64 @ Stack
            acc0 = 0.0
            for e in seq(0, N_EMBED):
                acc0 += xn[t, e] * fc1[j, e]
            h_pre[t, j] = acc0
            h[t, j] = select(0.0, acc0, acc0, 0.0)

    for t in seq(0, BLOCK_SIZE):
        for j in seq(0, N_EMBED):
            acc0: f64 @ Stack
            acc0 = 0.0
            for e in seq(0, 4 * N_EMBED):
                acc0 += h[t, e] * fc2[j, e]
            out[t, j] = acc0 + x[t, j]


@proc
def _mlp_bwd_fused(
    out: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    dw1: f64[4 * N_EMBED, N_EMBED] @ DRAM,
    dw2: f64[N_EMBED, 4 * N_EMBED] @ DRAM,
    dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM,
    h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM,
    fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM,
    fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM,
    inv_n: f64[1] @ DRAM,
):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            out[t, e] = 0.0

    for j in seq(0, N_EMBED):
        for e in seq(0, 4 * N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for t in seq(0, BLOCK_SIZE):
                acc += dx[t, j] * h[t, e]
            dw2[j, e] = acc

    for e in seq(0, 4 * N_EMBED):
        for k in seq(0, N_EMBED):
            dw1[e, k] = 0.0

    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, 4 * N_EMBED):
            dh: f64 @ Stack
            dh_pre: f64 @ Stack
            dh = 0.0
            for j in seq(0, N_EMBED):
                dh += dx[t, j] * fc2[j, e]
            dh_pre = select(0.0, h_pre[t, e], dh, 0.0)
            for k in seq(0, N_EMBED):
                dw1[e, k] += dh_pre * xn[t, k]
                out[t, k] += dh_pre * fc1[e, k]

    for i in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N_EMBED):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N_EMBED):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def _adam(
    N: size,
    param: f64[N] @ DRAM,
    grad: f64[N] @ DRAM,
    m: f64[N] @ DRAM,
    v: f64[N] @ DRAM,
    b1: f64[1] @ DRAM,
    b2: f64[1] @ DRAM,
    eps: f64[1] @ DRAM,
    lr: f64[1] @ DRAM,
    beta1_t: f64[1] @ DRAM,
    beta2_t: f64[1] @ DRAM,
):
    inv_b1: f64 @ Stack
    inv_b2: f64 @ Stack
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_b1 = 1.0 - b1[0]
    inv_b2 = 1.0 - b2[0]
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in seq(0, N):
        g: f64 @ Stack
        m_val: f64 @ Stack
        v_val: f64 @ Stack
        m_hat: f64 @ Stack
        v_hat: f64 @ Stack
        g = grad[i]
        m_val = b1[0] * m[i] + inv_b1 * g
        v_val = b2[0] * v[i] + inv_b2 * g * g
        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t
        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + eps[0])
        m[i] = m_val
        v[i] = v_val


@proc
def _attn_bwd_fused(
    out: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    dwq: f64[N_EMBED, N_EMBED] @ DRAM,
    dwk: f64[N_EMBED, N_EMBED] @ DRAM,
    dwv: f64[N_EMBED, N_EMBED] @ DRAM,
    dwo: f64[N_EMBED, N_EMBED] @ DRAM,
    dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
    attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM,
    out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    wq: f64[N_EMBED, N_EMBED] @ DRAM,
    wk: f64[N_EMBED, N_EMBED] @ DRAM,
    wv: f64[N_EMBED, N_EMBED] @ DRAM,
    wo: f64[N_EMBED, N_EMBED] @ DRAM,
):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            out[t, e] = 0.0

    for row in seq(0, N_EMBED):
        for e in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for t in seq(0, BLOCK_SIZE):
                acc += dx[t, row] * out_flat[t, e]
            dwo[row, e] = acc

    for row in seq(0, N_EMBED):
        for e in seq(0, N_EMBED):
            dwq[row, e] = 0.0
            dwk[row, e] = 0.0
            dwv[row, e] = 0.0

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for j in seq(0, N_EMBED):
                    acc += dx[t, j] * wo[j, h * HEAD_DIM + d]
                dattn_out[h, t, d] = acc

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            dq_acc: f64[HEAD_DIM] @ Stack
            dot: f64 @ Stack
            dot = 0.0
            for d in seq(0, HEAD_DIM):
                dq_acc[d] = 0.0
            for s in seq(0, BLOCK_SIZE):
                dattn_w: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * v[h, s, d]
                attn_tmp[s] = dattn_w
                dot += dattn_w * attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                dlogit: f64 @ Stack
                dlogit = attn_w[h, t, s] * (attn_tmp[s] - dot) * INV_SCALE

                for d in seq(0, HEAD_DIM):
                    dk_contrib: f64 @ Stack
                    dv_contrib: f64 @ Stack
                    dq_acc[d] += dlogit * k[h, s, d]
                    dk_contrib = dlogit * q[h, t, d]
                    dv_contrib = attn_w[h, t, s] * dattn_out[h, t, d]

                    for e in seq(0, N_EMBED):
                        out[s, e] += dk_contrib * wk[h * HEAD_DIM + d, e]
                        out[s, e] += dv_contrib * wv[h * HEAD_DIM + d, e]
                        dwk[h * HEAD_DIM + d, e] += dk_contrib * xn[s, e]
                        dwv[h * HEAD_DIM + d, e] += dv_contrib * xn[s, e]

            for d in seq(0, HEAD_DIM):
                for e in seq(0, N_EMBED):
                    out[t, e] += dq_acc[d] * wq[h * HEAD_DIM + d, e]
                    dwq[h * HEAD_DIM + d, e] += dq_acc[d] * xn[t, e]

    for i in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N_EMBED):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * (1.0 / N_EMBED) * dot
        for j in seq(0, N_EMBED):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def _lm_head_step_fused(
    V: size,
    dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    dweight: f64[V, N_EMBED] @ DRAM,
    logits: f64[BLOCK_SIZE, V] @ DRAM,
    x: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    lm_head: f64[V, N_EMBED] @ DRAM,
    loss_mask: f64[BLOCK_SIZE] @ DRAM,
    inv_sum_mask: f64[1] @ DRAM,
    target0: size,
    target1: size,
    target2: size,
    target3: size,
    target4: size,
    target5: size,
    target6: size,
    target7: size,
    target8: size,
    target9: size,
    target10: size,
    target11: size,
    target12: size,
    target13: size,
    target14: size,
    target15: size,
):
    assert target0 < V
    assert target1 < V
    assert target2 < V
    assert target3 < V
    assert target4 < V
    assert target5 < V
    assert target6 < V
    assert target7 < V
    assert target8 < V
    assert target9 < V
    assert target10 < V
    assert target11 < V
    assert target12 < V
    assert target13 < V
    assert target14 < V
    assert target15 < V

    for t in seq(0, BLOCK_SIZE):
        for v_idx in seq(0, V):
            acc: f64 @ Stack
            acc = 0.0
            for e in seq(0, N_EMBED):
                acc += x[t, e] * lm_head[v_idx, e]
            logits[t, v_idx] = acc

    for t in seq(0, BLOCK_SIZE):
        mx: f64 @ Stack
        sum_val: f64 @ Stack
        scale: f64 @ Stack
        val: f64 @ Stack
        inv_denom: f64 @ Stack

        mx = logits[t, 0]
        for v_idx in seq(1, V):
            mx = select(mx, logits[t, v_idx], logits[t, v_idx], mx)

        sum_val = 0.0
        for v_idx in seq(0, V):
            val = expf(logits[t, v_idx] - mx)
            logits[t, v_idx] = val
            sum_val += val

        inv_denom = 1.0 / sum_val

        if t == 0:
            scale = loss_mask[0] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[0, v_idx] = logits[0, v_idx] * scale
            logits[0, target0] += -inv_sum_mask[0] * loss_mask[0]
        if t == 1:
            scale = loss_mask[1] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[1, v_idx] = logits[1, v_idx] * scale
            logits[1, target1] += -inv_sum_mask[0] * loss_mask[1]
        if t == 2:
            scale = loss_mask[2] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[2, v_idx] = logits[2, v_idx] * scale
            logits[2, target2] += -inv_sum_mask[0] * loss_mask[2]
        if t == 3:
            scale = loss_mask[3] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[3, v_idx] = logits[3, v_idx] * scale
            logits[3, target3] += -inv_sum_mask[0] * loss_mask[3]
        if t == 4:
            scale = loss_mask[4] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[4, v_idx] = logits[4, v_idx] * scale
            logits[4, target4] += -inv_sum_mask[0] * loss_mask[4]
        if t == 5:
            scale = loss_mask[5] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[5, v_idx] = logits[5, v_idx] * scale
            logits[5, target5] += -inv_sum_mask[0] * loss_mask[5]
        if t == 6:
            scale = loss_mask[6] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[6, v_idx] = logits[6, v_idx] * scale
            logits[6, target6] += -inv_sum_mask[0] * loss_mask[6]
        if t == 7:
            scale = loss_mask[7] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[7, v_idx] = logits[7, v_idx] * scale
            logits[7, target7] += -inv_sum_mask[0] * loss_mask[7]
        if t == 8:
            scale = loss_mask[8] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[8, v_idx] = logits[8, v_idx] * scale
            logits[8, target8] += -inv_sum_mask[0] * loss_mask[8]
        if t == 9:
            scale = loss_mask[9] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[9, v_idx] = logits[9, v_idx] * scale
            logits[9, target9] += -inv_sum_mask[0] * loss_mask[9]
        if t == 10:
            scale = loss_mask[10] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[10, v_idx] = logits[10, v_idx] * scale
            logits[10, target10] += -inv_sum_mask[0] * loss_mask[10]
        if t == 11:
            scale = loss_mask[11] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[11, v_idx] = logits[11, v_idx] * scale
            logits[11, target11] += -inv_sum_mask[0] * loss_mask[11]
        if t == 12:
            scale = loss_mask[12] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[12, v_idx] = logits[12, v_idx] * scale
            logits[12, target12] += -inv_sum_mask[0] * loss_mask[12]
        if t == 13:
            scale = loss_mask[13] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[13, v_idx] = logits[13, v_idx] * scale
            logits[13, target13] += -inv_sum_mask[0] * loss_mask[13]
        if t == 14:
            scale = loss_mask[14] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[14, v_idx] = logits[14, v_idx] * scale
            logits[14, target14] += -inv_sum_mask[0] * loss_mask[14]
        if t == 15:
            scale = loss_mask[15] * inv_sum_mask[0] * inv_denom
            for v_idx in seq(0, V):
                logits[15, v_idx] = logits[15, v_idx] * scale
            logits[15, target15] += -inv_sum_mask[0] * loss_mask[15]

    for v_idx in seq(0, V):
        for e in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for t in seq(0, BLOCK_SIZE):
                acc += logits[t, v_idx] * x[t, e]
            dweight[v_idx, e] = acc

    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for v_idx in seq(0, V):
                acc += logits[t, v_idx] * lm_head[v_idx, e]
            dx[t, e] = acc


@proc
def _embed_rms_fwd_tokens(
    V: size,
    emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    out: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    wte: f64[V, N_EMBED] @ DRAM,
    wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    input0: size,
    input1: size,
    input2: size,
    input3: size,
    input4: size,
    input5: size,
    input6: size,
    input7: size,
    input8: size,
    input9: size,
    input10: size,
    input11: size,
    input12: size,
    input13: size,
    input14: size,
    input15: size,
):
    assert input0 < V
    assert input1 < V
    assert input2 < V
    assert input3 < V
    assert input4 < V
    assert input5 < V
    assert input6 < V
    assert input7 < V
    assert input8 < V
    assert input9 < V
    assert input10 < V
    assert input11 < V
    assert input12 < V
    assert input13 < V
    assert input14 < V
    assert input15 < V

    for e in seq(0, N_EMBED):
        emb[0, e] = wte[input0, e] + wpe[0, e]
        emb[1, e] = wte[input1, e] + wpe[1, e]
        emb[2, e] = wte[input2, e] + wpe[2, e]
        emb[3, e] = wte[input3, e] + wpe[3, e]
        emb[4, e] = wte[input4, e] + wpe[4, e]
        emb[5, e] = wte[input5, e] + wpe[5, e]
        emb[6, e] = wte[input6, e] + wpe[6, e]
        emb[7, e] = wte[input7, e] + wpe[7, e]
        emb[8, e] = wte[input8, e] + wpe[8, e]
        emb[9, e] = wte[input9, e] + wpe[9, e]
        emb[10, e] = wte[input10, e] + wpe[10, e]
        emb[11, e] = wte[input11, e] + wpe[11, e]
        emb[12, e] = wte[input12, e] + wpe[12, e]
        emb[13, e] = wte[input13, e] + wpe[13, e]
        emb[14, e] = wte[input14, e] + wpe[14, e]
        emb[15, e] = wte[input15, e] + wpe[15, e]

    for i in seq(0, BLOCK_SIZE):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N_EMBED):
            sumsq += emb[i, j] * emb[i, j]
        scale = 1.0 / sqrt(sumsq * (1.0 / N_EMBED) + 1e-5)
        rms[i, 0] = scale
        for j in seq(0, N_EMBED):
            out[i, j] = emb[i, j] * scale


@proc
def _embed_rms_bwd_tokens(
    V: size,
    g_wte: f64[V, N_EMBED] @ DRAM,
    g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    x: f64[BLOCK_SIZE, N_EMBED] @ DRAM,
    rms: f64[BLOCK_SIZE, 1] @ DRAM,
    input0: size,
    input1: size,
    input2: size,
    input3: size,
    input4: size,
    input5: size,
    input6: size,
    input7: size,
    input8: size,
    input9: size,
    input10: size,
    input11: size,
    input12: size,
    input13: size,
    input14: size,
    input15: size,
):
    assert input0 < V
    assert input1 < V
    assert input2 < V
    assert input3 < V
    assert input4 < V
    assert input5 < V
    assert input6 < V
    assert input7 < V
    assert input8 < V
    assert input9 < V
    assert input10 < V
    assert input11 < V
    assert input12 < V
    assert input13 < V
    assert input14 < V
    assert input15 < V

    scale: f64[BLOCK_SIZE] @ Stack
    corr: f64[BLOCK_SIZE] @ Stack

    for i in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        dot = 0.0
        scale[i] = rms[i, 0]
        for j in seq(0, N_EMBED):
            dot += dout[i, j] * x[i, j]
        corr[i] = scale[i] * scale[i] * scale[i] * (1.0 / N_EMBED) * dot

    for e in seq(0, N_EMBED):
        dx0: f64 @ Stack
        dx1: f64 @ Stack
        dx2: f64 @ Stack
        dx3: f64 @ Stack
        dx4: f64 @ Stack
        dx5: f64 @ Stack
        dx6: f64 @ Stack
        dx7: f64 @ Stack
        dx8: f64 @ Stack
        dx9: f64 @ Stack
        dx10: f64 @ Stack
        dx11: f64 @ Stack
        dx12: f64 @ Stack
        dx13: f64 @ Stack
        dx14: f64 @ Stack
        dx15: f64 @ Stack

        dx0 = dout[0, e] * scale[0] - x[0, e] * corr[0]
        dx1 = dout[1, e] * scale[1] - x[1, e] * corr[1]
        dx2 = dout[2, e] * scale[2] - x[2, e] * corr[2]
        dx3 = dout[3, e] * scale[3] - x[3, e] * corr[3]
        dx4 = dout[4, e] * scale[4] - x[4, e] * corr[4]
        dx5 = dout[5, e] * scale[5] - x[5, e] * corr[5]
        dx6 = dout[6, e] * scale[6] - x[6, e] * corr[6]
        dx7 = dout[7, e] * scale[7] - x[7, e] * corr[7]
        dx8 = dout[8, e] * scale[8] - x[8, e] * corr[8]
        dx9 = dout[9, e] * scale[9] - x[9, e] * corr[9]
        dx10 = dout[10, e] * scale[10] - x[10, e] * corr[10]
        dx11 = dout[11, e] * scale[11] - x[11, e] * corr[11]
        dx12 = dout[12, e] * scale[12] - x[12, e] * corr[12]
        dx13 = dout[13, e] * scale[13] - x[13, e] * corr[13]
        dx14 = dout[14, e] * scale[14] - x[14, e] * corr[14]
        dx15 = dout[15, e] * scale[15] - x[15, e] * corr[15]

        g_wte[input0, e] += dx0
        g_wpe[0, e] = dx0
        g_wte[input1, e] += dx1
        g_wpe[1, e] = dx1
        g_wte[input2, e] += dx2
        g_wpe[2, e] = dx2
        g_wte[input3, e] += dx3
        g_wpe[3, e] = dx3
        g_wte[input4, e] += dx4
        g_wpe[4, e] = dx4
        g_wte[input5, e] += dx5
        g_wpe[5, e] = dx5
        g_wte[input6, e] += dx6
        g_wpe[6, e] = dx6
        g_wte[input7, e] += dx7
        g_wpe[7, e] = dx7
        g_wte[input8, e] += dx8
        g_wpe[8, e] = dx8
        g_wte[input9, e] += dx9
        g_wpe[9, e] = dx9
        g_wte[input10, e] += dx10
        g_wpe[10, e] = dx10
        g_wte[input11, e] += dx11
        g_wpe[11, e] = dx11
        g_wte[input12, e] += dx12
        g_wpe[12, e] = dx12
        g_wte[input13, e] += dx13
        g_wpe[13, e] = dx13
        g_wte[input14, e] += dx14
        g_wpe[14, e] = dx14
        g_wte[input15, e] += dx15
        g_wpe[15, e] = dx15


ATTN_FWD = jit(simplify(_attn_fwd_fused))._raw
ATTN_BWD = jit(simplify(_attn_bwd_fused))._raw
MLP_FWD = jit(simplify(_mlp_fwd_fused))._raw
MLP_BWD = jit(simplify(_mlp_bwd_fused))._raw

docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1

state_dict = {
    "wte": random_matrix(vocab_size, N_EMBED),
    "wpe": random_matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": random_matrix(vocab_size, N_EMBED),
    **{f"layer{i}.attn_wq": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": random_matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": random_matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

total_params = sum(t._size for t in state_dict.values())
flat_params = empty_array((total_params,), dtype=float)
flat_ptr = flat_params.ctypes.data
elt_bytes = ctypes.sizeof(flat_params._ctype)
offset = 0
for name, tensor in state_dict.items():
    ctypes.memmove(flat_ptr + offset * elt_bytes, tensor.ctypes.data, tensor._size * elt_bytes)
    state_dict[name] = flat_view(flat_params, offset, tensor.shape)
    offset += tensor._size

flat_grads = zeros_array((total_params,), dtype=float)
grads = {}
offset = 0
for name, tensor in state_dict.items():
    grads[name] = flat_view(flat_grads, offset, tensor.shape)
    offset += tensor._size

opt_m = zeros_array((total_params,), dtype=float)
opt_v = zeros_array((total_params,), dtype=float)
opt_lr = empty_array((1,), dtype=float)
opt_bc1 = empty_array((1,), dtype=float)
opt_bc2 = empty_array((1,), dtype=float)

EMB = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
RMS_INIT = empty_array((BLOCK_SIZE, 1), dtype=float)
X0 = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
X1 = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
LOGITS = empty_array((BLOCK_SIZE, vocab_size), dtype=float)
ATTN_XN = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
ATTN_RMS = empty_array((BLOCK_SIZE, 1), dtype=float)
Q = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)
K = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)
V_BUF = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)
ATTN_W = empty_array((N_HEAD, BLOCK_SIZE, BLOCK_SIZE), dtype=float)
OUT_FLAT = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
MLP_XN = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
MLP_RMS = empty_array((BLOCK_SIZE, 1), dtype=float)
H_PRE = empty_array((BLOCK_SIZE, 4 * N_EMBED), dtype=float)
H_BUF = empty_array((BLOCK_SIZE, 4 * N_EMBED), dtype=float)
DX0 = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
DX1 = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
DATTN_OUT = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)

LM_HEAD_STEP = jit(simplify(_lm_head_step_fused.partial_eval(V=vocab_size)), raw=True)
EMBED_RMS_FWD = jit(simplify(_embed_rms_fwd_tokens.partial_eval(V=vocab_size)), raw=True)
EMBED_RMS_BWD = jit(simplify(_embed_rms_bwd_tokens.partial_eval(V=vocab_size)), raw=True)
ADAM_STEP = jit(simplify(_adam.partial_eval(N=total_params)))._raw

P_WTE_PTR = state_dict["wte"].ctypes.data
P_WPE_PTR = state_dict["wpe"].ctypes.data
P_LM_HEAD_PTR = state_dict["lm_head"].ctypes.data
P_ATTN_WQ_PTR = state_dict["layer0.attn_wq"].ctypes.data
P_ATTN_WK_PTR = state_dict["layer0.attn_wk"].ctypes.data
P_ATTN_WV_PTR = state_dict["layer0.attn_wv"].ctypes.data
P_ATTN_WO_PTR = state_dict["layer0.attn_wo"].ctypes.data
P_MLP_FC1_PTR = state_dict["layer0.mlp_fc1"].ctypes.data
P_MLP_FC2_PTR = state_dict["layer0.mlp_fc2"].ctypes.data

G_WTE_PTR = grads["wte"].ctypes.data
G_WPE_PTR = grads["wpe"].ctypes.data
G_LM_HEAD_PTR = grads["lm_head"].ctypes.data
G_ATTN_WQ_PTR = grads["layer0.attn_wq"].ctypes.data
G_ATTN_WK_PTR = grads["layer0.attn_wk"].ctypes.data
G_ATTN_WV_PTR = grads["layer0.attn_wv"].ctypes.data
G_ATTN_WO_PTR = grads["layer0.attn_wo"].ctypes.data
G_MLP_FC1_PTR = grads["layer0.mlp_fc1"].ctypes.data
G_MLP_FC2_PTR = grads["layer0.mlp_fc2"].ctypes.data

EMB_PTR = EMB.ctypes.data
RMS_INIT_PTR = RMS_INIT.ctypes.data
X0_PTR = X0.ctypes.data
X1_PTR = X1.ctypes.data
LOGITS_PTR = LOGITS.ctypes.data
ATTN_XN_PTR = ATTN_XN.ctypes.data
ATTN_RMS_PTR = ATTN_RMS.ctypes.data
Q_PTR = Q.ctypes.data
K_PTR = K.ctypes.data
V_BUF_PTR = V_BUF.ctypes.data
ATTN_W_PTR = ATTN_W.ctypes.data
OUT_FLAT_PTR = OUT_FLAT.ctypes.data
MLP_XN_PTR = MLP_XN.ctypes.data
MLP_RMS_PTR = MLP_RMS.ctypes.data
H_PRE_PTR = H_PRE.ctypes.data
H_BUF_PTR = H_BUF.ctypes.data
DX0_PTR = DX0.ctypes.data
DX1_PTR = DX1.ctypes.data
DATTN_OUT_PTR = DATTN_OUT.ctypes.data

OPT_PARAMS_PTR = flat_params.ctypes.data
OPT_GRADS_PTR = flat_grads.ctypes.data
OPT_M_PTR = opt_m.ctypes.data
OPT_V_PTR = opt_v.ctypes.data
OPT_LR_PTR = opt_lr.ctypes.data
OPT_BC1_PTR = opt_bc1.ctypes.data
OPT_BC2_PTR = opt_bc2.ctypes.data

G_WTE_BYTES = grads["wte"]._size * elt_bytes
G_WPE_BYTES = grads["wpe"]._size * elt_bytes

ATTN_FWD_ARGS = (
    X1_PTR,
    ATTN_XN_PTR,
    ATTN_RMS_PTR,
    Q_PTR,
    K_PTR,
    V_BUF_PTR,
    ATTN_W_PTR,
    OUT_FLAT_PTR,
    X0_PTR,
    P_ATTN_WQ_PTR,
    P_ATTN_WK_PTR,
    P_ATTN_WV_PTR,
    P_ATTN_WO_PTR,
)
MLP_FWD_ARGS = (
    DX0_PTR,
    MLP_XN_PTR,
    MLP_RMS_PTR,
    H_PRE_PTR,
    H_BUF_PTR,
    X1_PTR,
    P_MLP_FC1_PTR,
    P_MLP_FC2_PTR,
    RMS_INV_N.ctypes.data,
    RMS_EPS.ctypes.data,
)
MLP_BWD_ARGS = (
    DX0_PTR,
    G_MLP_FC1_PTR,
    G_MLP_FC2_PTR,
    DX1_PTR,
    X1_PTR,
    MLP_XN_PTR,
    MLP_RMS_PTR,
    H_PRE_PTR,
    H_BUF_PTR,
    P_MLP_FC1_PTR,
    P_MLP_FC2_PTR,
    RMS_INV_N.ctypes.data,
)
ATTN_BWD_ARGS = (
    DX1_PTR,
    G_ATTN_WQ_PTR,
    G_ATTN_WK_PTR,
    G_ATTN_WV_PTR,
    G_ATTN_WO_PTR,
    DATTN_OUT_PTR,
    DX0_PTR,
    X0_PTR,
    ATTN_XN_PTR,
    ATTN_RMS_PTR,
    Q_PTR,
    K_PTR,
    V_BUF_PTR,
    ATTN_W_PTR,
    OUT_FLAT_PTR,
    P_ATTN_WQ_PTR,
    P_ATTN_WK_PTR,
    P_ATTN_WV_PTR,
    P_ATTN_WO_PTR,
)
ADAM_ARGS = (
    OPT_PARAMS_PTR,
    OPT_GRADS_PTR,
    OPT_M_PTR,
    OPT_V_PTR,
    ADAM_B1.ctypes.data,
    ADAM_B2.ctypes.data,
    ADAM_EPS.ctypes.data,
    OPT_LR_PTR,
    OPT_BC1_PTR,
    OPT_BC2_PTR,
)

c2i = {ch: i for i, ch in enumerate(uchars)}
bos = vocab_size - 1


def tokenize(doc: str):
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    inputs = [0] * BLOCK_SIZE
    targets = [0] * BLOCK_SIZE
    loss_mask = zeros_array((BLOCK_SIZE,), dtype=float)
    for i in range(n):
        inputs[i] = tokens[i]
        targets[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    inv_sum_mask = scalar_array(1.0 / max(1, n))
    return (
        (EMB_PTR, X0_PTR, RMS_INIT_PTR, P_WTE_PTR, P_WPE_PTR, *inputs),
        (DX1_PTR, G_LM_HEAD_PTR, LOGITS_PTR, DX0_PTR, P_LM_HEAD_PTR, loss_mask.ctypes.data, inv_sum_mask.ctypes.data, *targets),
        (G_WTE_PTR, G_WPE_PTR, DX1_PTR, EMB_PTR, RMS_INIT_PTR, *inputs),
        loss_mask,
        inv_sum_mask,
    )


tokenized = [tokenize(doc) for doc in docs]

step_times = []
for step in range(NUM_STEPS):
    opt_lr._buf[0] = LR_T[step]
    opt_bc1._buf[0] = BC1[step]
    opt_bc2._buf[0] = BC2[step]
    embed_args, lm_head_args, embed_bwd_args, _, _ = tokenized[step % len(tokenized)]
    ctypes.memset(G_WTE_PTR, 0, G_WTE_BYTES)
    ctypes.memset(G_WPE_PTR, 0, G_WPE_BYTES)
    t0 = time.perf_counter()
    EMBED_RMS_FWD(*embed_args)
    ATTN_FWD(*ATTN_FWD_ARGS)
    MLP_FWD(*MLP_FWD_ARGS)
    LM_HEAD_STEP(*lm_head_args)
    MLP_BWD(*MLP_BWD_ARGS)
    ATTN_BWD(*ATTN_BWD_ARGS)
    EMBED_RMS_BWD(*embed_bwd_args)
    ADAM_STEP(*ADAM_ARGS)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)


class W:
    __slots__ = ("data",)

    def __init__(self, data: float):
        self.data = data


assert_weights_match({name: [[W(float(tensor[i, j])) for j in range(tensor.shape[1])] for i in range(tensor.shape[0])] for name, tensor in state_dict.items()})
