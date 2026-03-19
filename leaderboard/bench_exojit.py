# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
# ]
# ///

import ctypes
import math
import random
import sys
import time
from collections import namedtuple
from functools import cache
from pathlib import Path

from exo import *
from exo.libs.externs import expf, select, sqrt
from exo.stdlib.scheduling import divide_loop, fission, reorder_loops, simplify
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
LOSS_FLOOR = sys.float_info.min
ADAM_PARAMS = {
    "LR_T": [0.01 * (1.0 - step / NUM_STEPS) for step in range(NUM_STEPS)],
    "BC1": [1.0 - 0.85 ** (step + 1) for step in range(NUM_STEPS)],
    "BC2": [1.0 - 0.99 ** (step + 1) for step in range(NUM_STEPS)],
}


AttnCache = namedtuple("AttnCache", ["x_pre", "xn", "rms", "q", "k", "v", "attn_w", "out_flat"])
MlpCache = namedtuple("MlpCache", ["x_pre", "xn", "rms", "h_pre", "h"])
FwdCache = namedtuple("FwdCache", ["input_ids", "target_ids", "loss_mask", "sum_mask", "emb", "rms_init", "x", "probs", "layer_caches"])


class _CtypesProxy:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def array_numel(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def _ctype_for_dtype(dtype):
    if dtype is float:
        return ctypes.c_double
    if dtype is int:
        return ctypes.c_int64
    raise TypeError(f"unsupported dtype: {dtype!r}")


def _zero_for_dtype(dtype):
    return 0.0 if dtype is float else 0


def _default_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class Tensor:
    __slots__ = ("shape", "dtype", "_ctype", "_buf", "_offset", "_strides")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype=float,
        *,
        buffer=None,
        offset: int = 0,
        strides: tuple[int, ...] | None = None,
        fill=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ctype = _ctype_for_dtype(dtype)
        self._offset = offset
        self._strides = _default_strides(self.shape) if strides is None else strides
        if buffer is None:
            self._buf = (self._ctype * array_numel(self.shape))()
            if fill is not None:
                for i in range(array_numel(self.shape)):
                    self._buf[i] = fill
        else:
            self._buf = buffer

    @property
    def ctypes(self):
        return _CtypesProxy(ctypes.addressof(self._buf) + self._offset * ctypes.sizeof(self._ctype))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _flat_index(self, key: tuple[int, ...]) -> int:
        if len(key) != self.ndim:
            raise IndexError(f"expected {self.ndim} indices, got {len(key)}")
        idx = self._offset
        for k, dim, stride in zip(key, self.shape, self._strides, strict=True):
            if k < 0:
                k += dim
            if k < 0 or k >= dim:
                raise IndexError("index out of range")
            idx += k * stride
        return idx

    def _scalar_at(self, key: tuple[int, ...]):
        return self._buf[self._flat_index(key)]

    def _set_scalar_at(self, key: tuple[int, ...], value):
        self._buf[self._flat_index(key)] = value

    def _row_view(self, i: int):
        if self.ndim == 0:
            raise TypeError("cannot index a scalar tensor")
        if i < 0:
            i += self.shape[0]
        if i < 0 or i >= self.shape[0]:
            raise IndexError("index out of range")
        if self.ndim == 1:
            return Tensor((1,), dtype=self.dtype, buffer=self._buf, offset=self._offset + i * self._strides[0], strides=(1,))
        return Tensor(self.shape[1:], dtype=self.dtype, buffer=self._buf, offset=self._offset + i * self._strides[0], strides=self._strides[1:])

    def _copy_new(self):
        out = Tensor(self.shape, dtype=self.dtype)
        for idx in iter_indices(self.shape):
            out[idx] = self[idx]
        return out

    def copy(self):
        return self._copy_new()

    def reshape(self, shape: tuple[int, ...]):
        if array_numel(shape) != array_numel(self.shape):
            raise ValueError("cannot reshape tensor to different size")
        return Tensor(shape, dtype=self.dtype, buffer=self._buf, offset=self._offset, strides=_default_strides(shape))

    def sum(self):
        total = 0.0 if self.dtype is float else 0
        for idx in iter_indices(self.shape):
            total += self[idx]
        return total

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        if self.ndim == 1:
            for i in range(self.shape[0]):
                yield self[i]
            return
        for i in range(self.shape[0]):
            yield self._row_view(i)

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            if self.ndim == 2 and key.ndim == 1:
                out = Tensor((key.shape[0], self.shape[1]), dtype=self.dtype)
                for i in range(key.shape[0]):
                    row = int(key[i])
                    for j in range(self.shape[1]):
                        out[i, j] = self[row, j]
                return out
            raise TypeError("unsupported tensor index")
        if isinstance(key, slice):
            if self.ndim != 1:
                raise TypeError("slice indexing only supported for 1D tensors")
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise TypeError("slice step other than 1 is unsupported")
            length = max(0, stop - start)
            return Tensor((length,), dtype=self.dtype, buffer=self._buf, offset=self._offset + start * self._strides[0], strides=(1,))
        if isinstance(key, tuple):
            if len(key) == 2 and key[1] is None and isinstance(key[0], slice):
                start, stop, step = key[0].indices(self.shape[0])
                if step != 1:
                    raise TypeError("slice step other than 1 is unsupported")
                length = max(0, stop - start)
                return Tensor((length, 1), dtype=self.dtype, buffer=self._buf, offset=self._offset + start * self._strides[0], strides=(self._strides[0], 0))
            if all(isinstance(k, int) for k in key):
                return self._scalar_at(tuple(key))
            raise TypeError("unsupported tensor index")
        if isinstance(key, int):
            if self.ndim == 1:
                return self._scalar_at((key,))
            return self._row_view(key)
        raise TypeError(f"unsupported tensor index type: {type(key).__name__}")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if self.ndim == 1:
                self._set_scalar_at((key,), value)
                return
            raise TypeError("only 1D scalar assignment by integer is supported")
        if isinstance(key, tuple) and all(isinstance(k, int) for k in key):
            self._set_scalar_at(tuple(key), value)
            return
        if isinstance(key, slice):
            if self.ndim != 1:
                raise TypeError("slice assignment only supported for 1D tensors")
            start, stop, step = key.indices(self.shape[0])
            if step != 1:
                raise TypeError("slice step other than 1 is unsupported")
            if isinstance(value, Tensor):
                if value.ndim != 1 or value.shape[0] != stop - start:
                    raise ValueError("shape mismatch in slice assignment")
                for i in range(stop - start):
                    self[start + i] = value[i]
            else:
                for i in range(start, stop):
                    self[i] = value
            return
        raise TypeError("unsupported tensor assignment")

    def _elementwise(self, other, op, in_place: bool):
        if isinstance(other, Tensor):
            if self.shape == other.shape:
                out = self if in_place else Tensor(self.shape, dtype=self.dtype)
                for idx in iter_indices(self.shape):
                    out[idx] = op(self[idx], other[idx])
                return out
            if self.ndim == 2 and other.ndim == 2 and other.shape[1] == 1 and other.shape[0] == self.shape[0]:
                out = self if in_place else Tensor(self.shape, dtype=self.dtype)
                for i in range(self.shape[0]):
                    rhs = other[i, 0]
                    for j in range(self.shape[1]):
                        out[i, j] = op(self[i, j], rhs)
                return out
            raise ValueError("shape mismatch in tensor operation")
        out = self if in_place else Tensor(self.shape, dtype=self.dtype)
        for idx in iter_indices(self.shape):
            out[idx] = op(self[idx], other)
        return out

    def __add__(self, other):
        return self._elementwise(other, lambda a, b: a + b, False)

    def __iadd__(self, other):
        return self._elementwise(other, lambda a, b: a + b, True)

    def __sub__(self, other):
        return self._elementwise(other, lambda a, b: a - b, False)

    def __isub__(self, other):
        return self._elementwise(other, lambda a, b: a - b, True)

    def __mul__(self, other):
        return self._elementwise(other, lambda a, b: a * b, False)

    def __imul__(self, other):
        return self._elementwise(other, lambda a, b: a * b, True)

    def __float__(self):
        if self.numel() != 1:
            raise TypeError("only scalar tensors can be converted to float")
        return float(self._buf[self._offset])

    def numel(self) -> int:
        return array_numel(self.shape)


BLOCK_INDEX = Tensor((BLOCK_SIZE,), dtype=int)
for i in range(BLOCK_SIZE):
    BLOCK_INDEX[i] = i


def scalar_array(value: float) -> Tensor:
    out = Tensor((1,), dtype=float)
    out[0] = value
    return out


def empty_array(shape: tuple[int, ...], dtype=float) -> Tensor:
    return Tensor(shape, dtype=dtype)


def zeros_array(shape: tuple[int, ...], dtype=float) -> Tensor:
    out = Tensor(shape, dtype=dtype)
    for idx in iter_indices(shape):
        out[idx] = _zero_for_dtype(dtype)
    return out


def empty_like_array(x: Tensor) -> Tensor:
    return empty_array(x.shape, x.dtype)


def flat_view(x: Tensor, offset: int, shape: tuple[int, ...]) -> Tensor:
    size = array_numel(shape)
    return x[offset : offset + size].reshape(shape)


def random_matrix(nout: int, nin: int, std: float = 0.08) -> Tensor:
    out = empty_array((nout, nin), dtype=float)
    for i in range(nout):
        for j in range(nin):
            out[i, j] = random.gauss(0.0, std)
    return out


RMS_INV_N = scalar_array(1.0 / N_EMBED)
RMS_EPS = scalar_array(1e-5)
ADAM_B1 = scalar_array(0.85)
ADAM_B2 = scalar_array(0.99)
ADAM_EPS = scalar_array(1e-8)
INV_SCALE_ARRAY = scalar_array(INV_SCALE)
CAUSAL_MASK_ARRAY = scalar_array(CAUSAL_MASK_VALUE)


def softmax(x: Tensor) -> Tensor:
    if x.ndim != 2:
        raise TypeError("softmax expects a 2D tensor")
    for r in range(x.shape[0]):
        mx = x[r, 0]
        for j in range(1, x.shape[1]):
            mx = max(mx, x[r, j])
        total = 0.0
        for j in range(x.shape[1]):
            val = math.exp(x[r, j] - mx)
            x[r, j] = val
            total += val
        for j in range(x.shape[1]):
            x[r, j] = x[r, j] / total
    return x


@proc
def _matmul_nt(M: size, K: size, N: size, out: f64[M, N] @ DRAM, a: f64[M, K] @ DRAM, b: f64[N, K] @ DRAM):
    for i in par(0, M):
        for j in seq(0, N):
            out[i, j] = 0.0
            for k in seq(0, K):
                out[i, j] += a[i, k] * b[j, k]


def _schedule_matmul(p, k: int, n: int):
    p = fission(p, p.find("for k in _: _").before(), n_lifts=2)
    p = reorder_loops(p, "j k")
    do_k = k > 64
    do_j = n > 64
    if do_k:
        p = divide_loop(p, "k", 64, ["ko", "ki"], perfect=True)
    if do_j:
        p = divide_loop(p, "j #1", 64, ["jo", "ji"], perfect=True)
        if do_k:
            p = reorder_loops(p, "ki jo")
    return simplify(p)


@cache
def _jit_matmul_nt(m: int, k: int, n: int):
    return jit(_schedule_matmul(_matmul_nt.partial_eval(M=m, K=k, N=n), k, n))


def matmul_nt(a: Tensor, b: Tensor, out: Tensor) -> Tensor:
    rows, inner = a.shape
    out_cols, b_inner = b.shape
    if inner != b_inner or out.shape != (rows, out_cols):
        raise ValueError("shape mismatch in matmul_nt")
    _jit_matmul_nt(rows, inner, out_cols)._raw(out.ctypes.data, a.ctypes.data, b.ctypes.data)
    return out


def iter_indices(shape: tuple[int, ...]):
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                yield (i, j)
    elif len(shape) == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    yield (i, j, k)
    else:
        raise ValueError(f"unsupported shape rank: {len(shape)}")


@proc
def _zero_1d(N: size, x: f64[N] @ DRAM):
    for i in par(0, N):
        x[i] = 0.0


def zero_array(x: Tensor) -> Tensor:
    _jit_zero_1d(x.shape[0])._raw(x.ctypes.data)
    return x


@proc
def _softmax_2d(M: size, N: size, out: f64[M, N] @ DRAM, inp: f64[M, N] @ DRAM):
    for r in seq(0, M):
        mx: f64 @ Stack
        sum_val: f64 @ Stack
        t: f64 @ Stack

        mx = inp[r, 0]
        for i in seq(1, N):
            mx = select(mx, inp[r, i], inp[r, i], mx)

        sum_val = 0.0
        for j in seq(0, N):
            t = inp[r, j] - mx
            out[r, j] = expf(t)
            sum_val += out[r, j]

        for k in seq(0, N):
            out[r, k] = out[r, k] / sum_val


@cache
def _jit_softmax_2d(m: int, n: int):
    return jit(simplify(_softmax_2d.partial_eval(M=m, N=n)))


@cache
def _jit_zero_1d(n: int):
    return jit(simplify(_zero_1d.partial_eval(N=n)))


def cross_entropy_loss(probs: Tensor, target_ids: Tensor, loss_mask: Tensor, sum_mask: float) -> float:
    total = 0.0
    for i in range(BLOCK_SIZE):
        if loss_mask[i] != 0:
            value = probs[i, int(target_ids[i])]
            total += math.log(max(LOSS_FLOOR, min(value, 1.0)))
    return float(-total / sum_mask)


@proc
def _rmsnorm_fwd(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, inp: f64[M, N] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N):
            sumsq += inp[i, j] * inp[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = inp[i, j] * scale


@proc
def _rmsnorm_bwd(M: size, N: size, dx: f64[M, N] @ DRAM, dout: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, inv_n: f64[1] @ DRAM):
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += dout[i, j] * x[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            dx[i, j] = dout[i, j] * scale - x[i, j] * corr


@proc
def _attn_qkv_fwd(q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM):
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
                mx = select(mx, logit, logit, mx)

            sum_val = 0.0
            for s in seq(0, BLOCK_SIZE):
                if s > t:
                    logit = CAUSAL_MASK_VALUE
                else:
                    logit = 0.0
                    for d in seq(0, HEAD_DIM):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * INV_SCALE
                t0 = logit - mx
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
def _attn_av_fwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out[t, h * HEAD_DIM + d] = acc


@proc
def _mlp_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
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
def _mlp_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dw1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dw2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
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
def _adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, b1: f64[1] @ DRAM, b2: f64[1] @ DRAM, eps: f64[1] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    inv_b1: f64 @ Stack
    inv_b2: f64 @ Stack
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_b1 = 1.0 - b1[0]
    inv_b2 = 1.0 - b2[0]
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in par(0, N):
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
def _attn_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM):
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
            dot: f64 @ Stack
            dot = 0.0
            for s in seq(0, BLOCK_SIZE):
                dattn_w: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * v[h, s, d]
                dot += dattn_w * attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                dattn_w: f64 @ Stack
                dlogit: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * v[h, s, d]
                dlogit = attn_w[h, t, s] * (dattn_w - dot) * INV_SCALE

                for d in seq(0, HEAD_DIM):
                    dq_contrib: f64 @ Stack
                    dk_contrib: f64 @ Stack
                    dv_contrib: f64 @ Stack
                    dq_contrib = dlogit * k[h, s, d]
                    dk_contrib = dlogit * q[h, t, d]
                    dv_contrib = attn_w[h, t, s] * dattn_out[h, t, d]

                    for e in seq(0, N_EMBED):
                        out[t, e] += dq_contrib * wq[h * HEAD_DIM + d, e]
                        out[s, e] += dk_contrib * wk[h * HEAD_DIM + d, e]
                        out[s, e] += dv_contrib * wv[h * HEAD_DIM + d, e]
                        dwq[h * HEAD_DIM + d, e] += dq_contrib * xn[t, e]
                        dwk[h * HEAD_DIM + d, e] += dk_contrib * xn[s, e]
                        dwv[h * HEAD_DIM + d, e] += dv_contrib * xn[s, e]

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
def _lm_head_bwd(V: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[V, N_EMBED] @ DRAM, dlogits: f64[BLOCK_SIZE, V] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[V, N_EMBED] @ DRAM):
    for v_idx in seq(0, V):
        for e in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for t in seq(0, BLOCK_SIZE):
                acc += dlogits[t, v_idx] * x[t, e]
            dweight[v_idx, e] = acc

    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            acc: f64 @ Stack
            acc = 0.0
            for v_idx in seq(0, V):
                acc += dlogits[t, v_idx] * lm_head[v_idx, e]
            dx[t, e] = acc


JIT_RMSNORM_FWD = jit(simplify(_rmsnorm_fwd.partial_eval(M=BLOCK_SIZE, N=N_EMBED)))
JIT_RMSNORM_BWD = jit(simplify(_rmsnorm_bwd.partial_eval(M=BLOCK_SIZE, N=N_EMBED)))
JIT_ATTN_FWD = jit(simplify(_attn_fwd_fused))
JIT_ATTN_BWD = jit(simplify(_attn_bwd_fused))
JIT_MLP_FWD = jit(simplify(_mlp_fwd_fused))
JIT_MLP_BWD = jit(simplify(_mlp_bwd_fused))


@cache
def _jit_adam(n: int):
    return jit(simplify(_adam.partial_eval(N=n)))


def rmsnorm_fwd(x: Tensor) -> tuple[Tensor, Tensor]:
    out = empty_like_array(x)
    rms = empty_array((x.shape[0], 1), dtype=float)
    JIT_RMSNORM_FWD._raw(out.ctypes.data, rms.ctypes.data, x.ctypes.data, RMS_INV_N.ctypes.data, RMS_EPS.ctypes.data)
    return out, rms


def rmsnorm_bwd(dout: Tensor, x: Tensor, rms: Tensor) -> Tensor:
    dx = empty_like_array(x)
    JIT_RMSNORM_BWD._raw(dx.ctypes.data, dout.ctypes.data, x.ctypes.data, rms.ctypes.data, RMS_INV_N.ctypes.data)
    return dx


def attn_fwd(x: Tensor, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor) -> tuple[Tensor, AttnCache]:
    xn, rms = rmsnorm_fwd(x)
    q = empty_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)
    k = empty_like_array(q)
    v = empty_like_array(q)
    attn_w = empty_array((N_HEAD, BLOCK_SIZE, BLOCK_SIZE), dtype=float)
    out_flat = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    out = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    for h in range(N_HEAD):
        for t in range(BLOCK_SIZE):
            logits = [CAUSAL_MASK_VALUE] * BLOCK_SIZE
            for s in range(t + 1):
                logit = 0.0
                for d in range(HEAD_DIM):
                    logit += q[h, t, d] * k[h, s, d]
                logits[s] = logit * INV_SCALE
            mx = max(logits)
            weights = [math.exp(v - mx) for v in logits]
            denom = sum(weights)
            for s in range(BLOCK_SIZE):
                attn_w[h, t, s] = weights[s] / denom

    for h in range(N_HEAD):
        for t in range(BLOCK_SIZE):
            for d in range(HEAD_DIM):
                acc = 0.0
                for s in range(BLOCK_SIZE):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out_flat[t, h * HEAD_DIM + d] = acc

    for t in range(BLOCK_SIZE):
        for j in range(N_EMBED):
            acc = 0.0
            for e in range(N_EMBED):
                acc += out_flat[t, e] * wo[j, e]
            out[t, j] = acc + x[t, j]
    return out, AttnCache(x, xn, rms, q, k, v, attn_w, out_flat)


def attn_bwd(dx: Tensor, grads: dict, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor, c: AttnCache, li: int) -> Tensor:
    out = zeros_array((BLOCK_SIZE, N_EMBED), dtype=float)
    dattn_out = zeros_array((N_HEAD, BLOCK_SIZE, HEAD_DIM), dtype=float)

    for row in range(N_EMBED):
        for e in range(N_EMBED):
            acc = 0.0
            for t in range(BLOCK_SIZE):
                acc += dx[t, row] * c.out_flat[t, e]
            grads[f"layer{li}.attn_wo"][row, e] = acc

    for row in range(N_EMBED):
        for e in range(N_EMBED):
            grads[f"layer{li}.attn_wq"][row, e] = 0.0
            grads[f"layer{li}.attn_wk"][row, e] = 0.0
            grads[f"layer{li}.attn_wv"][row, e] = 0.0

    for h in range(N_HEAD):
        for t in range(BLOCK_SIZE):
            for d in range(HEAD_DIM):
                acc = 0.0
                for j in range(N_EMBED):
                    acc += dx[t, j] * wo[j, h * HEAD_DIM + d]
                dattn_out[h, t, d] = acc

    for h in range(N_HEAD):
        for t in range(BLOCK_SIZE):
            dot = 0.0
            for s in range(BLOCK_SIZE):
                dattn_w = 0.0
                for d in range(HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * c.v[h, s, d]
                dot += dattn_w * c.attn_w[h, t, s]

            for s in range(BLOCK_SIZE):
                dattn_w = 0.0
                for d in range(HEAD_DIM):
                    dattn_w += dattn_out[h, t, d] * c.v[h, s, d]
                dlogit = c.attn_w[h, t, s] * (dattn_w - dot) * INV_SCALE

                for d in range(HEAD_DIM):
                    dq_contrib = dlogit * c.k[h, s, d]
                    dk_contrib = dlogit * c.q[h, t, d]
                    dv_contrib = c.attn_w[h, t, s] * dattn_out[h, t, d]

                    for e in range(N_EMBED):
                        out[t, e] += dq_contrib * wq[h * HEAD_DIM + d, e]
                        out[s, e] += dk_contrib * wk[h * HEAD_DIM + d, e]
                        out[s, e] += dv_contrib * wv[h * HEAD_DIM + d, e]
                        grads[f"layer{li}.attn_wq"][h * HEAD_DIM + d, e] += dq_contrib * c.xn[t, e]
                        grads[f"layer{li}.attn_wk"][h * HEAD_DIM + d, e] += dk_contrib * c.xn[s, e]
                        grads[f"layer{li}.attn_wv"][h * HEAD_DIM + d, e] += dv_contrib * c.xn[s, e]

    for i in range(BLOCK_SIZE):
        dot = 0.0
        scale = c.rms[i, 0]
        for j in range(N_EMBED):
            dot += out[i, j] * c.x_pre[i, j]
        corr = scale * scale * scale * (1.0 / N_EMBED) * dot
        for j in range(N_EMBED):
            out[i, j] = out[i, j] * scale - c.x_pre[i, j] * corr + dx[i, j]
    return out


def mlp_fwd(x: Tensor, fc1: Tensor, fc2: Tensor) -> tuple[Tensor, MlpCache]:
    out = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    xn = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    rms = empty_array((BLOCK_SIZE, 1), dtype=float)
    h_pre = empty_array((BLOCK_SIZE, 4 * N_EMBED), dtype=float)
    h = empty_like_array(h_pre)
    JIT_MLP_FWD._raw(out.ctypes.data, xn.ctypes.data, rms.ctypes.data, h_pre.ctypes.data, h.ctypes.data, x.ctypes.data, fc1.ctypes.data, fc2.ctypes.data, RMS_INV_N.ctypes.data, RMS_EPS.ctypes.data)
    return out, MlpCache(x, xn, rms, h_pre, h)


def mlp_bwd(dx: Tensor, grads: dict, fc1: Tensor, fc2: Tensor, c: MlpCache, li: int) -> Tensor:
    out = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    JIT_MLP_BWD._raw(out.ctypes.data, grads[f"layer{li}.mlp_fc1"].ctypes.data, grads[f"layer{li}.mlp_fc2"].ctypes.data, dx.ctypes.data, c.x_pre.ctypes.data, c.xn.ctypes.data, c.rms.ctypes.data, c.h_pre.ctypes.data, c.h.ctypes.data, fc1.ctypes.data, fc2.ctypes.data, RMS_INV_N.ctypes.data)
    return out


def forward(params: dict, input_ids: Tensor, target_ids: Tensor, loss_mask: Tensor) -> tuple[float, FwdCache]:
    emb = params["wte"][input_ids] + params["wpe"]
    x, rms_init = rmsnorm_fwd(emb)

    layer_caches = []
    for li in range(N_LAYER):
        x, ac = attn_fwd(x, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"])
        x, mc = mlp_fwd(x, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"])
        layer_caches.append((ac, mc))

    logits = empty_array((BLOCK_SIZE, params["lm_head"].shape[0]), dtype=float)
    JIT_LOGITS._raw(logits.ctypes.data, x.ctypes.data, params["lm_head"].ctypes.data)
    probs = softmax(logits)
    sum_mask = float(loss_mask.sum())
    loss = cross_entropy_loss(probs, target_ids, loss_mask, sum_mask)
    return float(loss), FwdCache(input_ids, target_ids, loss_mask, sum_mask, emb, rms_init, x, probs, layer_caches)


def backward(params: dict, grads: dict, cache: FwdCache) -> None:
    dlogits = cache.probs.copy()
    inv_sum_mask = 1.0 / cache.sum_mask
    dlogits *= inv_sum_mask
    for i in range(BLOCK_SIZE):
        dlogits[i, int(cache.target_ids[i])] -= inv_sum_mask
    dlogits *= cache.loss_mask[:, None]

    dx = empty_array((BLOCK_SIZE, N_EMBED), dtype=float)
    JIT_LM_HEAD_BWD._raw(dx.ctypes.data, grads["lm_head"].ctypes.data, dlogits.ctypes.data, cache.x.ctypes.data, params["lm_head"].ctypes.data)

    for li in reversed(range(N_LAYER)):
        ac, mc = cache.layer_caches[li]
        dx = mlp_bwd(dx, grads, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"], mc, li)
        dx = attn_bwd(dx, grads, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"], ac, li)

    demb = rmsnorm_bwd(dx, cache.emb, cache.rms_init)
    for i in range(BLOCK_SIZE):
        row = int(cache.input_ids[i])
        for j in range(N_EMBED):
            grads["wte"][row, j] += demb[i, j]
    grads["wpe"] += demb


def step_fn(params: dict, opt_state: dict, grads: dict, input_ids: Tensor, target_ids: Tensor, loss_mask: Tensor, step: int) -> tuple[float, dict, dict]:
    JIT_ZERO_GRADS._raw(opt_state["flat_grads"].ctypes.data)
    loss, cache = forward(params, input_ids, target_ids, loss_mask)
    backward(params, grads, cache)

    opt_state["lr"][0] = ADAM_PARAMS["LR_T"][step]
    opt_state["bc1"][0] = ADAM_PARAMS["BC1"][step]
    opt_state["bc2"][0] = ADAM_PARAMS["BC2"][step]
    JIT_ADAM._raw(opt_state["flat_params"].ctypes.data, opt_state["flat_grads"].ctypes.data, opt_state["flat_m"].ctypes.data, opt_state["flat_v"].ctypes.data, ADAM_B1.ctypes.data, ADAM_B2.ctypes.data, ADAM_EPS.ctypes.data, opt_state["lr"].ctypes.data, opt_state["bc1"].ctypes.data, opt_state["bc2"].ctypes.data)
    return loss, params, opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[Tensor, Tensor, Tensor]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    input_ids = zeros_array((BLOCK_SIZE,), dtype=int)
    target_ids = zeros_array((BLOCK_SIZE,), dtype=int)
    loss_mask = zeros_array((BLOCK_SIZE,), dtype=float)
    for i in range(n):
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

state_dict = {
    "wte": random_matrix(len(uchars) + 1, N_EMBED),
    "wpe": random_matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": random_matrix(len(uchars) + 1, N_EMBED),
    **{f"layer{i}.attn_wq": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": random_matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": random_matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": random_matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

total_params = sum(array_numel(state_dict[k].shape) for k in state_dict)
flat_params = empty_array((total_params,), dtype=float)
offset = 0
for k in state_dict:
    arr = state_dict[k]
    for idx in iter_indices(arr.shape):
        flat_params[offset] = arr[idx]
        offset += 1
offset = 0
for k in state_dict:
    shape = state_dict[k].shape
    state_dict[k] = flat_view(flat_params, offset, shape)
    offset += array_numel(shape)

flat_grads = zeros_array((total_params,), dtype=float)
grads = {}
offset = 0
for k in state_dict:
    shape = state_dict[k].shape
    grads[k] = flat_view(flat_grads, offset, shape)
    offset += array_numel(shape)

opt_state = {
    "flat_m": zeros_array((total_params,), dtype=float),
    "flat_v": zeros_array((total_params,), dtype=float),
    "flat_params": flat_params,
    "flat_grads": flat_grads,
    "lr": empty_array((1,), dtype=float),
    "bc1": empty_array((1,), dtype=float),
    "bc2": empty_array((1,), dtype=float),
}

tokenized = [tokenize(doc, uchars) for doc in docs]

JIT_LOGITS = _jit_matmul_nt(BLOCK_SIZE, N_EMBED, len(uchars) + 1)
JIT_LM_HEAD_BWD = jit(simplify(_lm_head_bwd.partial_eval(V=len(uchars) + 1)))
JIT_ZERO_GRADS = _jit_zero_1d(total_params)
JIT_ADAM = _jit_adam(total_params)

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, grads, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
