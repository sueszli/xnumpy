from __future__ import annotations

import ctypes
import random
import sys
import time
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TypeVar

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from exo import *
from exo.libs.externs import expf, select
from exo.stdlib.scheduling import simplify
from utils.exo_alloc import Tensor, empty, zeros
from utils.exo_kernels import adam, add, matmul, matmul_left_t, matmul_right_t, relu, rmsnorm, rmsnorm_bwd
from utils.times import save_times
from utils.weights import assert_weights_match

from exojit.main import jit
from exojit.patches_exo import Stack

N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD
INV_SCALE = 1.0 / HEAD_DIM**0.5
CAUSAL_MASK_VALUE = -1e10


@proc
def lm_head_step_fused(vocab_size: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM):
    matmul_right_t(BLOCK_SIZE, vocab_size, N_EMBED, logits, x, lm_head, zero)
    for t in seq(0, BLOCK_SIZE):
        mx: f64 @ Stack
        sum_val: f64 @ Stack
        scale: f64 @ Stack
        val: f64 @ Stack
        inv_denom: f64 @ Stack
        mx = -1e10
        for v_idx in seq(0, vocab_size):
            mx = select(mx, logits[t, v_idx], logits[t, v_idx], mx)
        sum_val = zero[0]
        for v_idx in seq(0, vocab_size):
            val = expf(logits[t, v_idx] - mx)
            logits[t, v_idx] = val
            sum_val += val
        inv_denom = one[0] / sum_val
        scale = loss_mask[t] * inv_sum_mask[0] * inv_denom
        for v_idx in seq(0, vocab_size):
            logits[t, v_idx] = logits[t, v_idx] * scale
            if v_idx == target_ids[t]:
                logits[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]
    matmul_left_t(BLOCK_SIZE, vocab_size, N_EMBED, dweight, logits, x, zero)
    matmul(BLOCK_SIZE, N_EMBED, vocab_size, dx, logits, lm_head, zero)


@proc
def embed_rms_fwd(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            emb[t, e] = wpe[t, e]
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    emb[t, e] += wte[v, e]
    rmsnorm(BLOCK_SIZE, N_EMBED, out, rms, emb, zero, one, inv_n, eps)


@proc
def embed_rms_bwd(vocab_size: size, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = zero[0]
        scale = rms[t, 0]
        for e in seq(0, N_EMBED):
            dot += dout[t, e] * x[t, e]
        corr = scale * scale * scale * inv_n[0] * dot
        for e in seq(0, N_EMBED):
            dx: f64 @ Stack
            dx = dout[t, e] * scale - x[t, e] * corr
            g_wpe[t, e] = dx
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    g_wte[v, e] += dx


@proc
def attn_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    rmsnorm(BLOCK_SIZE, N_EMBED, xn, rms, x, zero, one, inv_n, eps)

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

    matmul_right_t(BLOCK_SIZE, N_EMBED, N_EMBED, out, out_flat, wo, zero)
    add(BLOCK_SIZE, N_EMBED, out, x)


@proc
def mlp_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    rmsnorm(BLOCK_SIZE, N_EMBED, xn, rms, x, zero, one, inv_n, eps)
    matmul_right_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, h_pre, xn, fc1, zero)
    relu(BLOCK_SIZE, 4 * N_EMBED, h, h_pre, zero)
    matmul_right_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, h, fc2, zero)
    add(BLOCK_SIZE, N_EMBED, out, x)


@proc
def mlp_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dw1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dw2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            out[t, e] = zero[0]
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dw2, dx, h, zero)
    for e in seq(0, 4 * N_EMBED):
        for k in seq(0, N_EMBED):
            dw1[e, k] = zero[0]

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

    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, out, dx, x_pre, rms, zero, inv_n)


@proc
def attn_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            out[t, e] = zero[0]
    matmul_left_t(BLOCK_SIZE, N_EMBED, N_EMBED, dwo, dx, out_flat, zero)
    for i in seq(0, N_EMBED):
        for j in seq(0, N_EMBED):
            dwq[i, j] = zero[0]
            dwk[i, j] = zero[0]
            dwv[i, j] = zero[0]

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

    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, out, dx, x_pre, rms, zero, inv_n)


@dataclass(frozen=True)
class TokenBatch:
    input_ids: Tensor
    target_ids: Tensor
    loss_mask: Tensor
    inv_sum_mask: Tensor


@dataclass(frozen=True)
class Params:
    wte: Tensor
    wpe: Tensor
    lm_head: Tensor
    attn_wq: Tensor
    attn_wk: Tensor
    attn_wv: Tensor
    attn_wo: Tensor
    mlp_fc1: Tensor
    mlp_fc2: Tensor


@dataclass(frozen=True)
class Scratch:
    emb: Tensor
    rms_init: Tensor
    x0: Tensor
    x1: Tensor
    logits: Tensor
    attn_xn: Tensor
    attn_rms: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    attn_w: Tensor
    out_flat: Tensor
    mlp_xn: Tensor
    mlp_rms: Tensor
    h_pre: Tensor
    h: Tensor
    dx0: Tensor
    dx1: Tensor
    dattn_out: Tensor


@dataclass(frozen=True)
class Scalars:
    opt_lr: Tensor
    opt_bc1: Tensor
    opt_bc2: Tensor
    zero: Tensor
    one: Tensor
    rms_inv_n: Tensor
    rms_eps: Tensor
    adam_b1: Tensor
    adam_b2: Tensor
    adam_eps: Tensor


@dataclass(frozen=True)
class OptState:
    m: Tensor
    v: Tensor


LayoutT = TypeVar("LayoutT")


def init_normal_(tensor: Tensor, *, scale: float) -> None:
    flat = tensor.view((tensor.numel,))
    for i in range(tensor.numel):
        flat[i] = random.gauss(0.0, scale)


SCALAR_LAYOUT = ((1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,))


def layout_numel(layout: tuple[tuple[int, ...], ...]) -> int:
    return sum(prod(shape) for shape in layout)


def bind_layout(cls: type[LayoutT], flat: Tensor, layout: tuple[tuple[int, ...], ...]) -> LayoutT:
    offset = 0
    views = []
    for shape in layout:
        views.append(flat.view(shape, offset=offset))
        offset += prod(shape)
    return cls(*views)


def param_layout(vocab_size: int) -> tuple[tuple[int, ...], ...]:
    return (
        (vocab_size, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (vocab_size, N_EMBED),
        (N_EMBED, N_EMBED),
        (N_EMBED, N_EMBED),
        (N_EMBED, N_EMBED),
        (N_EMBED, N_EMBED),
        (4 * N_EMBED, N_EMBED),
        (N_EMBED, 4 * N_EMBED),
    )


def scratch_layout(vocab_size: int) -> tuple[tuple[int, ...], ...]:
    return (
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, 1),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, vocab_size),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, 1),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, BLOCK_SIZE),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, 1),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
    )


def named_params(params: Params) -> tuple[tuple[str, Tensor], ...]:
    return (
        ("wte", params.wte),
        ("wpe", params.wpe),
        ("lm_head", params.lm_head),
        ("layer0.attn_wq", params.attn_wq),
        ("layer0.attn_wk", params.attn_wk),
        ("layer0.attn_wv", params.attn_wv),
        ("layer0.attn_wo", params.attn_wo),
        ("layer0.mlp_fc1", params.mlp_fc1),
        ("layer0.mlp_fc2", params.mlp_fc2),
    )


def tokenize(doc: str, c2i: dict[str, int], bos: int) -> TokenBatch:
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    inputs = zeros((BLOCK_SIZE,), dtype=int)
    targets = zeros((BLOCK_SIZE,), dtype=int)
    loss_mask = zeros((BLOCK_SIZE,))
    for i in range(n):
        inputs[i] = tokens[i]
        targets[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    inv_sum_mask = empty((1,))
    inv_sum_mask[0] = 1.0 / max(1, n)
    return TokenBatch(inputs, targets, loss_mask, inv_sum_mask)


def wrap_state_dict(params: Params) -> dict[str, list[list[object]]]:
    class W:
        __slots__ = ("data",)

        def __init__(self, data: float):
            self.data = data

    return {name: [[W(float(tensor[i, j])) for j in range(tensor.shape[1])] for i in range(tensor.shape[0])] for name, tensor in named_params(params)}


def main() -> None:
    random.seed(42)
    num_steps = 1000
    attn_fwd, attn_bwd, mlp_fwd, mlp_bwd = (jit(simplify(proc))._raw for proc in (attn_fwd_fused, attn_bwd_fused, mlp_fwd_fused, mlp_bwd_fused))

    docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
    random.shuffle(docs)
    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1

    params_layout = param_layout(vocab_size)
    flat_params = empty((layout_numel(params_layout),))
    params = bind_layout(Params, flat_params, params_layout)
    for _, tensor in named_params(params):
        init_normal_(tensor, scale=0.08)

    flat_grads = zeros((flat_params.numel,))
    grads = bind_layout(Params, flat_grads, params_layout)
    opt_state = OptState(m=zeros((flat_params.numel,)), v=zeros((flat_params.numel,)))

    scratch = bind_layout(Scratch, empty((layout_numel(scratch_layout(vocab_size)),)), scratch_layout(vocab_size))
    scalars = bind_layout(Scalars, empty((layout_numel(SCALAR_LAYOUT),)), SCALAR_LAYOUT)
    scalars.zero[0] = 0.0
    scalars.one[0] = 1.0
    scalars.rms_inv_n[0] = 1.0 / N_EMBED
    scalars.rms_eps[0] = 1e-5
    scalars.adam_b1[0] = 0.85
    scalars.adam_b2[0] = 0.99
    scalars.adam_eps[0] = 1e-8

    lm_head_step = jit(lm_head_step_fused, raw=True)
    embed_rms_fwd_step = jit(embed_rms_fwd, raw=True)
    embed_rms_bwd_step = jit(embed_rms_bwd, raw=True)
    adam_step = jit(simplify(adam.partial_eval(N=flat_params.numel)))._raw

    attn_fwd_args = (
        scratch.x1.ptr,
        scratch.attn_xn.ptr,
        scratch.attn_rms.ptr,
        scratch.q.ptr,
        scratch.k.ptr,
        scratch.v.ptr,
        scratch.attn_w.ptr,
        scratch.out_flat.ptr,
        scratch.x0.ptr,
        params.attn_wq.ptr,
        params.attn_wk.ptr,
        params.attn_wv.ptr,
        params.attn_wo.ptr,
        scalars.zero.ptr,
        scalars.one.ptr,
        scalars.rms_inv_n.ptr,
        scalars.rms_eps.ptr,
    )
    mlp_fwd_args = (
        scratch.dx0.ptr,
        scratch.mlp_xn.ptr,
        scratch.mlp_rms.ptr,
        scratch.h_pre.ptr,
        scratch.h.ptr,
        scratch.x1.ptr,
        params.mlp_fc1.ptr,
        params.mlp_fc2.ptr,
        scalars.zero.ptr,
        scalars.one.ptr,
        scalars.rms_inv_n.ptr,
        scalars.rms_eps.ptr,
    )
    mlp_bwd_args = (
        scratch.dx0.ptr,
        grads.mlp_fc1.ptr,
        grads.mlp_fc2.ptr,
        scratch.dx1.ptr,
        scratch.x1.ptr,
        scratch.mlp_xn.ptr,
        scratch.mlp_rms.ptr,
        scratch.h_pre.ptr,
        scratch.h.ptr,
        params.mlp_fc1.ptr,
        params.mlp_fc2.ptr,
        scalars.zero.ptr,
        scalars.rms_inv_n.ptr,
    )
    attn_bwd_args = (
        scratch.dx1.ptr,
        grads.attn_wq.ptr,
        grads.attn_wk.ptr,
        grads.attn_wv.ptr,
        grads.attn_wo.ptr,
        scratch.dattn_out.ptr,
        scratch.dx0.ptr,
        scratch.x0.ptr,
        scratch.attn_xn.ptr,
        scratch.attn_rms.ptr,
        scratch.q.ptr,
        scratch.k.ptr,
        scratch.v.ptr,
        scratch.attn_w.ptr,
        scratch.out_flat.ptr,
        params.attn_wq.ptr,
        params.attn_wk.ptr,
        params.attn_wv.ptr,
        params.attn_wo.ptr,
        scalars.zero.ptr,
        scalars.rms_inv_n.ptr,
    )
    adam_args = (
        flat_params.ptr,
        flat_grads.ptr,
        opt_state.m.ptr,
        opt_state.v.ptr,
        scalars.adam_b1.ptr,
        scalars.adam_b2.ptr,
        scalars.adam_eps.ptr,
        scalars.opt_lr.ptr,
        scalars.opt_bc1.ptr,
        scalars.opt_bc2.ptr,
    )

    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = vocab_size - 1
    tokenized = [tokenize(doc, c2i, bos) for doc in docs]

    g_wte_bytes = grads.wte.numel * grads.wte.itemsize
    g_wpe_bytes = grads.wpe.numel * grads.wpe.itemsize
    lr_t = [0.01 * (1.0 - step / num_steps) for step in range(num_steps)]
    bc1 = [1.0 - 0.85 ** (step + 1) for step in range(num_steps)]
    bc2 = [1.0 - 0.99 ** (step + 1) for step in range(num_steps)]
    memset = ctypes.memset
    perf_counter = time.perf_counter
    step_times = []

    for step in range(num_steps):
        scalars.opt_lr[0] = lr_t[step]
        scalars.opt_bc1[0] = bc1[step]
        scalars.opt_bc2[0] = bc2[step]
        batch = tokenized[step % len(tokenized)]
        embed_args = (
            vocab_size,
            scratch.emb.ptr,
            scratch.x0.ptr,
            scratch.rms_init.ptr,
            params.wte.ptr,
            params.wpe.ptr,
            scalars.zero.ptr,
            scalars.one.ptr,
            scalars.rms_inv_n.ptr,
            scalars.rms_eps.ptr,
            batch.input_ids.ptr,
        )
        lm_head_args = (
            vocab_size,
            scratch.dx1.ptr,
            grads.lm_head.ptr,
            scratch.logits.ptr,
            scratch.dx0.ptr,
            params.lm_head.ptr,
            batch.loss_mask.ptr,
            batch.inv_sum_mask.ptr,
            scalars.zero.ptr,
            scalars.one.ptr,
            batch.target_ids.ptr,
        )
        embed_bwd_args = (
            vocab_size,
            grads.wte.ptr,
            grads.wpe.ptr,
            scratch.dx1.ptr,
            scratch.emb.ptr,
            scratch.rms_init.ptr,
            scalars.zero.ptr,
            scalars.rms_inv_n.ptr,
            batch.input_ids.ptr,
        )
        memset(grads.wte.ptr, 0, g_wte_bytes)
        memset(grads.wpe.ptr, 0, g_wpe_bytes)
        t0 = perf_counter()
        embed_rms_fwd_step(*embed_args)
        attn_fwd(*attn_fwd_args)
        mlp_fwd(*mlp_fwd_args)
        lm_head_step(*lm_head_args)
        mlp_bwd(*mlp_bwd_args)
        attn_bwd(*attn_bwd_args)
        embed_rms_bwd_step(*embed_bwd_args)
        adam_step(*adam_args)
        step_times.append(perf_counter() - t0)

    save_times(step_times)
    assert_weights_match(wrap_state_dict(params))


if __name__ == "__main__":
    main()
