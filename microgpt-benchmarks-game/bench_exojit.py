from __future__ import annotations

import ctypes
import random
import sys
import time
from collections import namedtuple
from dataclasses import dataclass
from math import prod
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

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
NUM_STEPS = 1000

HEAD_DIM = N_EMBED // N_HEAD
CAUSAL_MASK_VALUE = -1e10
ATTN_SCALE = 1.0 / HEAD_DIM**0.5
INV_N = 1.0 / N_EMBED


#
# kernels
#


@proc
def add(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        for j in seq(0, N_EMBED):
            out[i, j] += x[i, j]


@proc
def matmul(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[K, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += x[i, k] * w[k, j]
            out[i, j] = acc


@proc
def matmul_right_t(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[N, K] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += x[i, k] * w[j, k]
            out[i, j] = acc


@proc
def matmul_left_t(M: size, N: size, K: size, out: f64[N, K] @ DRAM, x: f64[M, N] @ DRAM, w: f64[M, K] @ DRAM):
    for j in seq(0, N):
        for k in seq(0, K):
            acc: f64 @ Stack
            acc = 0.0
            for i in seq(0, M):
                acc += x[i, j] * w[i, k]
            out[j, k] = acc


@proc
def softmax(M: size, N: size, x: f64[M, N] @ DRAM):
    for i in seq(0, M):
        mx: f64 @ Stack
        sum_val: f64 @ Stack
        mx = CAUSAL_MASK_VALUE
        for j in seq(0, N):
            mx = select(mx, x[i, j], x[i, j], mx)
        sum_val = 0.0
        for j in seq(0, N):
            x[i, j] = expf(x[i, j] - mx)
            sum_val += x[i, j]
        for j in seq(0, N):
            x[i, j] = x[i, j] / sum_val


@proc
def relu(out: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        for j in seq(0, 4 * N_EMBED):
            out[i, j] = select(0.0, x[i, j], x[i, j], 0.0)


@proc
def relu_bwd(out: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        for j in seq(0, 4 * N_EMBED):
            out[i, j] = select(0.0, x_pre[i, j], dout[i, j], 0.0)


@proc
def rmsnorm(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N_EMBED):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * INV_N + 1e-5)
        rms[i, 0] = scale
        for j in seq(0, N_EMBED):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N_EMBED):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * INV_N * dot
        for j in seq(0, N_EMBED):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def cross_entropy_bwd(M: size, N: size, probs: f64[M, N] @ DRAM, target_ids: size[M] @ DRAM, loss_mask: f64[M] @ DRAM, inv_sum_mask: f64[1] @ DRAM):
    for t in seq(0, M):
        scale: f64 @ Stack
        scale = loss_mask[t] * inv_sum_mask[0]
        for v_idx in seq(0, N):
            probs[t, v_idx] = probs[t, v_idx] * scale
            if v_idx == target_ids[t]:
                probs[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]


@proc
def attention(out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM):
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for s in seq(0, BLOCK_SIZE):
                logit: f64 @ Stack
                if s > t:
                    logit = CAUSAL_MASK_VALUE
                else:
                    logit = 0.0
                    for d in seq(0, HEAD_DIM):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * ATTN_SCALE
                attn_w[h, t, s] = logit

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            mx: f64 @ Stack
            sum_val: f64 @ Stack
            mx = CAUSAL_MASK_VALUE
            for s in seq(0, BLOCK_SIZE):
                mx = select(mx, attn_w[h, t, s], attn_w[h, t, s], mx)
            sum_val = 0.0
            for s in seq(0, BLOCK_SIZE):
                attn_w[h, t, s] = expf(attn_w[h, t, s] - mx)
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
                out[h, t, d] = acc


@proc
def attention_bwd(dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dout: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                dv[h, t, d] = 0.0
                dq[h, t, d] = 0.0
        for s in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                dk[h, s, d] = 0.0

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            dot: f64 @ Stack
            dot = 0.0
            for s in seq(0, BLOCK_SIZE):
                dattn_w: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, HEAD_DIM):
                    dattn_w += dout[h, t, d] * v[h, s, d]
                    dv[h, s, d] += attn_w[h, t, s] * dout[h, t, d]
                attn_tmp[s] = dattn_w
                dot += dattn_w * attn_w[h, t, s]

            for s in seq(0, BLOCK_SIZE):
                dlogit: f64 @ Stack
                if s > t:
                    dlogit = 0.0
                else:
                    dlogit = attn_w[h, t, s] * (attn_tmp[s] - dot) * ATTN_SCALE
                for d in seq(0, HEAD_DIM):
                    dq[h, t, d] += dlogit * k[h, s, d]
                    dk[h, s, d] += dlogit * q[h, t, d]


@proc
def adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in seq(0, N):
        g: f64 @ Stack
        m_val: f64 @ Stack
        v_val: f64 @ Stack
        m_hat: f64 @ Stack
        v_hat: f64 @ Stack
        g = grad[i]
        m_val = 0.9 * m[i] + 0.1 * g
        v_val = 0.999 * v[i] + 0.001 * g * g
        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t
        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + 1e-8)
        m[i] = m_val
        v[i] = v_val


#
# layers
#


@proc
def embed_layer(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            emb[t, e] = wpe[t, e]
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    emb[t, e] += wte[v, e]
    rmsnorm(out, rms, emb)


@proc
def embed_layer_bwd(vocab_size: size, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[t, 0]
        for e in seq(0, N_EMBED):
            dot += dout[t, e] * x[t, e]
        corr = scale * scale * scale * INV_N * dot
        for e in seq(0, N_EMBED):
            dx: f64 @ Stack
            dx = dout[t, e] * scale - x[t, e] * corr
            g_wpe[t, e] = dx
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    g_wte[v, e] += dx


@proc
def attention_layer(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, attn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM):
    rmsnorm(xn, rms, x)

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

    attention(attn_out, q, k, v, attn_w)

    for t in seq(0, BLOCK_SIZE):
        for h in seq(0, N_HEAD):
            for d in seq(0, HEAD_DIM):
                out_flat[t, h * HEAD_DIM + d] = attn_out[h, t, d]

    matmul_right_t(BLOCK_SIZE, N_EMBED, N_EMBED, out, out_flat, wo)
    add(out, x)


@proc
def attention_layer_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            out[t, e] = 0.0
    matmul_left_t(BLOCK_SIZE, N_EMBED, N_EMBED, dwo, dx, out_flat)
    for i in seq(0, N_EMBED):
        for j in seq(0, N_EMBED):
            dwq[i, j] = 0.0
            dwk[i, j] = 0.0
            dwv[i, j] = 0.0

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for j in seq(0, N_EMBED):
                    acc += dx[t, j] * wo[j, h * HEAD_DIM + d]
                dattn_out[h, t, d] = acc

    attention_bwd(dq, dk, dv, dattn_out, q, k, v, attn_w)

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                for e in seq(0, N_EMBED):
                    out[t, e] += dq[h, t, d] * wq[h * HEAD_DIM + d, e]
                    dwq[h * HEAD_DIM + d, e] += dq[h, t, d] * xn[t, e]

    for h in seq(0, N_HEAD):
        for s in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                for e in seq(0, N_EMBED):
                    out[s, e] += dk[h, s, d] * wk[h * HEAD_DIM + d, e]
                    dwk[h * HEAD_DIM + d, e] += dk[h, s, d] * xn[s, e]
                    out[s, e] += dv[h, s, d] * wv[h * HEAD_DIM + d, e]
                    dwv[h * HEAD_DIM + d, e] += dv[h, s, d] * xn[s, e]

    rmsnorm_bwd(out, dx, x_pre, rms)


@proc
def mlp_layer(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM):
    rmsnorm(xn, rms, x)
    matmul_right_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, h_pre, xn, fc1)
    relu(h, h_pre)
    matmul_right_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, h, fc2)
    add(out, x)


@proc
def mlp_layer_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dfc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dfc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, dh: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dh_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM):
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dfc2, dx, h)
    matmul(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, dh, dx, fc2)
    relu_bwd(dh_pre, dh, h_pre)
    matmul_left_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, dfc1, dh_pre, xn)
    matmul(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, dh_pre, fc1)
    rmsnorm_bwd(out, dx, x_pre, rms)


#
# train loop
#


@jit(optimize=simplify, raw=True)
def train_step(vocab_size: size, total_params: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms_init: f64[BLOCK_SIZE, 1] @ DRAM, x1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, attn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dx0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dx1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dh: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dh_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, g_lm_head: f64[vocab_size, N_EMBED] @ DRAM, g_attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, g_mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, g_mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, flat_params: f64[total_params] @ DRAM, flat_grads: f64[total_params] @ DRAM, opt_m: f64[total_params] @ DRAM, opt_v: f64[total_params] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    embed_layer(vocab_size, emb, x0, rms_init, wte, wpe, input_ids)
    attention_layer(x1, attn_xn, attn_rms, q, k, v, attn_w, attn_out, out_flat, x0, attn_wq, attn_wk, attn_wv, attn_wo)
    mlp_layer(dx0, mlp_xn, mlp_rms, h_pre, h, x1, mlp_fc1, mlp_fc2)
    matmul_right_t(BLOCK_SIZE, vocab_size, N_EMBED, logits, dx0, lm_head)
    softmax(BLOCK_SIZE, vocab_size, logits)
    cross_entropy_bwd(BLOCK_SIZE, vocab_size, logits, target_ids, loss_mask, inv_sum_mask)
    matmul_left_t(BLOCK_SIZE, vocab_size, N_EMBED, g_lm_head, logits, dx0)
    matmul(BLOCK_SIZE, N_EMBED, vocab_size, dx1, logits, lm_head)
    mlp_layer_bwd(dx0, g_mlp_fc1, g_mlp_fc2, dh, dh_pre, dx1, x1, mlp_xn, mlp_rms, h_pre, h, mlp_fc1, mlp_fc2)
    attention_layer_bwd(dx1, g_attn_wq, g_attn_wk, g_attn_wv, g_attn_wo, dattn_out, dx0, x0, attn_xn, attn_rms, q, k, v, attn_w, out_flat, dq, dk, dv, attn_wq, attn_wk, attn_wv, attn_wo)
    embed_layer_bwd(vocab_size, g_wte, g_wpe, dx1, emb, rms_init, input_ids)
    adam(total_params, flat_params, flat_grads, opt_m, opt_v, lr, beta1_t, beta2_t)


PARAMS_FIELDS = "wte wpe lm_head attn_wq attn_wk attn_wv attn_wo mlp_fc1 mlp_fc2".split()
SCRATCH_FIELDS = "emb rms_init x0 x1 logits attn_xn attn_rms q k v attn_w attn_out out_flat mlp_xn mlp_rms h_pre h dh dh_pre dx0 dx1 dattn_out dq dk dv".split()
SCALARS_FIELDS = "opt_lr opt_bc1 opt_bc2".split()


def bind(fields: list[str], flat: Buf, layout: tuple[tuple[int, ...], ...]) -> dict[str, Buf]:
    off = 0
    d: dict[str, Buf] = {}
    for name, shape in zip(fields, layout):
        d[name] = flat.view(prod(shape), off)
        off += prod(shape)
    return d


def named_params(params: dict[str, Buf], layout: tuple[tuple[int, int], ...]) -> list[tuple[str, Buf, int]]:
    names = ("wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2")
    return [(n, params[PARAMS_FIELDS[i]], layout[i][1]) for i, n in enumerate(names)]


@dataclass(slots=True)
class Buf:
    ptr: int
    _a: object
    _o: int
    n: int

    def __init__(self, n, dtype=float, _a=None, _o=0):
        ct = ctypes.c_double if dtype is float else ctypes.c_int64
        self._a = (ct * n)() if _a is None else _a
        self._o = _o
        self.n = n
        self.ptr = ctypes.addressof(self._a) + _o * 8

    def view(self, n, off=0):
        return Buf(n, float, self._a, self._o + off)

    def __getitem__(self, i):
        return self._a[self._o + i]

    def __setitem__(self, i, v):
        self._a[self._o + i] = v


def tokenize(docs: list[str], uchars: list[str]) -> list[dict[str, Buf]]:
    bos = len(uchars)
    result = []
    for doc in docs:
        tokens = [bos] + [{ch: i for i, ch in enumerate(uchars)}[ch] for ch in doc] + [bos]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        inputs = Buf(BLOCK_SIZE, int)
        targets = Buf(BLOCK_SIZE, int)
        loss_mask = Buf(BLOCK_SIZE)
        for i in range(n):
            inputs[i] = tokens[i]
            targets[i] = tokens[i + 1]
            loss_mask[i] = 1.0
        inv_sum_mask = Buf(1)
        inv_sum_mask[0] = 1.0 / max(1, n)
        result.append({"input_ids": inputs, "target_ids": targets, "loss_mask": loss_mask, "inv_sum_mask": inv_sum_mask})
    return result


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1
tokenized = tokenize(docs, uchars)


if __name__ == "__main__":
    params_layout = [(vocab_size, N_EMBED), (BLOCK_SIZE, N_EMBED), (vocab_size, N_EMBED)] + [(N_EMBED, N_EMBED)] * 4 + [(4 * N_EMBED, N_EMBED), (N_EMBED, 4 * N_EMBED)]
    flat_params = Buf(sum(prod(s) for s in params_layout))
    params = bind(PARAMS_FIELDS, flat_params, params_layout)
    for _, buf, _ in named_params(params, params_layout):
        for i in range(buf.n):
            buf[i] = random.gauss(0.0, 0.08)

    flat_grads = Buf(flat_params.n)
    grads = bind(PARAMS_FIELDS, flat_grads, params_layout)
    opt_m, opt_v = Buf(flat_params.n), Buf(flat_params.n)

    sl = (
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
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, 1),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, 4 * N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (BLOCK_SIZE, N_EMBED),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
        (N_HEAD, BLOCK_SIZE, HEAD_DIM),
    )
    scratch = bind(SCRATCH_FIELDS, Buf(sum(prod(s) for s in sl)), sl)
    scalars = bind(SCALARS_FIELDS, Buf(3), ((1,) for _ in SCALARS_FIELDS))

    args = {
        "vocab_size": vocab_size,
        "total_params": flat_params.n,
        **{f: scratch[f].ptr for f in SCRATCH_FIELDS},
        **{f: params[f].ptr for f in PARAMS_FIELDS},
        **{"g_" + f: grads[f].ptr for f in PARAMS_FIELDS},
        "flat_params": flat_params.ptr,
        "flat_grads": flat_grads.ptr,
        "opt_m": opt_m.ptr,
        "opt_v": opt_v.ptr,
        "lr": scalars["opt_lr"].ptr,
        "beta1_t": scalars["opt_bc1"].ptr,
        "beta2_t": scalars["opt_bc2"].ptr,
    }

    grads_to_clear = [(grads["wte"].ptr, grads["wte"].n * 8), (grads["wpe"].ptr, grads["wpe"].n * 8)]
    lr_t = [0.01 * (1.0 - s / NUM_STEPS) for s in range(NUM_STEPS)]
    bc1 = [1.0 - 0.9 ** (s + 1) for s in range(NUM_STEPS)]
    bc2 = [1.0 - 0.999 ** (s + 1) for s in range(NUM_STEPS)]

    memset, perf_counter = ctypes.memset, time.perf_counter
    step_times = []
    for step, (lr, b1, b2) in enumerate(zip(lr_t, bc1, bc2)):
        scalars["opt_lr"][0] = lr
        scalars["opt_bc1"][0] = b1
        scalars["opt_bc2"][0] = b2
        batch = tokenized[step % len(tokenized)]
        for ptr, n in grads_to_clear:
            memset(ptr, 0, n)
        t0 = perf_counter()
        args.update({k: batch[k].ptr for k in ("loss_mask", "inv_sum_mask", "input_ids", "target_ids")})
        train_step(**args)
        step_times.append(perf_counter() - t0)

    save_times(step_times)
    W = namedtuple("W", ["data"])
    assert_weights_match({name: [[W(float(buf[i * cols + j])) for j in range(cols)] for i in range(buf.n // cols)] for name, buf, cols in named_params(params, params_layout)})
