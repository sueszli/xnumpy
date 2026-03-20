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
def rmsnorm(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
    for i in seq(0, BLOCK_SIZE):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N_EMBED):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + 1e-5)
        rms[i, 0] = scale
        for j in seq(0, N_EMBED):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, inv_n: f64[1] @ DRAM):
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
def cross_entropy_bwd(M: size, N: size, probs: f64[M, N] @ DRAM, target_ids: size[M] @ DRAM, loss_mask: f64[M] @ DRAM, inv_sum_mask: f64[1] @ DRAM):
    for t in seq(0, M):
        scale: f64 @ Stack
        scale = loss_mask[t] * inv_sum_mask[0]
        for v_idx in seq(0, N):
            probs[t, v_idx] = probs[t, v_idx] * scale
            if v_idx == target_ids[t]:
                probs[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]


@proc
def attention(out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, scale: f64[1] @ DRAM):
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
                    logit = logit * scale[0]
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
def attention_bwd(dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dout: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, scale: f64[1] @ DRAM):
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
                    dlogit = attn_w[h, t, s] * (attn_tmp[s] - dot) * scale[0]
                for d in seq(0, HEAD_DIM):
                    dq[h, t, d] += dlogit * k[h, s, d]
                    dk[h, s, d] += dlogit * q[h, t, d]


@proc
def adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM, beta1: f64[1] @ DRAM, beta2: f64[1] @ DRAM):
    inv_b1: f64 @ Stack
    inv_b2: f64 @ Stack
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_b1 = 1.0 - beta1[0]
    inv_b2 = 1.0 - beta2[0]
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in seq(0, N):
        g: f64 @ Stack
        m_val: f64 @ Stack
        v_val: f64 @ Stack
        m_hat: f64 @ Stack
        v_hat: f64 @ Stack
        g = grad[i]
        m_val = beta1[0] * m[i] + inv_b1 * g
        v_val = beta2[0] * v[i] + inv_b2 * g * g
        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t
        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + 1e-8)
        m[i] = m_val
        v[i] = v_val


#
# layers
#


@proc
def embed_layer(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            emb[t, e] = wpe[t, e]
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    emb[t, e] += wte[v, e]
    rmsnorm(out, rms, emb, inv_n)


@proc
def embed_layer_bwd(vocab_size: size, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
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
def attention_layer(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, attn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, attn_scale: f64[1] @ DRAM):
    rmsnorm(xn, rms, x, inv_n)

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

    attention(attn_out, q, k, v, attn_w, attn_scale)

    for t in seq(0, BLOCK_SIZE):
        for h in seq(0, N_HEAD):
            for d in seq(0, HEAD_DIM):
                out_flat[t, h * HEAD_DIM + d] = attn_out[h, t, d]

    matmul_right_t(BLOCK_SIZE, N_EMBED, N_EMBED, out, out_flat, wo)
    add(out, x)


@proc
def attention_layer_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, attn_scale: f64[1] @ DRAM):
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

    attention_bwd(dq, dk, dv, dattn_out, q, k, v, attn_w, attn_scale)

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

    rmsnorm_bwd(out, dx, x_pre, rms, inv_n)


@proc
def mlp_layer(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
    rmsnorm(xn, rms, x, inv_n)
    matmul_right_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, h_pre, xn, fc1)
    relu(h, h_pre)
    matmul_right_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, h, fc2)
    add(out, x)


@proc
def mlp_layer_bwd(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dfc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dfc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, dh: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dh_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dfc2, dx, h)
    matmul(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, dh, dx, fc2)
    relu_bwd(dh_pre, dh, h_pre)
    matmul_left_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, dfc1, dh_pre, xn)
    matmul(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, out, dh_pre, fc1)
    rmsnorm_bwd(out, dx, x_pre, rms, inv_n)


@proc
def lm_head_layer(vocab_size: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM):
    matmul_right_t(BLOCK_SIZE, vocab_size, N_EMBED, logits, x, lm_head)
    softmax(BLOCK_SIZE, vocab_size, logits)
    cross_entropy_bwd(BLOCK_SIZE, vocab_size, logits, target_ids, loss_mask, inv_sum_mask)
    matmul_left_t(BLOCK_SIZE, vocab_size, N_EMBED, dweight, logits, x)
    matmul(BLOCK_SIZE, N_EMBED, vocab_size, dx, logits, lm_head)


#
# train loop
#


@jit(optimize=simplify, raw=True)
def fwd_step(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms_init: f64[BLOCK_SIZE, 1] @ DRAM, x1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, attn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dx0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, attn_scale: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    embed_layer(vocab_size, emb, x0, rms_init, wte, wpe, inv_n, input_ids)
    attention_layer(x1, attn_xn, attn_rms, q, k, v, attn_w, attn_out, out_flat, x0, attn_wq, attn_wk, attn_wv, attn_wo, inv_n, attn_scale)
    mlp_layer(dx0, mlp_xn, mlp_rms, h_pre, h, x1, mlp_fc1, mlp_fc2, inv_n)


@jit(optimize=simplify, raw=True)
def bwd_step(vocab_size: size, dx1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, g_lm_head: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, dx0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM, g_mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, g_mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dh: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dh_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, g_attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms_init: f64[BLOCK_SIZE, 1] @ DRAM, x0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dq: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dk: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dv: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, x1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, attn_scale: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    lm_head_layer(vocab_size, dx1, g_lm_head, logits, dx0, lm_head, loss_mask, inv_sum_mask, target_ids)
    mlp_layer_bwd(dx0, g_mlp_fc1, g_mlp_fc2, dh, dh_pre, dx1, x1, mlp_xn, mlp_rms, h_pre, h, mlp_fc1, mlp_fc2, inv_n)
    attention_layer_bwd(dx1, g_attn_wq, g_attn_wk, g_attn_wv, g_attn_wo, dattn_out, dx0, x0, attn_xn, attn_rms, q, k, v, attn_w, out_flat, dq, dk, dv, attn_wq, attn_wk, attn_wv, attn_wo, inv_n, attn_scale)
    embed_layer_bwd(vocab_size, g_wte, g_wpe, dx1, emb, rms_init, inv_n, input_ids)


PARAMS_FIELDS = "wte wpe lm_head attn_wq attn_wk attn_wv attn_wo mlp_fc1 mlp_fc2".split()
SCRATCH_FIELDS = "emb rms_init x0 x1 logits attn_xn attn_rms q k v attn_w attn_out out_flat mlp_xn mlp_rms h_pre h dh dh_pre dx0 dx1 dattn_out dq dk dv".split()
SCALARS_FIELDS = "opt_lr opt_bc1 opt_bc2 rms_inv_n opt_beta1 opt_beta2 attn_scale".split()


def bind(fields, flat, layout):
    off = 0
    d = {}
    for name, shape in zip(fields, layout):
        d[name] = flat.view(prod(shape), off)
        off += prod(shape)
    return d


def scratch_layout(vocab_size):
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


def named_params(params, layout):
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


def tokenize(doc, c2i, bos):
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
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
    return {"input_ids": inputs, "target_ids": targets, "loss_mask": loss_mask, "inv_sum_mask": inv_sum_mask}


if __name__ == "__main__":
    docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
    random.shuffle(docs)
    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1

    params_layout = (
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
    flat_params = Buf(sum(prod(s) for s in params_layout))
    params = bind(PARAMS_FIELDS, flat_params, params_layout)
    for name, buf, _ in named_params(params, params_layout):
        for i in range(buf.n):
            buf[i] = random.gauss(0.0, 0.08)

    flat_grads = Buf(flat_params.n)
    grads = bind(PARAMS_FIELDS, flat_grads, params_layout)
    opt_state = {"m": Buf(flat_params.n), "v": Buf(flat_params.n)}

    scratch = bind(SCRATCH_FIELDS, Buf(sum(prod(s) for s in scratch_layout(vocab_size))), scratch_layout(vocab_size))
    scalars = bind(SCALARS_FIELDS, Buf(7), ((1,), (1,), (1,), (1,), (1,), (1,), (1,)))
    scalars["rms_inv_n"][0] = 1.0 / N_EMBED
    scalars["opt_beta1"][0] = 0.9
    scalars["opt_beta2"][0] = 0.999
    scalars["attn_scale"][0] = 1.0 / HEAD_DIM**0.5

    adam_step = jit(simplify(adam.partial_eval(N=flat_params.n)))._raw

    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = vocab_size - 1
    tokenized = [tokenize(doc, c2i, bos) for doc in docs]

    g_wte_bytes = grads["wte"].n * 8
    g_wpe_bytes = grads["wpe"].n * 8
    lr_t = [0.01 * (1.0 - step / NUM_STEPS) for step in range(NUM_STEPS)]
    bc1 = [1.0 - 0.9 ** (step + 1) for step in range(NUM_STEPS)]
    bc2 = [1.0 - 0.999 ** (step + 1) for step in range(NUM_STEPS)]
    memset = ctypes.memset
    perf_counter = time.perf_counter
    step_times = []

    for step in range(NUM_STEPS):
        scalars["opt_lr"][0] = lr_t[step]
        scalars["opt_bc1"][0] = bc1[step]
        scalars["opt_bc2"][0] = bc2[step]
        batch = tokenized[step % len(tokenized)]
        memset(grads["wte"].ptr, 0, g_wte_bytes)
        memset(grads["wpe"].ptr, 0, g_wpe_bytes)
        t0 = perf_counter()

        fwd_step(vocab_size, scratch["emb"].ptr, scratch["x0"].ptr, scratch["rms_init"].ptr, scratch["x1"].ptr, scratch["attn_xn"].ptr, scratch["attn_rms"].ptr, scratch["q"].ptr, scratch["k"].ptr, scratch["v"].ptr, scratch["attn_w"].ptr, scratch["attn_out"].ptr, scratch["out_flat"].ptr, scratch["mlp_xn"].ptr, scratch["mlp_rms"].ptr, scratch["h_pre"].ptr, scratch["h"].ptr, scratch["dx0"].ptr, params["wte"].ptr, params["wpe"].ptr, params["attn_wq"].ptr, params["attn_wk"].ptr, params["attn_wv"].ptr, params["attn_wo"].ptr, params["mlp_fc1"].ptr, params["mlp_fc2"].ptr, scalars["rms_inv_n"].ptr, scalars["attn_scale"].ptr, batch["input_ids"].ptr)

        bwd_step(vocab_size, scratch["dx1"].ptr, grads["lm_head"].ptr, scratch["logits"].ptr, scratch["dx0"].ptr, params["lm_head"].ptr, batch["loss_mask"].ptr, batch["inv_sum_mask"].ptr, batch["target_ids"].ptr, grads["mlp_fc2"].ptr, grads["mlp_fc1"].ptr, scratch["dh"].ptr, scratch["dh_pre"].ptr, grads["attn_wq"].ptr, grads["attn_wk"].ptr, grads["attn_wv"].ptr, grads["attn_wo"].ptr, scratch["dattn_out"].ptr, grads["wte"].ptr, grads["wpe"].ptr, scratch["emb"].ptr, scratch["rms_init"].ptr, scratch["x0"].ptr, scratch["attn_xn"].ptr, scratch["attn_rms"].ptr, scratch["q"].ptr, scratch["k"].ptr, scratch["v"].ptr, scratch["attn_w"].ptr, scratch["out_flat"].ptr, scratch["dq"].ptr, scratch["dk"].ptr, scratch["dv"].ptr, scratch["x1"].ptr, scratch["mlp_xn"].ptr, scratch["mlp_rms"].ptr, scratch["h_pre"].ptr, scratch["h"].ptr, params["attn_wq"].ptr, params["attn_wk"].ptr, params["attn_wv"].ptr, params["attn_wo"].ptr, params["mlp_fc1"].ptr, params["mlp_fc2"].ptr, scalars["rms_inv_n"].ptr, scalars["attn_scale"].ptr, batch["input_ids"].ptr)

        adam_step(flat_params.ptr, flat_grads.ptr, opt_state["m"].ptr, opt_state["v"].ptr, scalars["opt_lr"].ptr, scalars["opt_bc1"].ptr, scalars["opt_bc2"].ptr, scalars["opt_beta1"].ptr, scalars["opt_beta2"].ptr)

        step_times.append(perf_counter() - t0)

    save_times(step_times)
    W = namedtuple("W", ["data"])
    assert_weights_match({name: [[W(float(buf[i * cols + j])) for j in range(cols)] for i in range(buf.n // cols)] for name, buf, cols in named_params(params, params_layout)})
