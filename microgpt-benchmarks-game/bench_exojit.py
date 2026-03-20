# /// script
# requires-python = "==3.14.*"
# dependencies = []
# ///

from __future__ import annotations

import ctypes
import random
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from exo import *
from exo.libs.externs import expf, select, sqrt
from exo.stdlib.scheduling import simplify
from utils.exo_alloc import Tensor, empty, full, normal, reshape, zeros, zeros_like
from utils.exo_kernels import add, fill, fill3, matmul, matmul_left_t, matmul_right_t, relu, rmsnorm, rmsnorm_bwd
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


def pack_tensors(tensors: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor], int]:
    total = sum(t._size for t in tensors.values())
    flat = empty((total,), dtype=float)
    flat_ptr = flat.ctypes.data
    elt_bytes = ctypes.sizeof(flat._ctype)
    views = {}
    offset = 0
    for name, tensor in tensors.items():
        ctypes.memmove(flat_ptr + offset * elt_bytes, tensor.ctypes.data, tensor._size * elt_bytes)
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor._size
    return flat, views, elt_bytes


def view_tensors(flat: Tensor, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    views = {}
    offset = 0
    for name, tensor in tensors.items():
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor._size
    return views


def tensor_ptrs(tensors: dict[str, Tensor]) -> dict[str, int]:
    return {name: tensor.ctypes.data for name, tensor in tensors.items()}


@proc
def embed_token(V: size, emb_row: [f64][N_EMBED] @ DRAM, wte: f64[V, N_EMBED] @ DRAM, wpe_row: [f64][N_EMBED] @ DRAM, input: size):
    assert input < V
    for e in seq(0, N_EMBED):
        emb_row[e] = wte[input, e] + wpe_row[e]


@proc
def embed_rms_bwd_token(V: size, g_wte: f64[V, N_EMBED] @ DRAM, g_wpe_row: [f64][N_EMBED] @ DRAM, dout_row: [f64][N_EMBED] @ DRAM, x_row: [f64][N_EMBED] @ DRAM, rms_row: [f64][1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, input: size):
    assert input < V
    dot: f64 @ Stack
    scale: f64 @ Stack
    corr: f64 @ Stack
    dot = zero[0]
    scale = rms_row[0]
    for e in seq(0, N_EMBED):
        dot += dout_row[e] * x_row[e]
    corr = scale * scale * scale * inv_n[0] * dot
    for e in seq(0, N_EMBED):
        dx: f64 @ Stack
        dx = dout_row[e] * scale - x_row[e] * corr
        g_wte[input, e] += dx
        g_wpe_row[e] = dx


@proc
def softmax_xent_row(V: size, row: [f64][V] @ DRAM, loss: [f64][1] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target: size):
    assert target < V
    mx: f64 @ Stack
    sum_val: f64 @ Stack
    scale: f64 @ Stack
    val: f64 @ Stack
    inv_denom: f64 @ Stack
    mx = row[0]
    for v_idx in seq(1, V):
        mx = select(mx, row[v_idx], row[v_idx], mx)
    sum_val = zero[0]
    for v_idx in seq(0, V):
        val = expf(row[v_idx] - mx)
        row[v_idx] = val
        sum_val += val
    inv_denom = one[0] / sum_val
    scale = loss[0] * inv_sum_mask[0] * inv_denom
    for v_idx in seq(0, V):
        row[v_idx] = row[v_idx] * scale
    row[target] += -inv_sum_mask[0] * loss[0]


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
    fill(BLOCK_SIZE, N_EMBED, out, zero)
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dw2, dx, h, zero)
    fill(4 * N_EMBED, N_EMBED, dw1, zero)

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
def adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, b1: f64[1] @ DRAM, b2: f64[1] @ DRAM, eps: f64[1] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
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
def attn_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
    fill(BLOCK_SIZE, N_EMBED, out, zero)
    matmul_left_t(BLOCK_SIZE, N_EMBED, N_EMBED, dwo, dx, out_flat, zero)
    fill3(N_EMBED, N_EMBED, dwq, dwk, dwv, zero)

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


@proc
def lm_head_step_fused(V: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[V, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, V] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[V, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target0: size, target1: size, target2: size, target3: size, target4: size, target5: size, target6: size, target7: size, target8: size, target9: size, target10: size, target11: size, target12: size, target13: size, target14: size, target15: size):
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
    matmul_right_t(BLOCK_SIZE, V, N_EMBED, logits, x, lm_head, zero)
    softmax_xent_row(V, logits[0, :], loss_mask[0:1], inv_sum_mask, zero, one, target0)
    softmax_xent_row(V, logits[1, :], loss_mask[1:2], inv_sum_mask, zero, one, target1)
    softmax_xent_row(V, logits[2, :], loss_mask[2:3], inv_sum_mask, zero, one, target2)
    softmax_xent_row(V, logits[3, :], loss_mask[3:4], inv_sum_mask, zero, one, target3)
    softmax_xent_row(V, logits[4, :], loss_mask[4:5], inv_sum_mask, zero, one, target4)
    softmax_xent_row(V, logits[5, :], loss_mask[5:6], inv_sum_mask, zero, one, target5)
    softmax_xent_row(V, logits[6, :], loss_mask[6:7], inv_sum_mask, zero, one, target6)
    softmax_xent_row(V, logits[7, :], loss_mask[7:8], inv_sum_mask, zero, one, target7)
    softmax_xent_row(V, logits[8, :], loss_mask[8:9], inv_sum_mask, zero, one, target8)
    softmax_xent_row(V, logits[9, :], loss_mask[9:10], inv_sum_mask, zero, one, target9)
    softmax_xent_row(V, logits[10, :], loss_mask[10:11], inv_sum_mask, zero, one, target10)
    softmax_xent_row(V, logits[11, :], loss_mask[11:12], inv_sum_mask, zero, one, target11)
    softmax_xent_row(V, logits[12, :], loss_mask[12:13], inv_sum_mask, zero, one, target12)
    softmax_xent_row(V, logits[13, :], loss_mask[13:14], inv_sum_mask, zero, one, target13)
    softmax_xent_row(V, logits[14, :], loss_mask[14:15], inv_sum_mask, zero, one, target14)
    softmax_xent_row(V, logits[15, :], loss_mask[15:16], inv_sum_mask, zero, one, target15)
    matmul_left_t(BLOCK_SIZE, V, N_EMBED, dweight, logits, x, zero)
    matmul(BLOCK_SIZE, N_EMBED, V, dx, logits, lm_head, zero)


@proc
def embed_rms_fwd_tokens(V: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[V, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM, input0: size, input1: size, input2: size, input3: size, input4: size, input5: size, input6: size, input7: size, input8: size, input9: size, input10: size, input11: size, input12: size, input13: size, input14: size, input15: size):
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
    embed_token(V, emb[0, :], wte, wpe[0, :], input0)
    embed_token(V, emb[1, :], wte, wpe[1, :], input1)
    embed_token(V, emb[2, :], wte, wpe[2, :], input2)
    embed_token(V, emb[3, :], wte, wpe[3, :], input3)
    embed_token(V, emb[4, :], wte, wpe[4, :], input4)
    embed_token(V, emb[5, :], wte, wpe[5, :], input5)
    embed_token(V, emb[6, :], wte, wpe[6, :], input6)
    embed_token(V, emb[7, :], wte, wpe[7, :], input7)
    embed_token(V, emb[8, :], wte, wpe[8, :], input8)
    embed_token(V, emb[9, :], wte, wpe[9, :], input9)
    embed_token(V, emb[10, :], wte, wpe[10, :], input10)
    embed_token(V, emb[11, :], wte, wpe[11, :], input11)
    embed_token(V, emb[12, :], wte, wpe[12, :], input12)
    embed_token(V, emb[13, :], wte, wpe[13, :], input13)
    embed_token(V, emb[14, :], wte, wpe[14, :], input14)
    embed_token(V, emb[15, :], wte, wpe[15, :], input15)
    rmsnorm(BLOCK_SIZE, N_EMBED, out, rms, emb, zero, one, inv_n, eps)


@proc
def embed_rms_bwd_tokens(V: size, g_wte: f64[V, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, input0: size, input1: size, input2: size, input3: size, input4: size, input5: size, input6: size, input7: size, input8: size, input9: size, input10: size, input11: size, input12: size, input13: size, input14: size, input15: size):
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
    embed_rms_bwd_token(V, g_wte, g_wpe[0, :], dout[0, :], x[0, :], rms[0, 0:1], zero, inv_n, input0)
    embed_rms_bwd_token(V, g_wte, g_wpe[1, :], dout[1, :], x[1, :], rms[1, 0:1], zero, inv_n, input1)
    embed_rms_bwd_token(V, g_wte, g_wpe[2, :], dout[2, :], x[2, :], rms[2, 0:1], zero, inv_n, input2)
    embed_rms_bwd_token(V, g_wte, g_wpe[3, :], dout[3, :], x[3, :], rms[3, 0:1], zero, inv_n, input3)
    embed_rms_bwd_token(V, g_wte, g_wpe[4, :], dout[4, :], x[4, :], rms[4, 0:1], zero, inv_n, input4)
    embed_rms_bwd_token(V, g_wte, g_wpe[5, :], dout[5, :], x[5, :], rms[5, 0:1], zero, inv_n, input5)
    embed_rms_bwd_token(V, g_wte, g_wpe[6, :], dout[6, :], x[6, :], rms[6, 0:1], zero, inv_n, input6)
    embed_rms_bwd_token(V, g_wte, g_wpe[7, :], dout[7, :], x[7, :], rms[7, 0:1], zero, inv_n, input7)
    embed_rms_bwd_token(V, g_wte, g_wpe[8, :], dout[8, :], x[8, :], rms[8, 0:1], zero, inv_n, input8)
    embed_rms_bwd_token(V, g_wte, g_wpe[9, :], dout[9, :], x[9, :], rms[9, 0:1], zero, inv_n, input9)
    embed_rms_bwd_token(V, g_wte, g_wpe[10, :], dout[10, :], x[10, :], rms[10, 0:1], zero, inv_n, input10)
    embed_rms_bwd_token(V, g_wte, g_wpe[11, :], dout[11, :], x[11, :], rms[11, 0:1], zero, inv_n, input11)
    embed_rms_bwd_token(V, g_wte, g_wpe[12, :], dout[12, :], x[12, :], rms[12, 0:1], zero, inv_n, input12)
    embed_rms_bwd_token(V, g_wte, g_wpe[13, :], dout[13, :], x[13, :], rms[13, 0:1], zero, inv_n, input13)
    embed_rms_bwd_token(V, g_wte, g_wpe[14, :], dout[14, :], x[14, :], rms[14, 0:1], zero, inv_n, input14)
    embed_rms_bwd_token(V, g_wte, g_wpe[15, :], dout[15, :], x[15, :], rms[15, 0:1], zero, inv_n, input15)


def tokenize(doc: str, c2i: dict[str, int], bos: int, ptrs: dict[str, dict[str, int]], zero_ptr: int, one_ptr: int, rms_inv_n_ptr: int, rms_eps_ptr: int):
    p, g, s = ptrs["param"], ptrs["grad"], ptrs["scratch"]
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    inputs = [0] * BLOCK_SIZE
    targets = [0] * BLOCK_SIZE
    loss_mask = zeros((BLOCK_SIZE,), dtype=float)
    for i in range(n):
        inputs[i] = tokens[i]
        targets[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    inv_sum_mask = full((1,), 1.0 / max(1, n), dtype=float)
    return (
        (s["emb"], s["x0"], s["rms_init"], p["wte"], p["wpe"], zero_ptr, one_ptr, rms_inv_n_ptr, rms_eps_ptr, *inputs),
        (s["dx1"], g["lm_head"], s["logits"], s["dx0"], p["lm_head"], loss_mask.ctypes.data, inv_sum_mask.ctypes.data, zero_ptr, one_ptr, *targets),
        (g["wte"], g["wpe"], s["dx1"], s["emb"], s["rms_init"], zero_ptr, rms_inv_n_ptr, *inputs),
        loss_mask,
        inv_sum_mask,
    )


def wrap_state_dict(state_dict: dict[str, Tensor]) -> dict[str, list[list[object]]]:
    class W:
        __slots__ = ("data",)

        def __init__(self, data: float):
            self.data = data

    return {name: [[W(float(tensor[i, j])) for j in range(tensor.shape[1])] for i in range(tensor.shape[0])] for name, tensor in state_dict.items()}


def main() -> None:
    random.seed(42)
    n_layer = 1
    num_steps = 1000
    attn_fwd, attn_bwd, mlp_fwd, mlp_bwd = (jit(simplify(proc))._raw for proc in (attn_fwd_fused, attn_bwd_fused, mlp_fwd_fused, mlp_bwd_fused))

    docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
    random.shuffle(docs)
    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1

    state_dict = {
        "wte": normal((vocab_size, N_EMBED), scale=0.08),
        "wpe": normal((BLOCK_SIZE, N_EMBED), scale=0.08),
        "lm_head": normal((vocab_size, N_EMBED), scale=0.08),
    }
    for i in range(n_layer):
        prefix = f"layer{i}"
        for name, shape in (
            ("attn_wq", (N_EMBED, N_EMBED)),
            ("attn_wk", (N_EMBED, N_EMBED)),
            ("attn_wv", (N_EMBED, N_EMBED)),
            ("attn_wo", (N_EMBED, N_EMBED)),
            ("mlp_fc1", (4 * N_EMBED, N_EMBED)),
            ("mlp_fc2", (N_EMBED, 4 * N_EMBED)),
        ):
            state_dict[f"{prefix}.{name}"] = normal(shape, scale=0.08)

    flat_params, state_dict, elt_bytes = pack_tensors(state_dict)
    flat_grads, opt_m, opt_v = zeros_like(flat_params), zeros_like(flat_params), zeros_like(flat_params)
    grads = view_tensors(flat_grads, state_dict)

    opt_lr, opt_bc1, opt_bc2 = empty((1,)), empty((1,)), empty((1,))
    zero, one, rms_inv_n, rms_eps, adam_b1, adam_b2, adam_eps = (full((1,), x) for x in (0.0, 1.0, 1.0 / N_EMBED, 1e-5, 0.85, 0.99, 1e-8))

    scratch = {
        name: empty(shape, dtype=float)
        for name, shape in (
            ("emb", (BLOCK_SIZE, N_EMBED)),
            ("rms_init", (BLOCK_SIZE, 1)),
            ("x0", (BLOCK_SIZE, N_EMBED)),
            ("x1", (BLOCK_SIZE, N_EMBED)),
            ("logits", (BLOCK_SIZE, vocab_size)),
            ("attn_xn", (BLOCK_SIZE, N_EMBED)),
            ("attn_rms", (BLOCK_SIZE, 1)),
            ("q", (N_HEAD, BLOCK_SIZE, HEAD_DIM)),
            ("k", (N_HEAD, BLOCK_SIZE, HEAD_DIM)),
            ("v_buf", (N_HEAD, BLOCK_SIZE, HEAD_DIM)),
            ("attn_w", (N_HEAD, BLOCK_SIZE, BLOCK_SIZE)),
            ("out_flat", (BLOCK_SIZE, N_EMBED)),
            ("mlp_xn", (BLOCK_SIZE, N_EMBED)),
            ("mlp_rms", (BLOCK_SIZE, 1)),
            ("h_pre", (BLOCK_SIZE, 4 * N_EMBED)),
            ("h_buf", (BLOCK_SIZE, 4 * N_EMBED)),
            ("dx0", (BLOCK_SIZE, N_EMBED)),
            ("dx1", (BLOCK_SIZE, N_EMBED)),
            ("dattn_out", (N_HEAD, BLOCK_SIZE, HEAD_DIM)),
        )
    }

    lm_head_step = jit(simplify(lm_head_step_fused.partial_eval(V=vocab_size)), raw=True)
    embed_rms_fwd = jit(simplify(embed_rms_fwd_tokens.partial_eval(V=vocab_size)), raw=True)
    embed_rms_bwd = jit(simplify(embed_rms_bwd_tokens.partial_eval(V=vocab_size)), raw=True)
    adam_step = jit(simplify(adam.partial_eval(N=flat_params._size)))._raw

    ptrs = {
        "param": tensor_ptrs(state_dict),
        "grad": tensor_ptrs(grads),
        "scratch": tensor_ptrs(scratch),
    }
    p, g, s = ptrs["param"], ptrs["grad"], ptrs["scratch"]

    attn_fwd_args = (
        s["x1"],
        s["attn_xn"],
        s["attn_rms"],
        s["q"],
        s["k"],
        s["v_buf"],
        s["attn_w"],
        s["out_flat"],
        s["x0"],
        p["layer0.attn_wq"],
        p["layer0.attn_wk"],
        p["layer0.attn_wv"],
        p["layer0.attn_wo"],
        zero.ctypes.data,
        one.ctypes.data,
        rms_inv_n.ctypes.data,
        rms_eps.ctypes.data,
    )
    mlp_fwd_args = (
        s["dx0"],
        s["mlp_xn"],
        s["mlp_rms"],
        s["h_pre"],
        s["h_buf"],
        s["x1"],
        p["layer0.mlp_fc1"],
        p["layer0.mlp_fc2"],
        zero.ctypes.data,
        one.ctypes.data,
        rms_inv_n.ctypes.data,
        rms_eps.ctypes.data,
    )
    mlp_bwd_args = (
        s["dx0"],
        g["layer0.mlp_fc1"],
        g["layer0.mlp_fc2"],
        s["dx1"],
        s["x1"],
        s["mlp_xn"],
        s["mlp_rms"],
        s["h_pre"],
        s["h_buf"],
        p["layer0.mlp_fc1"],
        p["layer0.mlp_fc2"],
        zero.ctypes.data,
        rms_inv_n.ctypes.data,
    )
    attn_bwd_args = (
        s["dx1"],
        g["layer0.attn_wq"],
        g["layer0.attn_wk"],
        g["layer0.attn_wv"],
        g["layer0.attn_wo"],
        s["dattn_out"],
        s["dx0"],
        s["x0"],
        s["attn_xn"],
        s["attn_rms"],
        s["q"],
        s["k"],
        s["v_buf"],
        s["attn_w"],
        s["out_flat"],
        p["layer0.attn_wq"],
        p["layer0.attn_wk"],
        p["layer0.attn_wv"],
        p["layer0.attn_wo"],
        zero.ctypes.data,
        rms_inv_n.ctypes.data,
    )
    adam_args = (
        flat_params.ctypes.data,
        flat_grads.ctypes.data,
        opt_m.ctypes.data,
        opt_v.ctypes.data,
        adam_b1.ctypes.data,
        adam_b2.ctypes.data,
        adam_eps.ctypes.data,
        opt_lr.ctypes.data,
        opt_bc1.ctypes.data,
        opt_bc2.ctypes.data,
    )

    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = vocab_size - 1
    tokenized = [tokenize(doc, c2i, bos, ptrs, zero.ctypes.data, one.ctypes.data, rms_inv_n.ctypes.data, rms_eps.ctypes.data) for doc in docs]

    g_wte_bytes = grads["wte"]._size * elt_bytes
    g_wpe_bytes = grads["wpe"]._size * elt_bytes
    lr_t = [0.01 * (1.0 - step / num_steps) for step in range(num_steps)]
    bc1 = [1.0 - 0.85 ** (step + 1) for step in range(num_steps)]
    bc2 = [1.0 - 0.99 ** (step + 1) for step in range(num_steps)]
    memset = ctypes.memset
    perf_counter = time.perf_counter
    step_times = []

    for step in range(num_steps):
        opt_lr._buf[0] = lr_t[step]
        opt_bc1._buf[0] = bc1[step]
        opt_bc2._buf[0] = bc2[step]
        embed_args, lm_head_args, embed_bwd_args, _, _ = tokenized[step % len(tokenized)]
        memset(g["wte"], 0, g_wte_bytes)
        memset(g["wpe"], 0, g_wpe_bytes)
        t0 = perf_counter()
        embed_rms_fwd(*embed_args)
        attn_fwd(*attn_fwd_args)
        mlp_fwd(*mlp_fwd_args)
        lm_head_step(*lm_head_args)
        mlp_bwd(*mlp_bwd_args)
        attn_bwd(*attn_bwd_args)
        embed_rms_bwd(*embed_bwd_args)
        adam_step(*adam_args)
        step_times.append(perf_counter() - t0)

    save_times(step_times)
    assert_weights_match(wrap_state_dict(state_dict))


if __name__ == "__main__":
    main()
