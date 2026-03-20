from __future__ import annotations

from exo import *
from exo.libs.externs import expf, select, sqrt

from exojit.patches_exo import Stack, log


@proc
def add(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM):
    # out += x
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] += x[i, j]


@proc
def matmul(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[K, N] @ DRAM):
    # out = x @ w
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += x[i, k] * w[k, j]
            out[i, j] = acc


@proc
def matmul_right_t(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[N, K] @ DRAM):
    # out = x @ w^t
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = 0.0
            for k in seq(0, K):
                acc += x[i, k] * w[j, k]
            out[i, j] = acc


@proc
def matmul_left_t(M: size, N: size, K: size, out: f64[N, K] @ DRAM, x: f64[M, N] @ DRAM, w: f64[M, K] @ DRAM):
    # out = x^t @ w
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
        mx = -1e10
        for j in seq(0, N):
            mx = select(mx, x[i, j], x[i, j], mx)
        sum_val = 0.0
        for j in seq(0, N):
            x[i, j] = expf(x[i, j] - mx)
            sum_val += x[i, j]
        for j in seq(0, N):
            x[i, j] = x[i, j] / sum_val


@proc
def relu(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM):
    # out = max(x, 0)
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = select(0.0, x[i, j], x[i, j], 0.0)


@proc
def relu_bwd(M: size, N: size, out: f64[M, N] @ DRAM, dout: f64[M, N] @ DRAM, x_pre: f64[M, N] @ DRAM):
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = select(0.0, x_pre[i, j], dout[i, j], 0.0)


@proc
def rmsnorm(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, x: f64[M, N] @ DRAM, inv_n: f64[1] @ DRAM):
    # out = x / sqrt(mean(x^2) + eps)
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + 1e-5)
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(M: size, N: size, out: f64[M, N] @ DRAM, dx: f64[M, N] @ DRAM, x_pre: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, inv_n: f64[1] @ DRAM):
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = 0.0
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def cross_entropy(M: size, N: size, loss: f64[1] @ DRAM, probs: f64[M, N] @ DRAM, target_ids: size[M] @ DRAM, loss_mask: f64[M] @ DRAM, inv_sum_mask: f64[1] @ DRAM):
    total: f64 @ Stack
    total = 0.0
    for t in seq(0, M):
        p_target: f64 @ Stack
        p_target = 0.0
        for v_idx in seq(0, N):
            if v_idx == target_ids[t]:
                p_target = probs[t, v_idx]
        total += loss_mask[t] * log(p_target + 1e-10)
    loss[0] = -inv_sum_mask[0] * total


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


@proc
def dot_product_attention(S: size, H: size, D: size, out: f64[H, S, D] @ DRAM, q: f64[H, S, D] @ DRAM, k: f64[H, S, D] @ DRAM, v: f64[H, S, D] @ DRAM, attn_w: f64[H, S, S] @ DRAM, scale: f64[1] @ DRAM):
    for h in seq(0, H):
        for t in seq(0, S):
            for s in seq(0, S):
                logit: f64 @ Stack
                if s > t:
                    logit = -1e10
                else:
                    logit = 0.0
                    for d in seq(0, D):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * scale[0]
                attn_w[h, t, s] = logit

    for h in seq(0, H):
        for t in seq(0, S):
            mx: f64 @ Stack
            sum_val: f64 @ Stack
            mx = -1e10
            for s in seq(0, S):
                mx = select(mx, attn_w[h, t, s], attn_w[h, t, s], mx)
            sum_val = 0.0
            for s in seq(0, S):
                attn_w[h, t, s] = expf(attn_w[h, t, s] - mx)
                sum_val += attn_w[h, t, s]
            for s in seq(0, S):
                attn_w[h, t, s] = attn_w[h, t, s] / sum_val

    for h in seq(0, H):
        for t in seq(0, S):
            for d in seq(0, D):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, S):
                    acc += attn_w[h, t, s] * v[h, s, d]
                out[h, t, d] = acc


@proc
def dot_product_attention_bwd(S: size, H: size, D: size, dq: f64[H, S, D] @ DRAM, dk: f64[H, S, D] @ DRAM, dv: f64[H, S, D] @ DRAM, dout: f64[H, S, D] @ DRAM, q: f64[H, S, D] @ DRAM, k: f64[H, S, D] @ DRAM, v: f64[H, S, D] @ DRAM, attn_w: f64[H, S, S] @ DRAM, scale: f64[1] @ DRAM):
    attn_tmp: f64[S] @ Stack
    for h in seq(0, H):
        for t in seq(0, S):
            for d in seq(0, D):
                dv[h, t, d] = 0.0
                dq[h, t, d] = 0.0
        for s in seq(0, S):
            for d in seq(0, D):
                dk[h, s, d] = 0.0

    for h in seq(0, H):
        for t in seq(0, S):
            dot: f64 @ Stack
            dot = 0.0
            for s in seq(0, S):
                dattn_w: f64 @ Stack
                dattn_w = 0.0
                for d in seq(0, D):
                    dattn_w += dout[h, t, d] * v[h, s, d]
                    dv[h, s, d] += attn_w[h, t, s] * dout[h, t, d]
                attn_tmp[s] = dattn_w
                dot += dattn_w * attn_w[h, t, s]

            for s in seq(0, S):
                dlogit: f64 @ Stack
                if s > t:
                    dlogit = 0.0
                else:
                    dlogit = attn_w[h, t, s] * (attn_tmp[s] - dot) * scale[0]
                for d in seq(0, D):
                    dq[h, t, d] += dlogit * k[h, s, d]
                    dk[h, s, d] += dlogit * q[h, t, d]
