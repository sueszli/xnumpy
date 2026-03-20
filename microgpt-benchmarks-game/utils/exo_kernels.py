from __future__ import annotations

from exo import *
from exo.libs.externs import select, sqrt

from exojit.patches_exo import Stack


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
def rmsnorm(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, x: f64[M, N] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    # out = x / sqrt(mean(x^2) + eps)
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = 0.0
        for j in seq(0, N):
            sumsq += x[i, j] * x[i, j]
        scale = 1.0 / sqrt(sumsq * inv_n[0] + eps[0])
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
def adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    # adam step over flat buffer (with bias correction from caller)
    inv_b1: f64 @ Stack
    inv_b2: f64 @ Stack
    inv_beta1_t: f64 @ Stack
    inv_beta2_t: f64 @ Stack
    inv_b1 = 1.0 - 0.85
    inv_b2 = 1.0 - 0.99
    inv_beta1_t = 1.0 / beta1_t[0]
    inv_beta2_t = 1.0 / beta2_t[0]

    for i in seq(0, N):
        g: f64 @ Stack
        m_val: f64 @ Stack
        v_val: f64 @ Stack
        m_hat: f64 @ Stack
        v_hat: f64 @ Stack
        g = grad[i]
        m_val = 0.85 * m[i] + inv_b1 * g
        v_val = 0.99 * v[i] + inv_b2 * g * g
        m_hat = m_val * inv_beta1_t
        v_hat = v_val * inv_beta2_t
        param[i] = param[i] - lr[0] * m_hat / (sqrt(v_hat) + 1e-8)
        m[i] = m_val
        v[i] = v_val


# TODO: attention

# @proc
# def scaled_dot_product_attention(
#     N_HEAD: size,
#     BLOCK_SIZE: size,
#     HEAD_DIM: size,
#     out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
#     q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
#     k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
#     v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM,
#     attn_w: f64[N_HEAD, BLOCK_SIZE, BLOCK_SIZE] @ DRAM,
#     scale: f64 @ Stack,
# ):
#     CAUSAL_MASK_VALUE = -1e10

#     # 1. Compute attention logits with causal mask + online softmax max
#     for h in seq(0, N_HEAD):
#         for t in seq(0, BLOCK_SIZE):
#             mx: f64 @ Stack
#             mx = CAUSAL_MASK_VALUE
#             for s in seq(0, BLOCK_SIZE):
#                 logit: f64 @ Stack
#                 if s > t:
#                     logit = CAUSAL_MASK_VALUE
#                 else:
#                     logit = 0.0
#                     for d in seq(0, HEAD_DIM):
#                         logit += q[h, t, d] * k[h, s, d]
#                     logit = logit * scale
#                 attn_w[h, t, s] = logit
#                 mx = select(mx, attn_w[h, t, s], attn_w[h, t, s], mx)

#     # 2. Softmax: exp(x - max) / sum
#             sum_val: f64 @ Stack
#             sum_val = 0.0
#             for s in seq(0, BLOCK_SIZE):
#                 attn_w[h, t, s] = expf(attn_w[h, t, s] - mx)
#                 sum_val += attn_w[h, t, s]
#             for s in seq(0, BLOCK_SIZE):
#                 attn_w[h, t, s] = attn_w[h, t, s] / sum_val

#     # 3. Aggregate: weighted sum of values
#     for h in seq(0, N_HEAD):
#         for t in seq(0, BLOCK_SIZE):
#             for d in seq(0, HEAD_DIM):
#                 acc: f64 @ Stack
#                 acc = 0.0
#                 for s in seq(0, BLOCK_SIZE):
#                     acc += attn_w[h, t, s] * v[h, s, d]
#                 out[h, t, d] = acc

#  In `attn_fwd_fused`, replace lines 113-148 with:
#    ```python
#    scaled_dot_product_attention(
#        N_HEAD, BLOCK_SIZE, HEAD_DIM,
#        out_flat_3d,  # need to reshape or use different layout
#        q, k, v, attn_w,
#        INV_SCALE
#    )
#    ```

# Flatten `out_flat_3d[N_HEAD, BLOCK_SIZE, HEAD_DIM]` to `out_flat[BLOCK_SIZE, N_EMBED]`
#    - Or modify kernel output to match existing layout directly
