from __future__ import annotations

import linecache

from exo import *
from exo.libs.externs import expf, select, sqrt

from exojit.patches_exo import Stack


@proc
def fill(M: size, N: size, x: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    # x[i, j] = value
    for i in seq(0, M):
        for j in seq(0, N):
            x[i, j] = value[0]


@proc
def add(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM):
    # out += x
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] += x[i, j]


@proc
def matmul_right_t(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[N, K] @ DRAM, zero: f64[1] @ DRAM):
    # out = x @ w^t
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[j, k]
            out[i, j] = acc


@proc
def matmul(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[K, N] @ DRAM, zero: f64[1] @ DRAM):
    # out = x @ w
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[k, j]
            out[i, j] = acc


@proc
def matmul_left_t(M: size, N: size, K: size, out: f64[N, K] @ DRAM, x: f64[M, N] @ DRAM, w: f64[M, K] @ DRAM, zero: f64[1] @ DRAM):
    # out = x^t @ w
    for j in seq(0, N):
        for k in seq(0, K):
            acc: f64 @ Stack
            acc = zero[0]
            for i in seq(0, M):
                acc += x[i, j] * w[i, k]
            out[j, k] = acc


@proc
def rmsnorm(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    # out = x / sqrt(mean(x^2) + eps)
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = zero[0]
        for j in seq(0, N):
            sumsq += x[i, j] * x[i, j]
        scale = one[0] / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(M: size, N: size, out: f64[M, N] @ DRAM, dx: f64[M, N] @ DRAM, x_pre: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    # out = dnorm + dx residual
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = zero[0]
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def relu(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM):
    # out = max(x, 0)
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = select(zero[0], x[i, j], x[i, j], zero[0])


@proc
def fill3(M: size, N: size, a: f64[M, N] @ DRAM, b: f64[M, N] @ DRAM, c: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    # fill three tensors with one scalar
    fill(M, N, a, value)
    fill(M, N, b, value)
    fill(M, N, c, value)


@proc
def softmax_xent_row(V: size, row: [f64][V] @ DRAM, loss: [f64][1] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, target: size):
    # in-place dlogits row for masked mean cross-entropy
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
def adam(N: size, param: f64[N] @ DRAM, grad: f64[N] @ DRAM, m: f64[N] @ DRAM, v: f64[N] @ DRAM, b1: f64[1] @ DRAM, b2: f64[1] @ DRAM, eps: f64[1] @ DRAM, lr: f64[1] @ DRAM, beta1_t: f64[1] @ DRAM, beta2_t: f64[1] @ DRAM):
    # flat adam update with externally supplied bias-correction terms
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


_PROC_FACTORY_ID = 0


def _make_proc(name: str, src: str, **extra_globals):
    global _PROC_FACTORY_ID
    scope = dict(globals())
    scope.update(extra_globals)
    filename = f"<exo_kernels_generated_{name}_{_PROC_FACTORY_ID}>"
    _PROC_FACTORY_ID += 1
    if not src.endswith("\n"):
        src += "\n"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    exec(compile(src, filename, "exec"), scope)
    return scope[name]


def make_lm_head_step_fused(block_size: int, n_embed: int):
    targets = ", ".join(f"target{i}: size" for i in range(block_size))
    lines = [
        "@proc",
        f"def lm_head_step_fused(V: size, dx: f64[{block_size}, {n_embed}] @ DRAM, dweight: f64[V, {n_embed}] @ DRAM, logits: f64[{block_size}, V] @ DRAM, x: f64[{block_size}, {n_embed}] @ DRAM, lm_head: f64[V, {n_embed}] @ DRAM, loss_mask: f64[{block_size}] @ DRAM, inv_sum_mask: f64[1] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, {targets}):",
    ]
    lines.extend(f"    assert target{i} < V" for i in range(block_size))
    lines.append(f"    matmul_right_t({block_size}, V, {n_embed}, logits, x, lm_head, zero)")
    lines.extend(f"    softmax_xent_row(V, logits[{i}, :], loss_mask[{i}:{i + 1}], inv_sum_mask, zero, one, target{i})" for i in range(block_size))
    lines.append(f"    matmul_left_t({block_size}, V, {n_embed}, dweight, logits, x, zero)")
    lines.append(f"    matmul({block_size}, {n_embed}, V, dx, logits, lm_head, zero)")
    return _make_proc(
        "lm_head_step_fused",
        "\n".join(lines),
    )


def make_embed_rms_fwd_tokens(block_size: int, n_embed: int, embed_token):
    inputs = ", ".join(f"input{i}: size" for i in range(block_size))
    lines = [
        "@proc",
        f"def embed_rms_fwd_tokens(V: size, emb: f64[{block_size}, {n_embed}] @ DRAM, out: f64[{block_size}, {n_embed}] @ DRAM, rms: f64[{block_size}, 1] @ DRAM, wte: f64[V, {n_embed}] @ DRAM, wpe: f64[{block_size}, {n_embed}] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM, {inputs}):",
    ]
    lines.extend(f"    assert input{i} < V" for i in range(block_size))
    lines.extend(f"    embed_token(V, emb[{i}, :], wte, wpe[{i}, :], input{i})" for i in range(block_size))
    lines.append(f"    rmsnorm({block_size}, {n_embed}, out, rms, emb, zero, one, inv_n, eps)")
    return _make_proc(
        "embed_rms_fwd_tokens",
        "\n".join(lines),
        embed_token=embed_token,
    )


def make_embed_rms_bwd_tokens(block_size: int, n_embed: int, embed_rms_bwd_token):
    inputs = ", ".join(f"input{i}: size" for i in range(block_size))
    lines = [
        "@proc",
        f"def embed_rms_bwd_tokens(V: size, g_wte: f64[V, {n_embed}] @ DRAM, g_wpe: f64[{block_size}, {n_embed}] @ DRAM, dout: f64[{block_size}, {n_embed}] @ DRAM, x: f64[{block_size}, {n_embed}] @ DRAM, rms: f64[{block_size}, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, {inputs}):",
    ]
    lines.extend(f"    assert input{i} < V" for i in range(block_size))
    lines.extend(f"    embed_rms_bwd_token(V, g_wte, g_wpe[{i}, :], dout[{i}, :], x[{i}, :], rms[{i}, 0:1], zero, inv_n, input{i})" for i in range(block_size))
    return _make_proc(
        "embed_rms_bwd_tokens",
        "\n".join(lines),
        embed_rms_bwd_token=embed_rms_bwd_token,
    )
