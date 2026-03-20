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
from exo.libs.externs import expf, select
from exo.stdlib.scheduling import simplify
from utils.exo_alloc import Tensor, empty, full, normal, pack_tensors, tensor_ptrs, view_tensors, zeros, zeros_like
from utils.exo_kernels import adam, add, fill, fill3, make_embed_rms_bwd_tokens, make_embed_rms_fwd_tokens, make_lm_head_step_fused, matmul_left_t, matmul_right_t, relu, rmsnorm, rmsnorm_bwd
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


lm_head_step_fused = make_lm_head_step_fused(BLOCK_SIZE, N_EMBED)
embed_rms_fwd_tokens = make_embed_rms_fwd_tokens(BLOCK_SIZE, N_EMBED, embed_token)
embed_rms_bwd_tokens = make_embed_rms_bwd_tokens(BLOCK_SIZE, N_EMBED, embed_rms_bwd_token)


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
