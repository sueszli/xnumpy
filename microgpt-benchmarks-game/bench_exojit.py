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
from exo.stdlib.scheduling import simplify
from utils.exo_kernels import adam, add, matmul, matmul_left_t, matmul_right_t, relu, relu_bwd, rmsnorm, rmsnorm_bwd, softmax
from utils.times import save_times
from utils.weights import assert_weights_match

from exojit.main import jit
from exojit.patches_exo import Stack


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


N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBED // N_HEAD
INV_SCALE = 1.0 / HEAD_DIM**0.5
CAUSAL_MASK_VALUE = -1e10


@proc
def lm_head_step_fused(vocab_size: size, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dweight: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM):
    matmul_right_t(BLOCK_SIZE, vocab_size, N_EMBED, logits, x, lm_head)
    softmax(BLOCK_SIZE, vocab_size, logits)
    for t in seq(0, BLOCK_SIZE):
        scale: f64 @ Stack
        scale = loss_mask[t] * inv_sum_mask[0]
        for v_idx in seq(0, vocab_size):
            logits[t, v_idx] = logits[t, v_idx] * scale
            if v_idx == target_ids[t]:
                logits[t, v_idx] += -inv_sum_mask[0] * loss_mask[t]
    matmul_left_t(BLOCK_SIZE, vocab_size, N_EMBED, dweight, logits, x)
    matmul(BLOCK_SIZE, N_EMBED, vocab_size, dx, logits, lm_head)


@proc
def embed_rms_fwd(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    for t in seq(0, BLOCK_SIZE):
        for e in seq(0, N_EMBED):
            emb[t, e] = wpe[t, e]
            for v in seq(0, vocab_size):
                if v == input_ids[t]:
                    emb[t, e] += wte[v, e]
    rmsnorm(BLOCK_SIZE, N_EMBED, out, rms, emb, inv_n)


@proc
def embed_rms_bwd(vocab_size: size, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dout: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
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
def attn_fwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD * BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
    rmsnorm(BLOCK_SIZE, N_EMBED, xn, rms, x, inv_n)

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
            for s in seq(0, BLOCK_SIZE):
                logit: f64 @ Stack
                if s > t:
                    logit = CAUSAL_MASK_VALUE
                else:
                    logit = 0.0
                    for d in seq(0, HEAD_DIM):
                        logit += q[h, t, d] * k[h, s, d]
                    logit = logit * INV_SCALE
                attn_w[h * BLOCK_SIZE + t, s] = logit

    softmax(N_HEAD * BLOCK_SIZE, BLOCK_SIZE, attn_w)

    for h in seq(0, N_HEAD):
        for t in seq(0, BLOCK_SIZE):
            for d in seq(0, HEAD_DIM):
                acc: f64 @ Stack
                acc = 0.0
                for s in seq(0, BLOCK_SIZE):
                    acc += attn_w[h * BLOCK_SIZE + t, s] * v[h, s, d]
                out_flat[t, h * HEAD_DIM + d] = acc

    matmul_right_t(BLOCK_SIZE, N_EMBED, N_EMBED, out, out_flat, wo)
    add(BLOCK_SIZE, N_EMBED, out, x)


@proc
def attn_bwd_fused(out: f64[BLOCK_SIZE, N_EMBED] @ DRAM, dwq: f64[N_EMBED, N_EMBED] @ DRAM, dwk: f64[N_EMBED, N_EMBED] @ DRAM, dwv: f64[N_EMBED, N_EMBED] @ DRAM, dwo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, dx: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x_pre: f64[BLOCK_SIZE, N_EMBED] @ DRAM, xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD * BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wq: f64[N_EMBED, N_EMBED] @ DRAM, wk: f64[N_EMBED, N_EMBED] @ DRAM, wv: f64[N_EMBED, N_EMBED] @ DRAM, wo: f64[N_EMBED, N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM):
    attn_tmp: f64[BLOCK_SIZE] @ Stack
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
                dot += dattn_w * attn_w[h * BLOCK_SIZE + t, s]

            for s in seq(0, BLOCK_SIZE):
                dlogit: f64 @ Stack
                dlogit = attn_w[h * BLOCK_SIZE + t, s] * (attn_tmp[s] - dot) * INV_SCALE

                for d in seq(0, HEAD_DIM):
                    dk_contrib: f64 @ Stack
                    dv_contrib: f64 @ Stack
                    dq_acc[d] += dlogit * k[h, s, d]
                    dk_contrib = dlogit * q[h, t, d]
                    dv_contrib = attn_w[h * BLOCK_SIZE + t, s] * dattn_out[h, t, d]

                    for e in seq(0, N_EMBED):
                        out[s, e] += dk_contrib * wk[h * HEAD_DIM + d, e]
                        out[s, e] += dv_contrib * wv[h * HEAD_DIM + d, e]
                        dwk[h * HEAD_DIM + d, e] += dk_contrib * xn[s, e]
                        dwv[h * HEAD_DIM + d, e] += dv_contrib * xn[s, e]

            for d in seq(0, HEAD_DIM):
                for e in seq(0, N_EMBED):
                    out[t, e] += dq_acc[d] * wq[h * HEAD_DIM + d, e]
                    dwq[h * HEAD_DIM + d, e] += dq_acc[d] * xn[t, e]

    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, out, dx, x_pre, rms, inv_n)


@proc
def fwd(vocab_size: size, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms_init: f64[BLOCK_SIZE, 1] @ DRAM, x1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD * BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dx0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, wte: f64[vocab_size, N_EMBED] @ DRAM, wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    embed_rms_fwd(vocab_size, emb, x0, rms_init, wte, wpe, inv_n, input_ids)
    attn_fwd_fused(x1, attn_xn, attn_rms, q, k, v, attn_w, out_flat, x0, attn_wq, attn_wk, attn_wv, attn_wo, inv_n)
    rmsnorm(BLOCK_SIZE, N_EMBED, mlp_xn, mlp_rms, x1, inv_n)
    matmul_right_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, h_pre, mlp_xn, mlp_fc1)
    relu(BLOCK_SIZE, 4 * N_EMBED, h, h_pre)
    matmul_right_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dx0, h, mlp_fc2)
    add(BLOCK_SIZE, N_EMBED, dx0, x1)


@proc
def bwd(vocab_size: size, dx1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, g_lm_head: f64[vocab_size, N_EMBED] @ DRAM, logits: f64[BLOCK_SIZE, vocab_size] @ DRAM, dx0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, lm_head: f64[vocab_size, N_EMBED] @ DRAM, loss_mask: f64[BLOCK_SIZE] @ DRAM, inv_sum_mask: f64[1] @ DRAM, target_ids: size[BLOCK_SIZE] @ DRAM, g_mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, g_mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, dh: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, dh_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, g_attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, g_attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, dattn_out: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, g_wte: f64[vocab_size, N_EMBED] @ DRAM, g_wpe: f64[BLOCK_SIZE, N_EMBED] @ DRAM, emb: f64[BLOCK_SIZE, N_EMBED] @ DRAM, rms_init: f64[BLOCK_SIZE, 1] @ DRAM, x0: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, attn_rms: f64[BLOCK_SIZE, 1] @ DRAM, q: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, k: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, v: f64[N_HEAD, BLOCK_SIZE, HEAD_DIM] @ DRAM, attn_w: f64[N_HEAD * BLOCK_SIZE, BLOCK_SIZE] @ DRAM, out_flat: f64[BLOCK_SIZE, N_EMBED] @ DRAM, x1: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_xn: f64[BLOCK_SIZE, N_EMBED] @ DRAM, mlp_rms: f64[BLOCK_SIZE, 1] @ DRAM, h_pre: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, h: f64[BLOCK_SIZE, 4 * N_EMBED] @ DRAM, attn_wq: f64[N_EMBED, N_EMBED] @ DRAM, attn_wk: f64[N_EMBED, N_EMBED] @ DRAM, attn_wv: f64[N_EMBED, N_EMBED] @ DRAM, attn_wo: f64[N_EMBED, N_EMBED] @ DRAM, mlp_fc1: f64[4 * N_EMBED, N_EMBED] @ DRAM, mlp_fc2: f64[N_EMBED, 4 * N_EMBED] @ DRAM, inv_n: f64[1] @ DRAM, input_ids: size[BLOCK_SIZE] @ DRAM):
    lm_head_step_fused(vocab_size, dx1, g_lm_head, logits, dx0, lm_head, loss_mask, inv_sum_mask, target_ids)
    matmul_left_t(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, g_mlp_fc2, dx1, h)
    matmul(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, dh, dx1, mlp_fc2)
    relu_bwd(BLOCK_SIZE, 4 * N_EMBED, dh_pre, dh, h_pre)
    matmul_left_t(BLOCK_SIZE, 4 * N_EMBED, N_EMBED, g_mlp_fc1, dh_pre, mlp_xn)
    matmul(BLOCK_SIZE, N_EMBED, 4 * N_EMBED, dx0, dh_pre, mlp_fc1)
    rmsnorm_bwd(BLOCK_SIZE, N_EMBED, dx0, dx1, x1, mlp_rms, inv_n)
    attn_bwd_fused(dx1, g_attn_wq, g_attn_wk, g_attn_wv, g_attn_wo, dattn_out, dx0, x0, attn_xn, attn_rms, q, k, v, attn_w, out_flat, attn_wq, attn_wk, attn_wv, attn_wo, inv_n)
    embed_rms_bwd(vocab_size, g_wte, g_wpe, dx1, emb, rms_init, inv_n, input_ids)


PARAMS_FIELDS = "wte wpe lm_head attn_wq attn_wk attn_wv attn_wo mlp_fc1 mlp_fc2".split()
SCRATCH_FIELDS = "emb rms_init x0 x1 logits attn_xn attn_rms q k v attn_w out_flat mlp_xn mlp_rms h_pre h dh dh_pre dx0 dx1 dattn_out".split()
SCALARS_FIELDS = "opt_lr opt_bc1 opt_bc2 rms_inv_n".split()


def init_normal_(buf: Buf, *, scale: float):
    for i in range(buf.n):
        buf[i] = random.gauss(0.0, scale)


def layout_numel(layout):
    return sum(prod(s) for s in layout)


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
        (N_HEAD * BLOCK_SIZE, BLOCK_SIZE),
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
    )


def named_params(params, layout):
    names = ("wte", "wpe", "lm_head", "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo", "layer0.mlp_fc1", "layer0.mlp_fc2")
    return [(n, params[PARAMS_FIELDS[i]], layout[i][1]) for i, n in enumerate(names)]


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
    random.seed(42)
    num_steps = 1000
    fwd_step = jit(simplify(fwd))._raw
    bwd_step = jit(simplify(bwd))._raw

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
    flat_params = Buf(layout_numel(params_layout))
    params = bind(PARAMS_FIELDS, flat_params, params_layout)
    for name, buf, _ in named_params(params, params_layout):
        init_normal_(buf, scale=0.08)

    flat_grads = Buf(flat_params.n)
    grads = bind(PARAMS_FIELDS, flat_grads, params_layout)
    opt_state = {"m": Buf(flat_params.n), "v": Buf(flat_params.n)}

    scratch = bind(SCRATCH_FIELDS, Buf(layout_numel(scratch_layout(vocab_size))), scratch_layout(vocab_size))
    scalars = bind(SCALARS_FIELDS, Buf(4), ((1,), (1,), (1,), (1,)))
    scalars["rms_inv_n"][0] = 1.0 / N_EMBED

    adam_step = jit(simplify(adam.partial_eval(N=flat_params.n)))._raw

    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = vocab_size - 1
    tokenized = [tokenize(doc, c2i, bos) for doc in docs]

    g_wte_bytes = grads["wte"].n * 8
    g_wpe_bytes = grads["wpe"].n * 8
    lr_t = [0.01 * (1.0 - step / num_steps) for step in range(num_steps)]
    bc1 = [1.0 - 0.85 ** (step + 1) for step in range(num_steps)]
    bc2 = [1.0 - 0.99 ** (step + 1) for step in range(num_steps)]
    memset = ctypes.memset
    perf_counter = time.perf_counter
    step_times = []

    for step in range(num_steps):
        scalars["opt_lr"][0] = lr_t[step]
        scalars["opt_bc1"][0] = bc1[step]
        scalars["opt_bc2"][0] = bc2[step]
        batch = tokenized[step % len(tokenized)]
        memset(grads["wte"].ptr, 0, g_wte_bytes)
        memset(grads["wpe"].ptr, 0, g_wpe_bytes)
        t0 = perf_counter()

        fwd_step(vocab_size, scratch["emb"].ptr, scratch["x0"].ptr, scratch["rms_init"].ptr, scratch["x1"].ptr, scratch["attn_xn"].ptr, scratch["attn_rms"].ptr, scratch["q"].ptr, scratch["k"].ptr, scratch["v"].ptr, scratch["attn_w"].ptr, scratch["out_flat"].ptr, scratch["mlp_xn"].ptr, scratch["mlp_rms"].ptr, scratch["h_pre"].ptr, scratch["h"].ptr, scratch["dx0"].ptr, params["wte"].ptr, params["wpe"].ptr, params["attn_wq"].ptr, params["attn_wk"].ptr, params["attn_wv"].ptr, params["attn_wo"].ptr, params["mlp_fc1"].ptr, params["mlp_fc2"].ptr, scalars["rms_inv_n"].ptr, batch["input_ids"].ptr)

        bwd_step(vocab_size, scratch["dx1"].ptr, grads["lm_head"].ptr, scratch["logits"].ptr, scratch["dx0"].ptr, params["lm_head"].ptr, batch["loss_mask"].ptr, batch["inv_sum_mask"].ptr, batch["target_ids"].ptr, grads["mlp_fc2"].ptr, grads["mlp_fc1"].ptr, scratch["dh"].ptr, scratch["dh_pre"].ptr, grads["attn_wq"].ptr, grads["attn_wk"].ptr, grads["attn_wv"].ptr, grads["attn_wo"].ptr, scratch["dattn_out"].ptr, grads["wte"].ptr, grads["wpe"].ptr, scratch["emb"].ptr, scratch["rms_init"].ptr, scratch["x0"].ptr, scratch["attn_xn"].ptr, scratch["attn_rms"].ptr, scratch["q"].ptr, scratch["k"].ptr, scratch["v"].ptr, scratch["attn_w"].ptr, scratch["out_flat"].ptr, scratch["x1"].ptr, scratch["mlp_xn"].ptr, scratch["mlp_rms"].ptr, scratch["h_pre"].ptr, scratch["h"].ptr, params["attn_wq"].ptr, params["attn_wk"].ptr, params["attn_wv"].ptr, params["attn_wo"].ptr, params["mlp_fc1"].ptr, params["mlp_fc2"].ptr, scalars["rms_inv_n"].ptr, batch["input_ids"].ptr)

        adam_step(flat_params.ptr, flat_grads.ptr, opt_state["m"].ptr, opt_state["v"].ptr, scalars["opt_lr"].ptr, scalars["opt_bc1"].ptr, scalars["opt_bc2"].ptr)

        step_times.append(perf_counter() - t0)

    save_times(step_times)
    W = namedtuple("W", ["data"])
    assert_weights_match({name: [[W(float(buf[i * cols + j])) for j in range(cols)] for i in range(buf.n // cols)] for name, buf, cols in named_params(params, params_layout)})
