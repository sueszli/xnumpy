# /// script
# requires-python = "==3.14.*"
# dependencies = ["tqdm"]
# ///

import math
import random
import time
from collections import namedtuple
from operator import mul
from pathlib import Path

from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000

LayerCache = namedtuple("LayerCache", ["x_pre_attn", "xn_attn", "rms_attn", "q", "k", "v", "attn_w", "attn_out_flat", "x_pre_mlp", "xn_mlp", "rms_mlp", "h_pre", "h"])
FwdCache = namedtuple("FwdCache", ["input_ids", "target_ids", "loss_mask", "sum_mask", "emb", "rms_init", "x", "probs", "layer_caches"])


def linear_fwd(x: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    n = len(x)
    out_dim = len(W)
    out = [[0.0] * out_dim for _ in range(n)]
    for i in range(n):
        xi = x[i]
        out_i = out[i]
        for j in range(out_dim):
            out_i[j] = sum(map(mul, xi, W[j]))
    return out


def linear_bwd_x(dout: list[list[float]], W: list[list[float]]) -> list[list[float]]:
    n = len(dout)
    k = len(W[0])
    WT = list(zip(*W))
    dx = [[0.0] * k for _ in range(n)]
    for i in range(n):
        dout_i = dout[i]
        dx_i = dx[i]
        for d in range(k):
            dx_i[d] = sum(map(mul, dout_i, WT[d]))
    return dx


def linear_bwd_w(dout: list[list[float]], x: list[list[float]]) -> list[list[float]]:
    out_dim = len(dout[0])
    k = len(x[0])
    doutT = list(zip(*dout))
    xT = list(zip(*x))
    dW = [[0.0] * k for _ in range(out_dim)]
    for j in range(out_dim):
        doutT_j = doutT[j]
        dW_j = dW[j]
        for d in range(k):
            dW_j[d] = sum(map(mul, doutT_j, xT[d]))
    return dW


def rmsnorm_fwd(x: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    n = len(x)
    d = len(x[0])
    inv_d = 1.0 / d
    out = [[0.0] * d for _ in range(n)]
    rms = [0.0] * n
    for i in range(n):
        row = x[i]
        scale = (sum(map(mul, row, row)) * inv_d + 1e-5) ** -0.5
        rms[i] = scale
        out_i = out[i]
        for j in range(d):
            out_i[j] = row[j] * scale
    return out, rms


def rmsnorm_bwd(dout: list[list[float]], x: list[list[float]], rms: list[float]) -> list[list[float]]:
    n = len(x)
    d = len(x[0])
    dx = [[0.0] * d for _ in range(n)]
    for i in range(n):
        do, row, scale = dout[i], x[i], rms[i]
        scale3_over_d = (scale**3) / d
        dot = sum(map(mul, do, row))
        dx_i = dx[i]
        for j in range(d):
            dx_i[j] = do[j] * scale - scale3_over_d * row[j] * dot
    return dx


def layer_fwd(x: list[list[float]], params: dict, li: int) -> tuple[list[list[float]], LayerCache]:
    n = len(x)
    head_dim = N_EMBED // N_HEAD
    wq = params[f"layer{li}.attn_wq"]
    wk = params[f"layer{li}.attn_wk"]
    wv = params[f"layer{li}.attn_wv"]
    wo = params[f"layer{li}.attn_wo"]
    fc1 = params[f"layer{li}.mlp_fc1"]
    fc2 = params[f"layer{li}.mlp_fc2"]

    x_pre_attn = x
    xn_attn, rms_attn = rmsnorm_fwd(x)

    q_flat = linear_fwd(xn_attn, wq)
    k_flat = linear_fwd(xn_attn, wk)
    v_flat = linear_fwd(xn_attn, wv)
    q = [[row[h * head_dim : (h + 1) * head_dim] for row in q_flat] for h in range(N_HEAD)]
    k = [[row[h * head_dim : (h + 1) * head_dim] for row in k_flat] for h in range(N_HEAD)]
    v = [[row[h * head_dim : (h + 1) * head_dim] for row in v_flat] for h in range(N_HEAD)]

    inv_scale = 1.0 / head_dim**0.5
    attn_w = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]
    attn_out_flat = [[0.0] * N_EMBED for _ in range(n)]
    for h in range(N_HEAD):
        q_h, k_h, v_h = q[h], k[h], v[h]
        h_off = h * head_dim
        for i in range(n):
            q_h_i = q_h[i]
            qk_i = [sum(map(mul, q_h_i, k_h[j])) * inv_scale if j <= i else -1e10 for j in range(n)]
            max_val = max(qk_i)
            exps = [math.exp(v - max_val) for v in qk_i]
            total = sum(exps)
            aw_i = [e / total for e in exps]
            attn_w[h][i] = aw_i
            flat_i = attn_out_flat[i]
            for d in range(head_dim):
                val = 0.0
                for j in range(n):
                    val += aw_i[j] * v_h[j][d]
                flat_i[h_off + d] = val

    attn_proj = linear_fwd(attn_out_flat, wo)
    x = [[attn_proj[i][j] + x_pre_attn[i][j] for j in range(N_EMBED)] for i in range(n)]

    x_pre_mlp = x
    xn_mlp, rms_mlp = rmsnorm_fwd(x)

    h_pre = linear_fwd(xn_mlp, fc1)
    h_val = [[v if v > 0.0 else 0.0 for v in row] for row in h_pre]  # relu

    mlp_proj = linear_fwd(h_val, fc2)
    x = [[mlp_proj[i][j] + x_pre_mlp[i][j] for j in range(N_EMBED)] for i in range(n)]

    return x, LayerCache(x_pre_attn, xn_attn, rms_attn, q, k, v, attn_w, attn_out_flat, x_pre_mlp, xn_mlp, rms_mlp, h_pre, h_val)


def layer_bwd(dx: list[list[float]], params: dict, cache: LayerCache, li: int) -> tuple[list[list[float]], dict]:
    n = len(dx)
    head_dim = N_EMBED // N_HEAD
    wq = params[f"layer{li}.attn_wq"]
    wk = params[f"layer{li}.attn_wk"]
    wv = params[f"layer{li}.attn_wv"]
    wo = params[f"layer{li}.attn_wo"]
    fc1 = params[f"layer{li}.mlp_fc1"]
    fc2 = params[f"layer{li}.mlp_fc2"]

    inv_scale = 1.0 / head_dim**0.5

    dx_res_mlp = dx

    dfc2 = linear_bwd_w(dx, cache.h)
    dh_pre = linear_bwd_x(dx, fc2)
    mlp_dim = len(fc1)
    dh_pre = [[dh_pre[i][j] if cache.h_pre[i][j] > 0.0 else 0.0 for j in range(mlp_dim)] for i in range(n)]
    dfc1 = linear_bwd_w(dh_pre, cache.xn_mlp)
    dxn_mlp = linear_bwd_x(dh_pre, fc1)

    dx_rms = rmsnorm_bwd(dxn_mlp, cache.x_pre_mlp, cache.rms_mlp)
    dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rms, dx_res_mlp)]

    dx_res_attn = dx

    dwo = linear_bwd_w(dx, cache.attn_out_flat)
    dattn_out_flat = linear_bwd_x(dx, wo)

    attn_w_cache = cache.attn_w
    v_cache = cache.v
    dv = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
    dattn_w = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]
    for h in range(N_HEAD):
        aw_h, v_h = attn_w_cache[h], v_cache[h]
        h_off = h * head_dim
        dv_h, daw_h = dv[h], dattn_w[h]
        for i in range(n):
            dv_h_i = dv_h[i]
            for d in range(head_dim):
                val = 0.0
                for j in range(n):
                    val += aw_h[j][i] * dattn_out_flat[j][h_off + d]
                dv_h_i[d] = val
        for i in range(n):
            dat_i, daw_h_i = dattn_out_flat[i], daw_h[i]
            for j in range(n):
                val = 0.0
                for d in range(head_dim):
                    val += dat_i[h_off + d] * v_h[j][d]
                daw_h_i[j] = val

    dlogits_attn = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]
    for h in range(N_HEAD):
        aw_h, daw_h, dla_h = attn_w_cache[h], dattn_w[h], dlogits_attn[h]
        for i in range(n):
            aw_i, daw_i = aw_h[i], daw_h[i]
            s = sum(map(mul, daw_i, aw_i))
            dla_i = dla_h[i]
            for j in range(n):
                dla_i[j] = aw_i[j] * (daw_i[j] - s) * inv_scale

    k_cache = cache.k
    q_cache = cache.q
    dq = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
    dk = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
    for h in range(N_HEAD):
        dla_h = dlogits_attn[h]
        k_h, q_h = k_cache[h], q_cache[h]
        dq_h, dk_h = dq[h], dk[h]
        for i in range(n):
            dla_i, dq_i, dk_i = dla_h[i], dq_h[i], dk_h[i]
            for d in range(head_dim):
                val1, val2 = 0.0, 0.0
                for j in range(n):
                    val1 += dla_i[j] * k_h[j][d]
                    val2 += dla_h[j][i] * q_h[j][d]
                dq_i[d] = val1
                dk_i[d] = val2

    dq_flat = [[v for h in range(N_HEAD) for v in dq[h][i]] for i in range(n)]
    dk_flat = [[v for h in range(N_HEAD) for v in dk[h][i]] for i in range(n)]
    dv_flat = [[v for h in range(N_HEAD) for v in dv[h][i]] for i in range(n)]

    dwq = linear_bwd_w(dq_flat, cache.xn_attn)
    dwk = linear_bwd_w(dk_flat, cache.xn_attn)
    dwv = linear_bwd_w(dv_flat, cache.xn_attn)

    WqT = list(zip(*wq))
    WkT = list(zip(*wk))
    WvT = list(zip(*wv))
    dxn_attn = [[sum(map(mul, dq_flat[i], WqT[d])) + sum(map(mul, dk_flat[i], WkT[d])) + sum(map(mul, dv_flat[i], WvT[d])) for d in range(N_EMBED)] for i in range(n)]

    dx_rms = rmsnorm_bwd(dxn_attn, cache.x_pre_attn, cache.rms_attn)
    dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rms, dx_res_attn)]

    grads = {
        f"layer{li}.attn_wq": dwq,
        f"layer{li}.attn_wk": dwk,
        f"layer{li}.attn_wv": dwv,
        f"layer{li}.attn_wo": dwo,
        f"layer{li}.mlp_fc1": dfc1,
        f"layer{li}.mlp_fc2": dfc2,
    }
    return dx, grads


def forward(params: dict, input_ids: list[int], target_ids: list[int], loss_mask: list[float]) -> tuple[float, FwdCache]:
    n = len(input_ids)

    wte, wpe = params["wte"], params["wpe"]
    emb = [[0.0] * N_EMBED for _ in range(n)]
    for i in range(n):
        wte_i, wpe_i, emb_i = wte[input_ids[i]], wpe[i], emb[i]
        for j in range(N_EMBED):
            emb_i[j] = wte_i[j] + wpe_i[j]
    x, rms_init = rmsnorm_fwd(emb)

    layer_caches = []
    for li in range(N_LAYER):
        x, lc = layer_fwd(x, params, li)
        layer_caches.append(lc)

    lm_head = params["lm_head"]
    logits = linear_fwd(x, lm_head)  # (n, vocab_size)
    probs = []
    for i in range(n):
        logits_i = logits[i]
        max_val = max(logits_i)
        exps = [math.exp(v - max_val) for v in logits_i]
        total = sum(exps)
        probs.append([e / total for e in exps])

    sum_mask = sum(loss_mask) or 1.0
    loss = -sum(math.log(probs[i][target_ids[i]]) * loss_mask[i] for i in range(n)) / sum_mask

    return loss, FwdCache(input_ids, target_ids, loss_mask, sum_mask, emb, rms_init, x, probs, layer_caches)


def backward(params: dict, cache: FwdCache) -> dict:
    n = len(cache.input_ids)
    inv_sum_mask = 1.0 / cache.sum_mask
    probs, target_ids, loss_mask, x = cache.probs, cache.target_ids, cache.loss_mask, cache.x

    lm_head = params["lm_head"]
    dlogits = [[(p * inv_sum_mask) * loss_mask[i] for p in probs[i]] for i in range(n)]
    for i in range(n):
        dlogits[i][target_ids[i]] -= inv_sum_mask * loss_mask[i]

    dlm_head = linear_bwd_w(dlogits, x)  # (vocab_size, N_EMBED)
    dx = linear_bwd_x(dlogits, lm_head)  # (n, N_EMBED)

    grads = {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in params.items()}
    grads["lm_head"] = dlm_head

    for li in reversed(range(N_LAYER)):
        dx, layer_grads = layer_bwd(dx, params, cache.layer_caches[li], li)
        grads.update(layer_grads)

    demb = rmsnorm_bwd(dx, cache.emb, cache.rms_init)
    gwte, gwpe = grads["wte"], grads["wpe"]
    for i in range(n):
        tid, demb_i = cache.input_ids[i], demb[i]
        gwte_i, gwpe_i = gwte[tid], gwpe[i]
        for j in range(N_EMBED):
            d = demb_i[j]
            gwte_i[j] += d
            gwpe_i[j] += d

    return grads


def step_fn(params: dict, opt_state: dict, input_ids: list[int], target_ids: list[int], loss_mask: list[float], step: int) -> tuple[float, dict, dict]:
    loss, cache = forward(params, input_ids, target_ids, loss_mask)
    grads = backward(params, cache)

    lr = 0.01 * (1 - step / NUM_STEPS)
    beta1, beta2, eps = 0.85, 0.99, 1e-8
    inv_bias1 = 1.0 / (1 - beta1 ** (step + 1))
    inv_bias2 = 1.0 / (1 - beta2 ** (step + 1))
    lr_scaled = lr * inv_bias1
    one_minus_b1, one_minus_b2 = 1 - beta1, 1 - beta2

    m, v = opt_state["m"], opt_state["v"]

    for k in params:
        p_mat, g_mat, m_mat, v_mat = params[k], grads[k], m[k], v[k]
        nrow, ncol = len(p_mat), len(p_mat[0])
        for i in range(nrow):
            p_i, g_i, m_i, v_i = p_mat[i], g_mat[i], m_mat[i], v_mat[i]
            for j in range(ncol):
                g = g_i[j]
                m_ij = beta1 * m_i[j] + one_minus_b1 * g
                v_ij = beta2 * v_i[j] + one_minus_b2 * g * g
                p_i[j] -= lr_scaled * m_ij / ((v_ij * inv_bias2) ** 0.5 + eps)
                m_i[j] = m_ij
                v_i[j] = v_ij

    return loss, params, opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[list[int], list[int], list[float]]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)
    input_ids = [0] * BLOCK_SIZE
    target_ids = [0] * BLOCK_SIZE
    loss_mask = [0.0] * BLOCK_SIZE
    for i in range(n):
        input_ids[i] = tokens[i]
        target_ids[i] = tokens[i + 1]
        loss_mask[i] = 1.0
    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: [[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    "wte": matrix(len(uchars) + 1, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(len(uchars) + 1, N_EMBED),
    **{f"layer{i}.attn_wq": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

opt_state = {
    "m": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()},
    "v": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()},
}

tokenized = [tokenize(doc, uchars) for doc in docs]

step_times = []
for step in tqdm(range(NUM_STEPS)):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
