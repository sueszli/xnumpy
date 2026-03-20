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
from utils.times import save_times
from utils.weights import assert_weights_match

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000

AttnCache = namedtuple("AttnCache", ["x_pre", "xn", "rms", "q", "k", "v", "attn_w", "out_flat"])
MlpCache = namedtuple("MlpCache", ["x_pre", "xn", "rms", "h_pre", "h"])
FwdCache = namedtuple("FwdCache", ["input_ids", "target_ids", "loss_mask", "sum_mask", "emb", "rms_init", "x", "probs", "layer_caches"])


#
# kernels
#


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


def softmax(v: list[float]) -> list[float]:
    max_val = max(v)
    exps = [math.exp(x - max_val) for x in v]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_bwd(aw: list[list[float]], daw: list[list[float]], scale: float) -> list[list[float]]:
    return [[(a * (d - sum(map(mul, aw_i, daw_i)))) * scale for a, d in zip(aw_i, daw_i)] for aw_i, daw_i in zip(aw, daw)]


def madd(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def merge_heads(heads: list) -> list[list[float]]:
    return [[v for head in heads for v in head[i]] for i in range(len(heads[0]))]


#
# model
#


def rmsnorm_fwd(x: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    inv_d = 1.0 / len(x[0])
    out, rms = [], []
    for row in x:
        scale = (sum(map(mul, row, row)) * inv_d + 1e-5) ** -0.5
        rms.append(scale)
        out.append([v * scale for v in row])
    return out, rms


def rmsnorm_bwd(dout: list[list[float]], x: list[list[float]], rms: list[float]) -> list[list[float]]:
    inv_d = 1.0 / len(x[0])
    dx = []
    for do, row, scale in zip(dout, x, rms):
        scale3_over_d = (scale**3) * inv_d
        dot = sum(map(mul, do, row))
        dx.append([do_j * scale - scale3_over_d * row_j * dot for do_j, row_j in zip(do, row)])
    return dx


def attn_fwd(x: list[list[float]], wq, wk, wv, wo) -> tuple[list[list[float]], AttnCache]:
    n = len(x)
    head_dim = N_EMBED // N_HEAD
    xn, rms = rmsnorm_fwd(x)

    q_flat = linear_fwd(xn, wq)
    k_flat = linear_fwd(xn, wk)
    v_flat = linear_fwd(xn, wv)
    q = [[row[h * head_dim : (h + 1) * head_dim] for row in q_flat] for h in range(N_HEAD)]
    k = [[row[h * head_dim : (h + 1) * head_dim] for row in k_flat] for h in range(N_HEAD)]
    v = [[row[h * head_dim : (h + 1) * head_dim] for row in v_flat] for h in range(N_HEAD)]

    inv_scale = 1.0 / head_dim**0.5
    attn_w = [None] * N_HEAD
    out_flat = [[0.0] * N_EMBED for _ in range(n)]
    for h in range(N_HEAD):
        q_h, k_h, v_h = q[h], k[h], v[h]
        h_off = h * head_dim
        vhT = list(zip(*v_h))
        aw_h = [None] * n
        for i in range(n):
            q_h_i = q_h[i]
            qk_i = [sum(map(mul, q_h_i, k_h[j])) * inv_scale if j <= i else -1e10 for j in range(n)]
            aw_i = softmax(qk_i)
            aw_h[i] = aw_i
            out_flat[i][h_off : h_off + head_dim] = [sum(map(mul, aw_i, vd)) for vd in vhT]
        attn_w[h] = aw_h

    return madd(linear_fwd(out_flat, wo), x), AttnCache(x, xn, rms, q, k, v, attn_w, out_flat)


def attn_bwd(dx: list[list[float]], wq, wk, wv, wo, c: AttnCache, li: int) -> tuple[list[list[float]], dict]:
    head_dim = N_EMBED // N_HEAD
    inv_scale = 1.0 / head_dim**0.5
    n = len(dx)

    dx_res = dx
    dwo = linear_bwd_w(dx, c.out_flat)
    dattn_out_flat = linear_bwd_x(dx, wo)

    dv = [None] * N_HEAD
    dattn_w = [None] * N_HEAD
    for h in range(N_HEAD):
        h_off = h * head_dim
        dat_h = [row[h_off : h_off + head_dim] for row in dattn_out_flat]
        dv[h] = linear_bwd_w(c.attn_w[h], dat_h)
        dattn_w[h] = linear_fwd(dat_h, c.v[h])

    dlogits_attn = [softmax_bwd(c.attn_w[h], dattn_w[h], inv_scale) for h in range(N_HEAD)]

    dq = [None] * N_HEAD
    dk = [None] * N_HEAD
    for h in range(N_HEAD):
        dla_h = dlogits_attn[h]
        dq[h] = linear_fwd(dla_h, list(zip(*c.k[h])))
        dk[h] = linear_bwd_w(dla_h, c.q[h])

    dq_flat = merge_heads(dq)
    dk_flat = merge_heads(dk)
    dv_flat = merge_heads(dv)

    WqT = list(zip(*wq))
    WkT = list(zip(*wk))
    WvT = list(zip(*wv))
    dxn = [[sum(map(mul, dq_flat[i], WqT[d])) + sum(map(mul, dk_flat[i], WkT[d])) + sum(map(mul, dv_flat[i], WvT[d])) for d in range(N_EMBED)] for i in range(n)]

    grads = {
        f"layer{li}.attn_wq": linear_bwd_w(dq_flat, c.xn),
        f"layer{li}.attn_wk": linear_bwd_w(dk_flat, c.xn),
        f"layer{li}.attn_wv": linear_bwd_w(dv_flat, c.xn),
        f"layer{li}.attn_wo": dwo,
    }
    return madd(rmsnorm_bwd(dxn, c.x_pre, c.rms), dx_res), grads


def mlp_fwd(x: list[list[float]], fc1, fc2) -> tuple[list[list[float]], MlpCache]:
    xn, rms = rmsnorm_fwd(x)
    h_pre = linear_fwd(xn, fc1)
    h = [[v if v > 0.0 else 0.0 for v in row] for row in h_pre]
    return madd(linear_fwd(h, fc2), x), MlpCache(x, xn, rms, h_pre, h)


def mlp_bwd(dx: list[list[float]], fc1, fc2, c: MlpCache, li: int) -> tuple[list[list[float]], dict]:
    dx_res = dx
    dfc2 = linear_bwd_w(dx, c.h)
    dh_pre_raw = linear_bwd_x(dx, fc2)
    dh_pre = [[g if p > 0.0 else 0.0 for p, g in zip(hp_i, raw_i)] for hp_i, raw_i in zip(c.h_pre, dh_pre_raw)]
    dfc1 = linear_bwd_w(dh_pre, c.xn)
    dxn_mlp = linear_bwd_x(dh_pre, fc1)
    grads = {
        f"layer{li}.mlp_fc1": dfc1,
        f"layer{li}.mlp_fc2": dfc2,
    }
    return madd(rmsnorm_bwd(dxn_mlp, c.x_pre, c.rms), dx_res), grads


def forward(params: dict, input_ids: list[int], target_ids: list[int], loss_mask: list[float]) -> tuple[float, FwdCache]:
    n = len(input_ids)
    emb = madd([params["wte"][tid] for tid in input_ids], params["wpe"])
    x, rms_init = rmsnorm_fwd(emb)

    layer_caches = []
    for li in range(N_LAYER):
        x, ac = attn_fwd(x, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"])
        x, mc = mlp_fwd(x, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"])
        layer_caches.append((ac, mc))

    logits = linear_fwd(x, params["lm_head"])
    probs = [softmax(row) for row in logits]
    sum_mask = sum(loss_mask) or 1.0
    loss = -sum(math.log(probs[i][target_ids[i]]) * loss_mask[i] for i in range(n)) / sum_mask

    return loss, FwdCache(input_ids, target_ids, loss_mask, sum_mask, emb, rms_init, x, probs, layer_caches)


def backward(params: dict, cache: FwdCache) -> dict:
    n = len(cache.input_ids)
    inv_sum_mask = 1.0 / cache.sum_mask
    probs, target_ids, loss_mask, x = cache.probs, cache.target_ids, cache.loss_mask, cache.x

    dlogits = [[(p * inv_sum_mask) * loss_mask[i] for p in probs[i]] for i in range(n)]
    for i in range(n):
        dlogits[i][target_ids[i]] -= inv_sum_mask * loss_mask[i]

    grads = {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in params.items()}
    grads["lm_head"] = linear_bwd_w(dlogits, x)
    dx = linear_bwd_x(dlogits, params["lm_head"])

    for li in reversed(range(N_LAYER)):
        ac, mc = cache.layer_caches[li]
        dx, mlp_grads = mlp_bwd(dx, params[f"layer{li}.mlp_fc1"], params[f"layer{li}.mlp_fc2"], mc, li)
        dx, attn_grads = attn_bwd(dx, params[f"layer{li}.attn_wq"], params[f"layer{li}.attn_wk"], params[f"layer{li}.attn_wv"], params[f"layer{li}.attn_wo"], ac, li)
        grads.update(mlp_grads)
        grads.update(attn_grads)

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
