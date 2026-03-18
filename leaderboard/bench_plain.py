# /// script
# requires-python = "==3.14.*"
# dependencies = ["tqdm"]
# ///

import math
import random
import time
from collections import namedtuple
from pathlib import Path

from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


def rmsnorm_fwd(x_2d):
    n = len(x_2d)
    d = len(x_2d[0])
    out = [[0.0] * d for _ in range(n)]
    rms = [0.0] * n
    for i in range(n):
        x = x_2d[i]
        ms = sum(xi * xi for xi in x) / d
        scale = (ms + 1e-5) ** -0.5
        rms[i] = scale
        for j in range(d):
            out[i][j] = x[j] * scale
    return out, rms


def rmsnorm_bwd(dout_2d, x_2d, rms):
    n = len(x_2d)
    d = len(x_2d[0])
    dx = [[0.0] * d for _ in range(n)]
    for i in range(n):
        do = dout_2d[i]
        x = x_2d[i]
        scale = rms[i]
        dot = sum(doi * xi for doi, xi in zip(do, x))
        for j in range(d):
            dx[i][j] = do[j] * scale - (scale**3 / d) * x[j] * dot
    return dx


def transpose_2d(A):
    return [list(col) for col in zip(*A)]


def forward_backward(params, input_ids, target_ids, loss_mask):
    n = len(input_ids)
    grads = {k: [[0.0 for _ in row] for row in mat] for k, mat in params.items()}
    head_dim = N_EMBED // N_HEAD
    vocab_size = len(params["lm_head"])

    emb = [[0.0] * N_EMBED for _ in range(n)]
    for i in range(n):
        for j in range(N_EMBED):
            emb[i][j] = params["wte"][input_ids[i]][j] + params["wpe"][i][j]

    x, rms_init = rmsnorm_fwd(emb)

    layer_cache = []
    for li in range(N_LAYER):
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        x_pre_attn = x
        xn_attn, rms_attn = rmsnorm_fwd(x)

        q = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        k = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        v = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        for i in range(n):
            for j in range(N_EMBED):
                val_q = sum(xn_attn[i][d] * wq[j][d] for d in range(N_EMBED))
                val_k = sum(xn_attn[i][d] * wk[j][d] for d in range(N_EMBED))
                val_v = sum(xn_attn[i][d] * wv[j][d] for d in range(N_EMBED))
                h = j // head_dim
                d = j % head_dim
                q[h][i][d] = val_q
                k[h][i][d] = val_k
                v[h][i][d] = val_v

        attn_w = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]
        attn_out = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        attn_out_flat = [[0.0] * N_EMBED for _ in range(n)]
        scale = head_dim**0.5

        for h in range(N_HEAD):
            q_h, k_h, v_h = q[h], k[h], v[h]
            for i in range(n):
                qk_i = [0.0] * n
                for j in range(i + 1):
                    qk_i[j] = sum(q_h[i][d] * k_h[j][d] for d in range(head_dim)) / scale
                for j in range(i + 1, n):
                    qk_i[j] = -1e10

                max_val = max(qk_i)
                exps = [math.exp(val - max_val) for val in qk_i]
                total = sum(exps)
                aw_i = [e / total for e in exps]
                attn_w[h][i] = aw_i

                for d in range(head_dim):
                    val = sum(aw_i[j] * v_h[j][d] for j in range(n))
                    attn_out[h][i][d] = val
                    attn_out_flat[i][h * head_dim + d] = val

        x = [[0.0] * N_EMBED for _ in range(n)]
        for i in range(n):
            for j in range(N_EMBED):
                x[i][j] = sum(attn_out_flat[i][d] * wo[j][d] for d in range(N_EMBED)) + x_pre_attn[i][j]

        x_pre_mlp = x
        xn_mlp, rms_mlp = rmsnorm_fwd(x)

        h_pre = [[0.0] * (4 * N_EMBED) for _ in range(n)]
        h_val = [[0.0] * (4 * N_EMBED) for _ in range(n)]
        for i in range(n):
            for j in range(4 * N_EMBED):
                val = sum(xn_mlp[i][d] * fc1[j][d] for d in range(N_EMBED))
                h_pre[i][j] = val
                h_val[i][j] = val if val > 0.0 else 0.0

        x = [[0.0] * N_EMBED for _ in range(n)]
        for i in range(n):
            for j in range(N_EMBED):
                x[i][j] = sum(h_val[i][d] * fc2[j][d] for d in range(4 * N_EMBED)) + x_pre_mlp[i][j]

        layer_cache.append({"x_pre_attn": x_pre_attn, "xn_attn": xn_attn, "rms_attn": rms_attn, "q": q, "k": k, "v": v, "attn_w": attn_w, "attn_out_flat": attn_out_flat, "x_pre_mlp": x_pre_mlp, "xn_mlp": xn_mlp, "rms_mlp": rms_mlp, "h_pre": h_pre, "h": h_val})

    logits = [[0.0] * vocab_size for _ in range(n)]
    probs = [[0.0] * vocab_size for _ in range(n)]
    for i in range(n):
        for j in range(vocab_size):
            logits[i][j] = sum(x[i][d] * params["lm_head"][j][d] for d in range(N_EMBED))
        max_val = max(logits[i])
        exps = [math.exp(val - max_val) for val in logits[i]]
        total = sum(exps)
        probs[i] = [e / total for e in exps]

    sum_mask = sum(loss_mask)
    if sum_mask == 0:
        sum_mask = 1.0

    loss = -sum(math.log(probs[i][target_ids[i]]) * loss_mask[i] for i in range(n)) / sum_mask

    dlogits = [[(p / sum_mask) * loss_mask[i] for p in p_row] for i, p_row in enumerate(probs)]
    for i in range(n):
        dlogits[i][target_ids[i]] -= (1.0 / sum_mask) * loss_mask[i]

    dlm_head = [[0.0] * N_EMBED for _ in range(vocab_size)]
    for j in range(vocab_size):
        for d in range(N_EMBED):
            dlm_head[j][d] = sum(dlogits[i][j] * x[i][d] for i in range(n))
    grads["lm_head"] = dlm_head

    dx = [[0.0] * N_EMBED for _ in range(n)]
    for i in range(n):
        for d in range(N_EMBED):
            dx[i][d] = sum(dlogits[i][j] * params["lm_head"][j][d] for j in range(vocab_size))

    for li in reversed(range(N_LAYER)):
        cache = layer_cache[li]
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        dx_res_mlp = dx
        dfc2 = [[0.0] * (4 * N_EMBED) for _ in range(N_EMBED)]
        for j in range(N_EMBED):
            for d in range(4 * N_EMBED):
                dfc2[j][d] = sum(dx[i][j] * cache["h"][i][d] for i in range(n))
        grads[f"layer{li}.mlp_fc2"] = dfc2

        dh = [[0.0] * (4 * N_EMBED) for _ in range(n)]
        dh_pre = [[0.0] * (4 * N_EMBED) for _ in range(n)]
        for i in range(n):
            for j in range(4 * N_EMBED):
                val = sum(dx[i][d] * fc2[d][j] for d in range(N_EMBED))
                dh[i][j] = val
                dh_pre[i][j] = val if cache["h_pre"][i][j] > 0.0 else 0.0

        dfc1 = [[0.0] * N_EMBED for _ in range(4 * N_EMBED)]
        for j in range(4 * N_EMBED):
            for d in range(N_EMBED):
                dfc1[j][d] = sum(dh_pre[i][j] * cache["xn_mlp"][i][d] for i in range(n))
        grads[f"layer{li}.mlp_fc1"] = dfc1

        dxn_mlp = [[0.0] * N_EMBED for _ in range(n)]
        for i in range(n):
            for d in range(N_EMBED):
                dxn_mlp[i][d] = sum(dh_pre[i][j] * fc1[j][d] for j in range(4 * N_EMBED))

        dx_rmsnorm = rmsnorm_bwd(dxn_mlp, cache["x_pre_mlp"], cache["rms_mlp"])
        dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rmsnorm, dx_res_mlp)]

        dx_res_attn = dx
        dwo = [[0.0] * N_EMBED for _ in range(N_EMBED)]
        for j in range(N_EMBED):
            for d in range(N_EMBED):
                dwo[j][d] = sum(dx[i][j] * cache["attn_out_flat"][i][d] for i in range(n))
        grads[f"layer{li}.attn_wo"] = dwo

        dattn_out_flat = [[0.0] * N_EMBED for _ in range(n)]
        for i in range(n):
            for d in range(N_EMBED):
                dattn_out_flat[i][d] = sum(dx[i][j] * wo[j][d] for j in range(N_EMBED))

        dv = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        dattn_w = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]

        for h in range(N_HEAD):
            aw_h = cache["attn_w"][h]
            aw_h_T = transpose_2d(aw_h)
            v_h = cache["v"][h]
            for i in range(n):
                for d in range(head_dim):
                    dv[h][i][d] = sum(aw_h_T[i][j] * dattn_out_flat[j][h * head_dim + d] for j in range(n))

            for i in range(n):
                for j in range(n):
                    dattn_w[h][i][j] = sum(dattn_out_flat[i][h * head_dim + d] * v_h[j][d] for d in range(head_dim))

        dlogits_attn = [[[0.0] * n for _ in range(n)] for _ in range(N_HEAD)]
        for h in range(N_HEAD):
            aw_h = cache["attn_w"][h]
            d_aw_h = dattn_w[h]
            for i in range(n):
                sum_d_aw = sum(d_aw_h[i][j] * aw_h[i][j] for j in range(n))
                for j in range(n):
                    dlogits_attn[h][i][j] = aw_h[i][j] * (d_aw_h[i][j] - sum_d_aw) / scale

        dq = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        dk = [[[0.0] * head_dim for _ in range(n)] for _ in range(N_HEAD)]
        for h in range(N_HEAD):
            dl_h = dlogits_attn[h]
            dl_h_T = transpose_2d(dl_h)
            k_h = cache["k"][h]
            q_h = cache["q"][h]
            for i in range(n):
                for d in range(head_dim):
                    val_q = sum(dl_h[i][j] * k_h[j][d] for j in range(n))
                    val_k = sum(dl_h_T[i][j] * q_h[j][d] for j in range(n))
                    dq[h][i][d] = val_q
                    dk[h][i][d] = val_k

        dwq = [[0.0] * N_EMBED for _ in range(N_EMBED)]
        dwk = [[0.0] * N_EMBED for _ in range(N_EMBED)]
        dwv = [[0.0] * N_EMBED for _ in range(N_EMBED)]
        for j in range(N_EMBED):
            h = j // head_dim
            d = j % head_dim
            for k_dim in range(N_EMBED):
                dwq[j][k_dim] = sum(dq[h][i][d] * cache["xn_attn"][i][k_dim] for i in range(n))
                dwk[j][k_dim] = sum(dk[h][i][d] * cache["xn_attn"][i][k_dim] for i in range(n))
                dwv[j][k_dim] = sum(dv[h][i][d] * cache["xn_attn"][i][k_dim] for i in range(n))

        grads[f"layer{li}.attn_wq"] = dwq
        grads[f"layer{li}.attn_wk"] = dwk
        grads[f"layer{li}.attn_wv"] = dwv

        dxn_attn = [[0.0] * N_EMBED for _ in range(n)]
        for i in range(n):
            for k_dim in range(N_EMBED):
                val = 0.0
                for j in range(N_EMBED):
                    h = j // head_dim
                    d = j % head_dim
                    val += dq[h][i][d] * wq[j][k_dim] + dk[h][i][d] * wk[j][k_dim] + dv[h][i][d] * wv[j][k_dim]
                dxn_attn[i][k_dim] = val

        dx_rmsnorm = rmsnorm_bwd(dxn_attn, cache["x_pre_attn"], cache["rms_attn"])
        dx = [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(dx_rmsnorm, dx_res_attn)]

    demb = rmsnorm_bwd(dx, emb, rms_init)
    for i in range(n):
        tid = input_ids[i]
        for j in range(N_EMBED):
            grads["wte"][tid][j] += demb[i][j]
            grads["wpe"][i][j] += demb[i][j]

    return loss, grads


def step_fn(params, opt_state, input_ids, target_ids, loss_mask, step):
    loss, grads = forward_backward(params, input_ids, target_ids, loss_mask)

    learning_rate = 0.01
    beta1 = 0.85
    beta2 = 0.99
    eps_adam = 1e-8
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    m = opt_state["m"]
    v = opt_state["v"]

    new_params = {}
    new_m = {}
    new_v = {}

    for k in params:
        p_mat = params[k]
        g_mat = grads[k]
        m_mat = m[k]
        v_mat = v[k]

        nrow = len(p_mat)
        ncol = len(p_mat[0])

        new_p = [[0.0] * ncol for _ in range(nrow)]
        new_m_mat = [[0.0] * ncol for _ in range(nrow)]
        new_v_mat = [[0.0] * ncol for _ in range(nrow)]

        for i in range(nrow):
            for j in range(ncol):
                g = g_mat[i][j]
                m_ij = beta1 * m_mat[i][j] + (1 - beta1) * g
                v_ij = beta2 * v_mat[i][j] + (1 - beta2) * g**2
                m_hat = m_ij / (1 - beta1 ** (step + 1))
                v_hat = v_ij / (1 - beta2 ** (step + 1))
                new_p[i][j] = p_mat[i][j] - lr_t * m_hat / (v_hat**0.5 + eps_adam)
                new_m_mat[i][j] = m_ij
                new_v_mat[i][j] = v_ij

        new_params[k] = new_p
        new_m[k] = new_m_mat
        new_v[k] = new_v_mat

    return loss, new_params, {"m": new_m, "v": new_v}


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

opt_state = {"m": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()}, "v": {k: [[0.0] * len(mat[0]) for _ in mat] for k, mat in state_dict.items()}}

tokenized = [tokenize(doc, uchars) for doc in tqdm(docs, desc="tokenizing")]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {loss:.4f}", end="\r")

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
