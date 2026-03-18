# /// script
# requires-python = "==3.14.*"
# dependencies = ["numpy"]
# ///

import random
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000

LayerCache = namedtuple("LayerCache", ["x_pre_attn", "xn_attn", "rms_attn", "q", "k", "v", "attn_w", "attn_out_flat", "x_pre_mlp", "xn_mlp", "rms_mlp", "h_pre", "h"])
FwdCache = namedtuple("FwdCache", ["input_ids", "target_ids", "loss_mask", "sum_mask", "emb", "rms_init", "x", "probs", "layer_caches"])


def rmsnorm_fwd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rms = (np.mean(x * x, axis=-1, keepdims=True) + 1e-5) ** -0.5
    return x * rms, rms


def rmsnorm_bwd(dout: np.ndarray, x: np.ndarray, rms: np.ndarray) -> np.ndarray:
    dot = (dout * x).sum(axis=-1, keepdims=True)
    return dout * rms - (rms**3 / N_EMBED) * x * dot


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    e /= e.sum(axis=axis, keepdims=True)
    return e


def layer_fwd(x: np.ndarray, params: dict, li: int) -> tuple[np.ndarray, LayerCache]:
    x_pre_attn = x
    xn_attn, rms_attn = rmsnorm_fwd(x)
    q = (xn_attn @ params[f"layer{li}.attn_wq"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
    k = (xn_attn @ params[f"layer{li}.attn_wk"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
    v = (xn_attn @ params[f"layer{li}.attn_wv"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
    attn_w = softmax(q @ k.transpose(0, 2, 1) * 0.5 + np.triu(np.full((BLOCK_SIZE, BLOCK_SIZE), -1e10), 1))
    attn_out_flat = (attn_w @ v).transpose(1, 0, 2).reshape(BLOCK_SIZE, N_EMBED)
    x = attn_out_flat @ params[f"layer{li}.attn_wo"].T + x_pre_attn

    x_pre_mlp = x
    xn_mlp, rms_mlp = rmsnorm_fwd(x)
    h_pre = xn_mlp @ params[f"layer{li}.mlp_fc1"].T
    h = np.maximum(0.0, h_pre)
    x = h @ params[f"layer{li}.mlp_fc2"].T + x_pre_mlp

    return x, LayerCache(x_pre_attn, xn_attn, rms_attn, q, k, v, attn_w, attn_out_flat, x_pre_mlp, xn_mlp, rms_mlp, h_pre, h)


def layer_bwd(dx: np.ndarray, grads: dict, params: dict, c: LayerCache, li: int) -> np.ndarray:
    dx_res = dx
    np.matmul(dx.T, c.h, out=grads[f"layer{li}.mlp_fc2"])
    dh_pre = (dx @ params[f"layer{li}.mlp_fc2"]) * (c.h_pre > 0)
    np.matmul(dh_pre.T, c.xn_mlp, out=grads[f"layer{li}.mlp_fc1"])
    dx = rmsnorm_bwd(dh_pre @ params[f"layer{li}.mlp_fc1"], c.x_pre_mlp, c.rms_mlp) + dx_res

    dx_res = dx
    np.matmul(dx.T, c.attn_out_flat, out=grads[f"layer{li}.attn_wo"])
    dattn_out = (dx @ params[f"layer{li}.attn_wo"]).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
    dv = c.attn_w.transpose(0, 2, 1) @ dattn_out
    dattn_w = dattn_out @ c.v.transpose(0, 2, 1)
    dlogits_attn = c.attn_w * (dattn_w - (dattn_w * c.attn_w).sum(-1, keepdims=True)) * 0.5
    dq = dlogits_attn @ c.k
    dk = dlogits_attn.transpose(0, 2, 1) @ c.q
    dq_flat = dq.transpose(1, 0, 2).reshape(BLOCK_SIZE, N_EMBED)
    dk_flat = dk.transpose(1, 0, 2).reshape(BLOCK_SIZE, N_EMBED)
    dv_flat = dv.transpose(1, 0, 2).reshape(BLOCK_SIZE, N_EMBED)
    np.matmul(dq_flat.T, c.xn_attn, out=grads[f"layer{li}.attn_wq"])
    np.matmul(dk_flat.T, c.xn_attn, out=grads[f"layer{li}.attn_wk"])
    np.matmul(dv_flat.T, c.xn_attn, out=grads[f"layer{li}.attn_wv"])
    dxn_attn = dq_flat @ params[f"layer{li}.attn_wq"] + dk_flat @ params[f"layer{li}.attn_wk"] + dv_flat @ params[f"layer{li}.attn_wv"]
    return rmsnorm_bwd(dxn_attn, c.x_pre_attn, c.rms_attn) + dx_res


def forward(params: dict, input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray) -> tuple[float, FwdCache]:
    emb = params["wte"][input_ids] + params["wpe"][np.arange(BLOCK_SIZE)]
    x, rms_init = rmsnorm_fwd(emb)

    layer_caches = []
    for li in range(N_LAYER):
        x, lc = layer_fwd(x, params, li)
        layer_caches.append(lc)

    probs = softmax(x @ params["lm_head"].T)
    sum_mask = loss_mask.sum()
    loss = -(np.log(probs[np.arange(BLOCK_SIZE), target_ids]) * loss_mask).sum() / sum_mask

    return loss, FwdCache(input_ids, target_ids, loss_mask, sum_mask, emb, rms_init, x, probs, layer_caches)


def backward(params: dict, grads: dict, cache: FwdCache) -> None:
    grads["wte"][:] = 0
    grads["wpe"][:] = 0

    dlogits = cache.probs / cache.sum_mask
    dlogits[np.arange(BLOCK_SIZE), cache.target_ids] -= 1.0 / cache.sum_mask
    dlogits *= cache.loss_mask[:, None]

    np.matmul(dlogits.T, cache.x, out=grads["lm_head"])
    dx = dlogits @ params["lm_head"]

    for li in reversed(range(N_LAYER)):
        dx = layer_bwd(dx, grads, params, cache.layer_caches[li], li)

    demb = rmsnorm_bwd(dx, cache.emb, cache.rms_init)
    np.add.at(grads["wte"], cache.input_ids, demb)
    grads["wpe"] += demb


def step_fn(params: dict, opt_state: dict, grads: dict, input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray, step: int) -> tuple[float, dict, dict]:
    loss, cache = forward(params, input_ids, target_ids, loss_mask)
    backward(params, grads, cache)

    lr_t = 0.01 * (1 - step / NUM_STEPS)
    bc1 = 1.0 - 0.85 ** (step + 1)
    bc2 = 1.0 - 0.99 ** (step + 1)

    flat_m = opt_state["flat_m"]
    flat_v = opt_state["flat_v"]
    flat_params = opt_state["flat_params"]
    buf = opt_state["buf"]
    tmp = opt_state["tmp"]

    offset = 0
    for k in PARAM_KEYS:
        g = grads[k]
        s = g.size
        buf[offset : offset + s] = g.ravel()
        offset += s

    np.multiply(buf, 0.15, out=tmp)
    flat_m *= 0.85
    flat_m += tmp
    np.multiply(buf, buf, out=tmp)
    tmp *= 0.01
    flat_v *= 0.99
    flat_v += tmp

    np.divide(flat_m, bc1, out=tmp)
    np.divide(flat_v, bc2, out=buf)
    np.sqrt(buf, out=buf)
    buf += 1e-8
    np.divide(tmp, buf, out=tmp)
    tmp *= lr_t
    flat_params -= tmp

    return loss, params, opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2i = {ch: i for i, ch in enumerate(uchars)}
    bos = len(uchars)
    tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
    n = min(BLOCK_SIZE, len(tokens) - 1)

    input_ids = np.zeros(BLOCK_SIZE, dtype=np.int32)
    target_ids = np.zeros(BLOCK_SIZE, dtype=np.int32)
    loss_mask = np.zeros(BLOCK_SIZE, dtype=np.float32)

    input_ids[:n] = tokens[:n]
    target_ids[:n] = tokens[1 : n + 1]
    loss_mask[:n] = 1.0

    return input_ids, target_ids, loss_mask


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=np.float64)
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

PARAM_KEYS = list(state_dict.keys())
flat_params = np.concatenate([state_dict[k].ravel() for k in PARAM_KEYS])
total_params = flat_params.size
offset = 0
for k in PARAM_KEYS:
    n = state_dict[k].size
    state_dict[k] = flat_params[offset : offset + n].reshape(state_dict[k].shape)
    offset += n

opt_state = {
    "flat_m": np.zeros(total_params),
    "flat_v": np.zeros(total_params),
    "flat_params": flat_params,
    "buf": np.empty(total_params),
    "tmp": np.empty(total_params),
}

tokenized = [tokenize(doc, uchars) for doc in docs]
grads = {k: np.zeros_like(state_dict[k]) for k in PARAM_KEYS}

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, grads, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
