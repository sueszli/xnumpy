# /// script
# requires-python = "==3.14.*"
# dependencies = ["numpy"]
# ///

import functools
import random
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000

MASK = np.triu(np.full((BLOCK_SIZE, BLOCK_SIZE), -1e10), 1)


def rmsnorm_fwd(x: np.ndarray):
    rms = (np.mean(x**2, axis=-1, keepdims=True) + 1e-5) ** -0.5
    return x * rms, rms


def rmsnorm_bwd(dout: np.ndarray, x: np.ndarray, rms: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    dot = (dout * x).sum(axis=-1, keepdims=True)
    return dout * rms - (rms**3 / d) * x * dot


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def forward_backward(params: dict[str, np.ndarray], input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray):
    n = len(input_ids)

    grads = {}
    grads["wte"] = np.zeros_like(params["wte"])
    grads["wpe"] = np.zeros_like(params["wpe"])

    emb = params["wte"][input_ids] + params["wpe"][np.arange(n)]
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

        q = (xn_attn @ wq.T).reshape(n, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
        k = (xn_attn @ wk.T).reshape(n, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
        v = (xn_attn @ wv.T).reshape(n, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)
        mask = MASK[:n, :n]
        attn_w = softmax(q @ k.transpose(0, 2, 1) / (N_EMBED // N_HEAD) ** 0.5 + mask)
        attn_out = attn_w @ v
        attn_out_flat = attn_out.transpose(1, 0, 2).reshape(n, N_EMBED)
        x = attn_out_flat @ wo.T + x_pre_attn

        x_pre_mlp = x
        xn_mlp, rms_mlp = rmsnorm_fwd(x)

        h_pre = xn_mlp @ fc1.T
        h = np.maximum(0.0, h_pre)
        x = h @ fc2.T + x_pre_mlp

        layer_cache.append({"x_pre_attn": x_pre_attn, "xn_attn": xn_attn, "rms_attn": rms_attn, "q": q, "k": k, "v": v, "attn_w": attn_w, "attn_out_flat": attn_out_flat, "x_pre_mlp": x_pre_mlp, "xn_mlp": xn_mlp, "rms_mlp": rms_mlp, "h_pre": h_pre, "h": h})

    logits = x @ params["lm_head"].T
    probs = softmax(logits)

    sum_mask = loss_mask.sum()
    loss = -(np.log(probs[np.arange(n), target_ids]) * loss_mask).sum() / sum_mask

    dlogits = probs / sum_mask
    dlogits[np.arange(n), target_ids] -= 1.0 / sum_mask
    dlogits *= loss_mask[:, None]

    grads["lm_head"] = dlogits.T @ x
    dx = dlogits @ params["lm_head"]

    for li in reversed(range(N_LAYER)):
        cache = layer_cache[li]
        wq = params[f"layer{li}.attn_wq"]
        wk = params[f"layer{li}.attn_wk"]
        wv = params[f"layer{li}.attn_wv"]
        wo = params[f"layer{li}.attn_wo"]
        fc1 = params[f"layer{li}.mlp_fc1"]
        fc2 = params[f"layer{li}.mlp_fc2"]

        dx_res_mlp = dx
        grads[f"layer{li}.mlp_fc2"] = dx.T @ cache["h"]
        dh = dx @ fc2
        dh_pre = dh * (cache["h_pre"] > 0)
        grads[f"layer{li}.mlp_fc1"] = dh_pre.T @ cache["xn_mlp"]
        dxn_mlp = dh_pre @ fc1
        dx = rmsnorm_bwd(dxn_mlp, cache["x_pre_mlp"], cache["rms_mlp"]) + dx_res_mlp

        dx_res_attn = dx
        grads[f"layer{li}.attn_wo"] = dx.T @ cache["attn_out_flat"]
        dattn_out_flat = dx @ wo

        dattn_out = dattn_out_flat.reshape(n, N_HEAD, N_EMBED // N_HEAD).transpose(1, 0, 2)

        aw = cache["attn_w"]
        dv = aw.transpose(0, 2, 1) @ dattn_out
        dattn_w = dattn_out @ cache["v"].transpose(0, 2, 1)

        dlogits_attn = aw * (dattn_w - (dattn_w * aw).sum(-1, keepdims=True)) / (N_EMBED // N_HEAD) ** 0.5

        dq = dlogits_attn @ cache["k"]
        dk = dlogits_attn.transpose(0, 2, 1) @ cache["q"]

        dq_flat = dq.transpose(1, 0, 2).reshape(n, N_EMBED)
        dk_flat = dk.transpose(1, 0, 2).reshape(n, N_EMBED)
        dv_flat = dv.transpose(1, 0, 2).reshape(n, N_EMBED)

        grads[f"layer{li}.attn_wq"] = dq_flat.T @ cache["xn_attn"]
        grads[f"layer{li}.attn_wk"] = dk_flat.T @ cache["xn_attn"]
        grads[f"layer{li}.attn_wv"] = dv_flat.T @ cache["xn_attn"]
        dxn_attn = dq_flat @ wq + dk_flat @ wk + dv_flat @ wv

        dx = rmsnorm_bwd(dxn_attn, cache["x_pre_attn"], cache["rms_attn"]) + dx_res_attn

    demb = rmsnorm_bwd(dx, emb, rms_init)
    np.add.at(grads["wte"], input_ids, demb)
    grads["wpe"][:n] += demb

    return loss, grads


def step_fn(params: dict[str, np.ndarray], opt_state: dict[str, dict[str, np.ndarray]], input_ids: np.ndarray, target_ids: np.ndarray, loss_mask: np.ndarray, step: int) -> tuple[float, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    loss, grads = forward_backward(params, input_ids, target_ids, loss_mask)

    learning_rate = 0.01
    beta1 = 0.85
    beta2 = 0.99
    eps_adam = 1e-8
    lr_t = learning_rate * (1 - step / NUM_STEPS)

    m = opt_state["m"]
    v = opt_state["v"]

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    for k in params:
        m[k] *= beta1
        m[k] += (1 - beta1) * grads[k]
        v[k] *= beta2
        v[k] += (1 - beta2) * grads[k] ** 2
        m_hat = m[k] / bias_correction1
        v_hat = v[k] / bias_correction2
        params[k] -= lr_t * m_hat / (v_hat**0.5 + eps_adam)

    return loss, params, opt_state


@functools.cache
def char_to_id(uchars_tuple: tuple[str, ...]) -> dict[str, int]:
    return {ch: i for i, ch in enumerate(uchars_tuple)}


def tokenize(doc: str, uchars: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2i = char_to_id(tuple(uchars))
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

opt_state = {"m": {k: np.zeros_like(p) for k, p in state_dict.items()}, "v": {k: np.zeros_like(p) for k, p in state_dict.items()}}

tokenized = [tokenize(doc, uchars) for doc in docs]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, loss_mask, step)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {float(loss):.4f}", end="\r")

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
