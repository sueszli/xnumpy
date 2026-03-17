# /// script
# requires-python = "==3.14.*"
# dependencies = ["numpy", "tqdm"]
# ///

import random
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import assert_weights_match, save_times

random.seed(42)
docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: np.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=np.float64)
state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
    **{f"layer{i}.attn_wq": matrix(n_embd, n_embd) for i in range(n_layer)},
    **{f"layer{i}.attn_wk": matrix(n_embd, n_embd) for i in range(n_layer)},
    **{f"layer{i}.attn_wv": matrix(n_embd, n_embd) for i in range(n_layer)},
    **{f"layer{i}.attn_wo": matrix(n_embd, n_embd) for i in range(n_layer)},
    **{f"layer{i}.mlp_fc1": matrix(4 * n_embd, n_embd) for i in range(n_layer)},
    **{f"layer{i}.mlp_fc2": matrix(n_embd, 4 * n_embd) for i in range(n_layer)},
}


def _rmsnorm_fwd(x: np.ndarray):
    rms = (np.mean(x**2, axis=-1, keepdims=True) + 1e-5) ** -0.5
    return x * rms, rms


def _rmsnorm_bwd(dout: np.ndarray, x: np.ndarray, rms: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    dot = (dout * x).sum(axis=-1, keepdims=True)
    return dout * rms - (rms**3 / d) * x * dot


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def forward_backward(input_ids: np.ndarray, target_ids: np.ndarray):
    n = len(input_ids)
    grads = {k: np.zeros_like(v) for k, v in state_dict.items()}

    emb = state_dict["wte"][input_ids] + state_dict["wpe"][np.arange(n)]
    x, rms_init = _rmsnorm_fwd(emb)

    layer_cache = []
    for li in range(n_layer):
        wq = state_dict[f"layer{li}.attn_wq"]
        wk = state_dict[f"layer{li}.attn_wk"]
        wv = state_dict[f"layer{li}.attn_wv"]
        wo = state_dict[f"layer{li}.attn_wo"]
        fc1 = state_dict[f"layer{li}.mlp_fc1"]
        fc2 = state_dict[f"layer{li}.mlp_fc2"]

        x_pre_attn = x
        xn_attn, rms_attn = _rmsnorm_fwd(x)

        q = (xn_attn @ wq.T).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        k = (xn_attn @ wk.T).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        v = (xn_attn @ wv.T).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        mask = np.triu(np.full((n, n), -1e10), 1)
        attn_w = _softmax(q @ k.transpose(0, 2, 1) / head_dim**0.5 + mask)
        attn_out = attn_w @ v
        attn_out_flat = attn_out.transpose(1, 0, 2).reshape(n, n_embd)
        x = attn_out_flat @ wo.T + x_pre_attn

        x_pre_mlp = x
        xn_mlp, rms_mlp = _rmsnorm_fwd(x)

        h_pre = xn_mlp @ fc1.T
        h = np.maximum(0.0, h_pre)
        x = h @ fc2.T + x_pre_mlp

        layer_cache.append({"x_pre_attn": x_pre_attn, "xn_attn": xn_attn, "rms_attn": rms_attn, "q": q, "k": k, "v": v, "attn_w": attn_w, "attn_out_flat": attn_out_flat, "x_pre_mlp": x_pre_mlp, "xn_mlp": xn_mlp, "rms_mlp": rms_mlp, "h_pre": h_pre, "h": h})

    logits = x @ state_dict["lm_head"].T
    probs = _softmax(logits)
    loss = -np.log(probs[np.arange(n), target_ids]).mean()

    dlogits = probs / n
    dlogits[np.arange(n), target_ids] -= 1.0 / n

    grads["lm_head"] = dlogits.T @ x
    dx = dlogits @ state_dict["lm_head"]

    for li in reversed(range(n_layer)):
        cache = layer_cache[li]
        wq = state_dict[f"layer{li}.attn_wq"]
        wk = state_dict[f"layer{li}.attn_wk"]
        wv = state_dict[f"layer{li}.attn_wv"]
        wo = state_dict[f"layer{li}.attn_wo"]
        fc1 = state_dict[f"layer{li}.mlp_fc1"]
        fc2 = state_dict[f"layer{li}.mlp_fc2"]

        dx_res_mlp = dx
        grads[f"layer{li}.mlp_fc2"] = dx.T @ cache["h"]
        dh = dx @ fc2
        dh_pre = dh * (cache["h_pre"] > 0)
        grads[f"layer{li}.mlp_fc1"] = dh_pre.T @ cache["xn_mlp"]
        dxn_mlp = dh_pre @ fc1
        dx = _rmsnorm_bwd(dxn_mlp, cache["x_pre_mlp"], cache["rms_mlp"]) + dx_res_mlp

        dx_res_attn = dx
        grads[f"layer{li}.attn_wo"] = dx.T @ cache["attn_out_flat"]
        dattn_out_flat = dx @ wo

        dattn_out = dattn_out_flat.reshape(n, n_head, head_dim).transpose(1, 0, 2)

        aw = cache["attn_w"]
        dv = aw.transpose(0, 2, 1) @ dattn_out
        dattn_w = dattn_out @ cache["v"].transpose(0, 2, 1)

        dlogits_attn = aw * (dattn_w - (dattn_w * aw).sum(-1, keepdims=True)) / head_dim**0.5

        dq = dlogits_attn @ cache["k"]
        dk = dlogits_attn.transpose(0, 2, 1) @ cache["q"]

        dq_flat = dq.transpose(1, 0, 2).reshape(n, n_embd)
        dk_flat = dk.transpose(1, 0, 2).reshape(n, n_embd)
        dv_flat = dv.transpose(1, 0, 2).reshape(n, n_embd)

        grads[f"layer{li}.attn_wq"] = dq_flat.T @ cache["xn_attn"]
        grads[f"layer{li}.attn_wk"] = dk_flat.T @ cache["xn_attn"]
        grads[f"layer{li}.attn_wv"] = dv_flat.T @ cache["xn_attn"]
        dxn_attn = dq_flat @ wq + dk_flat @ wk + dv_flat @ wv

        dx = _rmsnorm_bwd(dxn_attn, cache["x_pre_attn"], cache["rms_attn"]) + dx_res_attn

    demb = _rmsnorm_bwd(dx, emb, rms_init)
    np.add.at(grads["wte"], input_ids, demb)
    grads["wpe"][:n] += demb

    return loss, grads


learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
num_steps = 1000
m = {k: np.zeros_like(p) for k, p in state_dict.items()}
v = {k: np.zeros_like(p) for k, p in state_dict.items()}

pbar = tqdm(range(num_steps))
step_times = []
for step in pbar:
    t0 = time.perf_counter()
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    input_ids = np.array(tokens[:n])
    target_ids = np.array(tokens[1 : n + 1])

    loss, grads = forward_backward(input_ids, target_ids)

    lr_t = learning_rate * (1 - step / num_steps)
    for k in state_dict:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
        m_hat = m[k] / (1 - beta1 ** (step + 1))
        v_hat = v[k] / (1 - beta2 ** (step + 1))
        state_dict[k] -= lr_t * m_hat / (v_hat**0.5 + eps_adam)

    step_times.append(time.perf_counter() - t0)
    pbar.set_postfix(loss=f"{loss:.4f}")

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat] for k, mat in state_dict.items()})
