# /// script
# requires-python = "==3.14.*"
# dependencies = ["torch"]
# ///

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from utils import assert_weights_match, save_times

random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


def forward(params: dict[str, torch.Tensor], input_ids: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    x = rmsnorm(params["wte"][input_ids] + params["wpe"])
    for li in range(N_LAYER):
        x_residual = x
        xn = rmsnorm(x)
        q = F.linear(xn, params[f"layer{li}.attn_wq"]).view(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        k = F.linear(xn, params[f"layer{li}.attn_wk"]).view(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        v = F.linear(xn, params[f"layer{li}.attn_wv"]).view(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD).transpose(0, 1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = F.linear(attn_out.transpose(0, 1).reshape(BLOCK_SIZE, N_EMBED), params[f"layer{li}.attn_wo"]) + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = F.linear(F.relu(F.linear(xn, params[f"layer{li}.mlp_fc1"])), params[f"layer{li}.mlp_fc2"]) + x_residual
    logits = F.linear(x, params["lm_head"])
    per_token_loss = F.cross_entropy(logits, target_ids, reduction="none")
    return (per_token_loss * loss_mask).sum() / loss_mask.sum()


@torch.compile
def train_step(params: dict[str, torch.Tensor], m: dict[str, torch.Tensor], v: dict[str, torch.Tensor], lr: torch.Tensor, bc1: torch.Tensor, bc2: torch.Tensor, input_ids: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> tuple:
    grads, loss = torch.func.grad_and_value(forward)(params, input_ids, target_ids, loss_mask)
    new_m = {k: 0.85 * m[k] + 0.15 * grads[k] for k in params}
    new_v = {k: 0.99 * v[k] + 0.01 * grads[k].pow(2) for k in params}
    new_params = {k: params[k] - lr * (new_m[k] / bc1) / ((new_v[k] / bc2).sqrt() + 1e-8) for k in params}
    return new_params, new_m, new_v, loss


def tokenize(docs: list[str], uchars: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def tokenize_doc(doc: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2i = {ch: i for i, ch in enumerate(uchars)}
        bos = len(uchars)
        tokens = [bos] + [c2i[ch] for ch in doc] + [bos]
        n = min(BLOCK_SIZE, len(tokens) - 1)

        input_ids = torch.zeros(BLOCK_SIZE, dtype=torch.long)
        target_ids = torch.zeros(BLOCK_SIZE, dtype=torch.long)
        loss_mask = torch.zeros(BLOCK_SIZE, dtype=torch.float64)

        input_ids[:n] = torch.tensor(tokens[:n], dtype=torch.long)
        target_ids[:n] = torch.tensor(tokens[1 : n + 1], dtype=torch.long)
        loss_mask[:n] = 1.0

        return input_ids, target_ids, loss_mask

    per_doc = [tokenize_doc(doc) for doc in docs]
    return map(torch.stack, zip(*[per_doc[step % len(per_doc)] for step in range(NUM_STEPS)]))


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=torch.float64)
state_dict: dict[str, torch.Tensor] = {
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

train_inputs, train_targets, train_masks = tokenize(docs, uchars)

m = {k: torch.zeros_like(v) for k, v in state_dict.items()}
v_mom = {k: torch.zeros_like(v) for k, v in state_dict.items()}
lrs = torch.tensor([0.01 * (1 - step / NUM_STEPS) for step in range(NUM_STEPS)], dtype=torch.float64)
bc1s = torch.tensor([1 - 0.85 ** (step + 1) for step in range(NUM_STEPS)], dtype=torch.float64)
bc2s = torch.tensor([1 - 0.99 ** (step + 1) for step in range(NUM_STEPS)], dtype=torch.float64)

# precompile
train_step(state_dict, m, v_mom, lrs[0], bc1s[0], bc2s[0], train_inputs[0], train_targets[0], train_masks[0])

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    state_dict, m, v_mom, loss = train_step(state_dict, m, v_mom, lrs[step], bc1s[step], bc2s[step], train_inputs[step], train_targets[step], train_masks[step])
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {loss.item():.4f}", end="\r")

save_times(step_times)
assert_weights_match(state_dict)
