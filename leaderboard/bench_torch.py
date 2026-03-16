# /// script
# dependencies = ["torch", "tqdm"]
# ///

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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


def matrix(nout: int, nin: int, std: float = 0.08) -> torch.Tensor:
    return torch.tensor([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)], dtype=torch.float64).requires_grad_(True)


state_dict = {"wte": matrix(vocab_size, n_embd), "wpe": matrix(block_size, n_embd), "lm_head": matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)


def rmsnorm(x: torch.Tensor) -> torch.Tensor:
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


@torch.compile(dynamic=True)
def forward(input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    n = input_ids.shape[0]
    x = rmsnorm(state_dict["wte"][input_ids] + state_dict["wpe"][torch.arange(n)])
    for li in range(n_layer):
        x_residual = x
        xn = rmsnorm(x)
        q = (xn @ state_dict[f"layer{li}.attn_wq"].T).view(n, n_head, head_dim).transpose(0, 1)
        k = (xn @ state_dict[f"layer{li}.attn_wk"].T).view(n, n_head, head_dim).transpose(0, 1)
        v = (xn @ state_dict[f"layer{li}.attn_wv"].T).view(n, n_head, head_dim).transpose(0, 1)
        mask = torch.triu(torch.full((n, n), float("-inf"), dtype=x.dtype), 1)
        attn_weights = F.softmax(q @ k.transpose(-2, -1) / head_dim**0.5 + mask, dim=-1)
        x = (attn_weights @ v).transpose(0, 1).reshape(n, n_embd) @ state_dict[f"layer{li}.attn_wo"].T + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = F.relu(xn @ state_dict[f"layer{li}.mlp_fc1"].T) @ state_dict[f"layer{li}.mlp_fc2"].T + x_residual
    return F.cross_entropy(x @ state_dict["lm_head"].T, target_ids)


learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
num_steps = 1000
m = {k: torch.zeros_like(p) for k, p in state_dict.items()}
v = {k: torch.zeros_like(p) for k, p in state_dict.items()}

pbar = tqdm(range(num_steps))
step_times = []
for step in pbar:
    t0 = time.perf_counter()
    doc = docs[step % len(docs)]
    tokens = torch.tensor([BOS] + [uchars.index(ch) for ch in doc] + [BOS])
    n = min(block_size, len(tokens) - 1)
    loss = forward(tokens[:n], tokens[1 : n + 1])
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    with torch.no_grad():
        for k, p in state_dict.items():
            m[k] = beta1 * m[k] + (1 - beta1) * p.grad
            v[k] = beta2 * v[k] + (1 - beta2) * p.grad**2
            m_hat = m[k] / (1 - beta1 ** (step + 1))
            v_hat = v[k] / (1 - beta2 ** (step + 1))
            p -= lr_t * m_hat / (v_hat.sqrt() + eps_adam)
            p.grad.zero_()
    pbar.set_postfix(loss=f"{loss.item():.4f}")
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
assert_weights_match(state_dict)
