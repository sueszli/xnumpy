# /// script
# requires-python = "==3.14.*"
# dependencies = ["jax[cpu]", "optax", "tqdm"]
# ///

import random
import time
from collections import namedtuple
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from utils import assert_weights_match, save_times

jax.config.update("jax_enable_x64", True)
random.seed(42)


N_LAYER = 1
N_EMBED = 16
BLOCK_SIZE = 16
N_HEAD = 4
NUM_STEPS = 1000


def rmsnorm(x: jax.Array) -> jax.Array:
    return x * (jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5) ** -0.5


def forward(params: dict[str, jax.Array], input_ids: jax.Array, target_ids: jax.Array, loss_mask: jax.Array) -> jax.Array:
    n = input_ids.shape[0]
    x = rmsnorm(params["wte"][input_ids] + params["wpe"][jnp.arange(n)])
    for i in range(N_LAYER):
        x_residual = x
        xn = rmsnorm(x)
        q = (xn @ params[f"layer{i}.attn_wq"].T).reshape(n, N_HEAD, N_EMBED // N_HEAD)
        k = (xn @ params[f"layer{i}.attn_wk"].T).reshape(n, N_HEAD, N_EMBED // N_HEAD)
        v = (xn @ params[f"layer{i}.attn_wv"].T).reshape(n, N_HEAD, N_EMBED // N_HEAD)
        mask = jnp.triu(jnp.full((n, n), -1e10), 1)
        attn_weights = jax.nn.softmax(jnp.einsum("ihd,jhd->hij", q, k) / (N_EMBED // N_HEAD) ** 0.5 + mask, axis=-1)
        x = jnp.einsum("hij,jhd->ihd", attn_weights, v).reshape(n, N_EMBED) @ params[f"layer{i}.attn_wo"].T + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = jax.nn.relu(xn @ params[f"layer{i}.mlp_fc1"].T) @ params[f"layer{i}.mlp_fc2"].T + x_residual
    per_token_loss = -jax.nn.log_softmax(x @ params["lm_head"].T, axis=-1)[jnp.arange(n), target_ids]
    return (per_token_loss * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnums=(5,))
def step_fn(params: dict[str, jax.Array], opt_state: optax.OptState, input_ids: jax.Array, target_ids: jax.Array, loss_mask: jax.Array, optimizer: optax.GradientTransformation) -> tuple[jax.Array, dict[str, jax.Array], optax.OptState]:
    loss, grads = jax.value_and_grad(forward, argnums=0)(params, input_ids, target_ids, loss_mask)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state


def tokenize(doc: str, uchars: list[str]) -> tuple[jax.Array, jax.Array, jax.Array]:
    bos = len(uchars)
    tokens = jnp.array([bos] + [uchars.index(ch) for ch in doc] + [bos])
    n = min(BLOCK_SIZE, len(tokens) - 1)
    pad = (0, BLOCK_SIZE - n)
    return jnp.pad(tokens[:n], pad), jnp.pad(tokens[1 : n + 1], pad), jnp.pad(jnp.ones(n), pad)


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1

matrix = lambda nout, nin, std=0.08: jnp.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])
state_dict: dict[str, jax.Array] = {
    "wte": matrix(vocab_size, N_EMBED),
    "wpe": matrix(BLOCK_SIZE, N_EMBED),
    "lm_head": matrix(vocab_size, N_EMBED),
    **{f"layer{i}.attn_wq": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wk": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wv": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.attn_wo": matrix(N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc1": matrix(4 * N_EMBED, N_EMBED) for i in range(N_LAYER)},
    **{f"layer{i}.mlp_fc2": matrix(N_EMBED, 4 * N_EMBED) for i in range(N_LAYER)},
}

optimizer = optax.adam(optax.linear_schedule(0.01, 0.0, NUM_STEPS), b1=0.85, b2=0.99, eps=1e-8)
opt_state = optimizer.init(state_dict)

tokenized = [tokenize(doc, uchars) for doc in tqdm(docs, desc="tokenizing")]

step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    input_ids, target_ids, loss_mask = tokenized[step % len(tokenized)]
    loss, state_dict, opt_state = step_fn(state_dict, opt_state, input_ids, target_ids, loss_mask, optimizer)
    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {float(loss):.4f}", end="\r")

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat.tolist()] for k, mat in state_dict.items()})
