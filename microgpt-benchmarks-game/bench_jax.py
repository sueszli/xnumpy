# /// script
# requires-python = "==3.14.*"
# dependencies = ["jax[cpu]", "optax", "numpy"]
# ///

import random
import time
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils.times import save_times
from utils.weights import assert_weights_match

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
    x = rmsnorm(params["wte"][input_ids] + params["wpe"])
    for i in range(N_LAYER):
        x_residual = x
        xn = rmsnorm(x)
        q = (xn @ params[f"layer{i}.attn_wq"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD)
        k = (xn @ params[f"layer{i}.attn_wk"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD)
        v = (xn @ params[f"layer{i}.attn_wv"].T).reshape(BLOCK_SIZE, N_HEAD, N_EMBED // N_HEAD)
        attn_out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        x = attn_out.reshape(BLOCK_SIZE, N_EMBED) @ params[f"layer{i}.attn_wo"].T + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = jax.nn.relu(xn @ params[f"layer{i}.mlp_fc1"].T) @ params[f"layer{i}.mlp_fc2"].T + x_residual
    logits = x @ params["lm_head"].T
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    return (per_token_loss * loss_mask).sum() / loss_mask.sum()


@jax.jit
def train_fn(params: dict[str, jax.Array], opt_state: optax.OptState, train_inputs: jax.Array, train_targets: jax.Array, train_masks: jax.Array) -> tuple[dict[str, jax.Array], jax.Array]:
    def scan_fn(train_state, step_batch):
        params, opt_state = train_state
        input_ids, target_ids, loss_mask = step_batch
        loss, grads = jax.value_and_grad(forward, argnums=0)(params, input_ids, target_ids, loss_mask)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), loss

    (params, _), losses = jax.lax.scan(scan_fn, (params, opt_state), (train_inputs, train_targets, train_masks))
    return params, losses


def tokenize(docs: list[str], uchars: list[str]) -> tuple[jax.Array, jax.Array, jax.Array]:
    def tokenize_doc(doc: str) -> tuple[jax.Array, jax.Array, jax.Array]:
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

    per_doc = [tokenize_doc(doc) for doc in docs]
    train_inputs, train_targets, train_masks = map(jnp.stack, zip(*[per_doc[step % len(per_doc)] for step in range(NUM_STEPS)]))
    return train_inputs, train_targets, train_masks


docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
uchars = sorted(set("".join(docs)))

matrix = lambda nout, nin, std=0.08: jnp.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])
state_dict: dict[str, jax.Array] = {
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

optimizer = optax.adam(optax.linear_schedule(0.01, 0.0, NUM_STEPS), b1=0.85, b2=0.99, eps=1e-8)
opt_state = optimizer.init(state_dict)

train_inputs, train_targets, train_masks = tokenize(docs, uchars)

t0 = time.perf_counter()
state_dict, losses = train_fn(state_dict, opt_state, train_inputs, train_targets, train_masks)
jax.block_until_ready(losses)
total_time = time.perf_counter() - t0
step_times = [total_time / NUM_STEPS] * NUM_STEPS  # we cant measure individual steps, everything is jitted at once

save_times(step_times)
W = namedtuple("W", ["data"])
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat.tolist()] for k, mat in state_dict.items()})
