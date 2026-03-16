# /// script
# requires-python = ">=3.11"
# dependencies = ["jax[cpu]", "optax"]
# ///

import random
import time
from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from utils import assert_weights_match, save_times

jax.config.update("jax_enable_x64", True)  # f64 instead of f32 to match precision
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


def matrix(nout: int, nin: int, std: float = 0.08) -> jax.Array:
    return jnp.array([[random.gauss(0, std) for _ in range(nin)] for _ in range(nout)])


state_dict: dict[str, jax.Array] = {"wte": matrix(vocab_size, n_embd), "wpe": matrix(block_size, n_embd), "lm_head": matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)


def rmsnorm(x: jax.Array) -> jax.Array:
    return x * (jnp.mean(x**2, axis=-1, keepdims=True) + 1e-5) ** -0.5


def forward(input_ids: jax.Array, target_ids: jax.Array, params: dict[str, jax.Array]) -> jax.Array:
    n = input_ids.shape[0]
    x = rmsnorm(params["wte"][input_ids] + params["wpe"][jnp.arange(n)])
    for li in range(n_layer):
        x_residual = x
        xn = rmsnorm(x)
        q = (xn @ params[f"layer{li}.attn_wq"].T).reshape(n, n_head, head_dim)
        k = (xn @ params[f"layer{li}.attn_wk"].T).reshape(n, n_head, head_dim)
        v = (xn @ params[f"layer{li}.attn_wv"].T).reshape(n, n_head, head_dim)
        mask = jnp.triu(jnp.full((n, n), -1e10), 1)
        attn_weights = jax.nn.softmax(jnp.einsum("ihd,jhd->hij", q, k) / head_dim**0.5 + mask, axis=-1)
        x = jnp.einsum("hij,jhd->ihd", attn_weights, v).reshape(n, n_embd) @ params[f"layer{li}.attn_wo"].T + x_residual
        x_residual = x
        xn = rmsnorm(x)
        x = jax.nn.relu(xn @ params[f"layer{li}.mlp_fc1"].T) @ params[f"layer{li}.mlp_fc2"].T + x_residual
    return -jax.nn.log_softmax(x @ params["lm_head"].T, axis=-1)[jnp.arange(n), target_ids].mean()


learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
num_steps = 1000
optimizer = optax.adam(optax.linear_schedule(learning_rate, 0.0, num_steps), b1=beta1, b2=beta2, eps=eps_adam)
opt_state = optimizer.init(state_dict)
step_fn = jax.jit(jax.value_and_grad(forward, argnums=2))

step_times = []
for step in range(num_steps):
    t0 = time.perf_counter()
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    input_ids = jnp.array(tokens[:n])
    target_ids = jnp.array(tokens[1 : n + 1])
    loss, grads = step_fn(input_ids, target_ids, state_dict)
    updates, opt_state = optimizer.update(grads, opt_state)
    state_dict = optax.apply_updates(state_dict, updates)
    print(f"step {step+1:4d} / {num_steps:4d} | loss {float(loss):.4f}", end="\r")  # float(loss) syncs JAX
    step_times.append(time.perf_counter() - t0)

save_times(step_times)
W = namedtuple("W", ["data"])  # assert_weights_match expects .data on each element. JAX arrays don't have it
assert_weights_match({k: [[W(float(v)) for v in row] for row in mat.tolist()] for k, mat in state_dict.items()})
