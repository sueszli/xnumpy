# /// script
# requires-python = "==3.14.*"
# dependencies = [
#   "exojit @ git+https://github.com/sueszli/exojit.git",
#   "numpy",
# ]
# ///

import math
import random
import time
from pathlib import Path

import numpy as np
from exo import *
from utils import assert_weights_match, save_times

from exojit.main import compile_jit


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        out = self.data**other
        return Value(out, (self,), (other * out / self.data,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        out = math.exp(self.data)
        return Value(out, (self,), (out,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        completed = set()
        stack = [self]
        while stack:
            v = stack[-1]
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    if child not in visited:
                        stack.append(child)
            else:
                stack.pop()
                if v not in completed:
                    completed.add(v)
                    topo.append(v)
        self.grad = 1
        for v in reversed(topo):
            vgrad = v.grad
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * vgrad


@proc
def matvec_16_16(out: f64[16] @ DRAM, x: f64[16] @ DRAM, W: f64[16, 16] @ DRAM):
    for i in seq(0, 16):
        out[i] = 0.0
        for k in seq(0, 16):
            out[i] += W[i, k] * x[k]


@proc
def matvec_64_16(out: f64[64] @ DRAM, x: f64[16] @ DRAM, W: f64[64, 16] @ DRAM):
    for i in seq(0, 64):
        out[i] = 0.0
        for k in seq(0, 16):
            out[i] += W[i, k] * x[k]


@proc
def matvec_16_64(out: f64[16] @ DRAM, x: f64[64] @ DRAM, W: f64[16, 64] @ DRAM):
    for i in seq(0, 16):
        out[i] = 0.0
        for k in seq(0, 64):
            out[i] += W[i, k] * x[k]


_linear_jit = {
    (16, 16): compile_jit(matvec_16_16)["matvec_16_16"],
    (64, 16): compile_jit(matvec_64_16)["matvec_64_16"],
    (16, 64): compile_jit(matvec_16_64)["matvec_16_64"],
}


def linear(x, w):
    nout = len(w)
    nin = len(x)
    jit_fn = _linear_jit.get((nout, nin))
    if jit_fn is not None:
        x_arr = np.array([xi.data for xi in x], dtype=np.float64)
        w_arr = np.array([[wi.data for wi in wo] for wo in w], dtype=np.float64)
        out_arr = np.empty(nout, dtype=np.float64)
        jit_fn(out_arr, x_arr, w_arr)
        result = []
        for i, wo in enumerate(w):
            children = (*wo, *x)
            local_grads = tuple(xi.data for xi in x) + tuple(wi.data for wi in wo)
            result.append(Value(float(out_arr[i]), children, local_grads))
        return result
    else:
        result = []
        for wo in w:
            data = sum(wi.data * xi.data for wi, xi in zip(wo, x))
            children = (*wo, *x)
            local_grads = (*(xi.data for xi in x), *(wi.data for wi in wo))
            result.append(Value(data, children, local_grads))
        return result


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


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

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
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
params = [p for mat in state_dict.values() for row in mat for p in row]

learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

char_to_id = {ch: i for i, ch in enumerate(uchars)}

num_steps = 1000
step_times = []
for step in range(num_steps):
    t0 = time.perf_counter()

    doc = docs[step % len(docs)]
    tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values, state_dict, n_layer, n_head, head_dim)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    bc1 = 1 - beta1 ** (step + 1)
    bc2 = 1 - beta2 ** (step + 1)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / bc1
        v_hat = v[i] / bc2
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0

    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")

save_times(step_times)
assert_weights_match(state_dict)
