# /// script
# requires-python = "==3.14.*"
# ///

# original microgpt (just refactored):
# - https://karpathy.github.io/2026/02/12/microgpt/
# - https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py

import math
import random
import time
from pathlib import Path

from utils.times import save_times
from utils.weights import dump_weights

random.seed(42)


# hyperparams
N_LAYER = 1  # num stacked transformer blocks
N_EMBED = 16  # embedding dimensions
BLOCK_SIZE = 16  # context window size (longest name is 15 chars)
N_HEAD = 4  # num attention heads
NUM_STEPS = 1000  # num training steps


# load
docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# tokenize
uchars = sorted(set("".join(docs)))  # unique chars as tokens
BOS = len(uchars)  # beginning of sequence BOS token
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")


class Value:
    # autograd

    __slots__ = ("data", "grad", "_children", "_local_grads")  # perf optimization

    def __init__(self, data: float, children: tuple["Value", ...] = (), local_grads: tuple[float, ...] = ()) -> None:
        self.data = data  # scalar value of this node (calculated during forward pass)
        self.grad = 0  # derivative of the loss w.r.t. this node (calculated in backward pass)
        self._children = children  # children of this node in graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float) -> "Value":
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self) -> "Value":
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self) -> "Value":
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> "Value":
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other)

    def __rsub__(self, other: "Value | float") -> "Value":
        return other + (-self)

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * other**-1

    def __rtruediv__(self, other: "Value | float") -> "Value":
        return other * self**-1

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


#
# training
#


# init params
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]  # torch.randn(nout, nin) * std
state_dict: dict[str, list[list[Value]]] = {
    "wte": matrix(vocab_size, N_EMBED),  # token embedding
    "wpe": matrix(BLOCK_SIZE, N_EMBED),  # positional embedding
    "lm_head": matrix(vocab_size, N_EMBED),  # language model head
}
for i in range(N_LAYER):
    state_dict[f"layer{i}.attn_wq"] = matrix(N_EMBED, N_EMBED)  # query
    state_dict[f"layer{i}.attn_wk"] = matrix(N_EMBED, N_EMBED)  # key
    state_dict[f"layer{i}.attn_wv"] = matrix(N_EMBED, N_EMBED)  # value
    state_dict[f"layer{i}.attn_wo"] = matrix(N_EMBED, N_EMBED)  # output
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * N_EMBED, N_EMBED)  # fully connected 1
    state_dict[f"layer{i}.mlp_fc2"] = matrix(N_EMBED, 4 * N_EMBED)  # fully connected 2
params: list[Value] = [p for mat in state_dict.values() for row in mat for p in row]  # flatten
print(f"num params: {len(params)}")


def linear(x: list[Value], w: list[list[Value]]) -> list[Value]:
    # x @ w.T
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: list[Value]) -> list[Value]:
    # F.softmax(logits, dim=-1)
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    # F.rms_norm(x, x.shape)
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


#   all embeddings                                    (d_model each)
#        │
#        ├─── @W_Q ──► Q_cat                          (d_head)
#        │
#        ├─── @W_K ──► K_the, K_fluffy, K_cat, K_sat  (d_head each)
#        │
#        └─── @W_V ──► V_the, V_fluffy, V_cat, V_sat ────────────────┐
#                          │                                         │
#               Q_cat · each K / sqrt(d_head)                        │
#                          │                                         │
#                       softmax                                      │
#                          │                                         │
#                   w = [0.01, 0.85, 0.13, 0.01]                     │
#                          │                                         │
#                          └──────── Σ  w_i · V_i ───────────────────┘
#                                           │
#                                     head_out_cat    (d_head)
#                                           │
#                                        @ W_O
#                                           │
#                                       delta_cat     (d_model)
#                                           │
#                               x  ──(+)──► x_new     (d_model)
#
#
#   Q  — what this token is looking for
#   K  — whether this token is worth looking at
#   V  — what this token injects when selected
#   O  — project head outputs back into residual-stream space
def gpt(
    token_id: int,
    pos_id: int,  # token position in the sequence
    keys: list[list[list[Value]]],  # KV-cache keys: [layer][past_token][d_model]
    values: list[list[list[Value]]],  # KV-cache values: [layer][past_token][d_model]
) -> list[Value]:
    tok_emb = state_dict["wte"][token_id]  # token embedding
    pos_emb = state_dict["wpe"][pos_id]  # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(N_LAYER):
        # multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])  # Q = x @ wq.T
        k = linear(x, state_dict[f"layer{li}.attn_wk"])  # K = x @ wk.T
        v = linear(x, state_dict[f"layer{li}.attn_wv"])  # V = x @ wv.T
        keys[li].append(k)  # causal mask is implicit. cache only holds past tokens, never future
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * N_EMBED // N_HEAD  # index offset
            q_h = q[hs : hs + N_EMBED // N_HEAD]
            k_h = [ki[hs : hs + N_EMBED // N_HEAD] for ki in keys[li]]
            v_h = [vi[hs : hs + N_EMBED // N_HEAD] for vi in values[li]]
            # scores = Q @ K.T / sqrt(N_EMBED // N_HEAD)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(N_EMBED // N_HEAD)) / (N_EMBED // N_HEAD) ** 0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            # head_out = weights @ V
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(N_EMBED // N_HEAD)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])  # x = concat(head_outs) @ wo.T
        x = [a + b for a, b in zip(x, x_residual)]

        # mlp block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])  # h = x @ mlp_fc1.T  -> expand
        x = [xi.relu() for xi in x]  # h = relu(h)
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])  # x = h @ mlp_fc2.T  -> contract
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])  # logits = x[-1] @ lm_head.T
    return logits


# adam optimizer values
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
m: list[float] = [0.0] * len(params)  # first moment buffer
v: list[float] = [0.0] * len(params)  # second moment buffer


# train loop
step_times = []
for step in range(NUM_STEPS):
    t0 = time.perf_counter()
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # turn to token ids
    n = min(BLOCK_SIZE, len(tokens) - 1)  # num prediction steps

    # forward
    keys: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
    values: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
    losses = []
    for pos_id in range(n):
        # predict
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        # compute loss
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    # compute gradients (∂loss/∂p for every param p)
    loss.backward()

    # adam optimizer weight update
    lr_t = learning_rate * (1 - step / NUM_STEPS)  # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)  # update weights
        p.grad = 0  # reset gradients

    step_times.append(time.perf_counter() - t0)
    print(f"step {step+1:4d} / {NUM_STEPS:4d} | loss {loss.data:.4f}", end="\r")

save_times(step_times)
dump_weights(state_dict)


#
# inference
#


print("\ninference:")
temperature = 0.5
for sample_idx in range(20):
    keys: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
    values: list[list[list[Value]]] = [[] for _ in range(N_LAYER)]
    token_id = BOS
    sample = []
    for pos_id in range(BLOCK_SIZE):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([logit / temperature for logit in logits])
        # sample from distribution
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
