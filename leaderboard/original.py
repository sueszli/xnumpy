# refactored version of original microgpt:
# - https://karpathy.github.io/2026/02/12/microgpt/
# - https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py


import math
import random
from pathlib import Path

random.seed(42)


# load
docs = (Path(__file__).parent / "input.txt").read_text().splitlines()
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# tokenize
uchars = sorted(set("".join(docs)))  # unique characters
BOS = len(uchars)  # special beginning of sequence (BOS) token
vocab_size = len(uchars) + 1  # +1 is for BOS
print(f"vocab size: {vocab_size}")


class Value:
    # autograd

    __slots__ = ("data", "grad", "_children", "_local_grads")  # perf optimization

    def __init__(self, data: float, children: tuple["Value", ...] = (), local_grads: tuple[float, ...] = ()) -> None:
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad = 0  # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children  # children of this node in the computation graph
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
        # original gpt2 uses gelu
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
n_layer = 1  # num stacked transformer blocks
n_embd = 16  # embedding dimensions
block_size = 16  # context window size (longest name is 15 chars)
n_head = 4  # num attention heads
head_dim = n_embd // n_head  # which subset of dims each head operates on
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]  # torch.randn(nout, nin) * std
state_dict: dict[str, list[list[Value]]] = {
    "wte": matrix(vocab_size, n_embd),  # weight token embedding
    "wpe": matrix(block_size, n_embd),  # weight positional embedding
    "lm_head": matrix(vocab_size, n_embd),  # language model head
}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)  # weight query
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)  # weight key
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)  # weight value
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)  # weight output
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)  # fully connected 1
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)  # fully connected 2
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
    # original gpt2 uses layernorm
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


# Transformer forward pass: token_ids -> logits
#
# Shapes (n=seq_len, d=n_embd=16, h=n_head=4, dh=head_dim=4, V=vocab_size):
# -----------------------------------------------------------------------
#   wte      (V, d)    token_id -> embedding vector
#   wpe      (T, d)    position -> embedding vector
#   attn_wq  (d, d)    x -> queries  (how much do I want to attend?)
#   attn_wk  (d, d)    x -> keys     (what do I offer to be attended to?)
#   attn_wv  (d, d)    x -> values   (what do I send if attended to?)
#   attn_wo  (d, d)    concat heads -> residual
#   mlp_fc1  (4d, d)   x -> hidden   (expand)
#   mlp_fc2  (d, 4d)   hidden -> x   (contract)
#   lm_head  (V, d)    x -> logits   (scores per token)
#
# Forward pass:
# -------------
#   x[t] = wte[token_ids[t]] + wpe[t]           # (d,)  embed + position
#
#   for each head h:
#     Q = x @ wq.T                              # (n, d)
#     K = x @ wk.T                              # (n, d)
#     V = x @ wv.T                              # (n, d)
#     scores = Q @ K.T / sqrt(head_dim)         # (n, n)  scaled dot-product
#     scores = masked_fill(scores, causal_mask) # (n, n)  can't see future
#     weights = softmax(scores)                 # (n, n)  sum to 1
#     head_out = weights @ V                    # (n, dh)
#   x = concat(head_outs) @ wo.T                # (n, d)  merge heads
#
#   h = relu(x @ mlp_fc1.T)                     # (n, 4d) expand
#   x = h @ mlp_fc2.T                           # (n, d)  contract
#
#   logits = x[-1] @ lm_head.T                  # (V,)    last token only
#   probs  = softmax(logits)                    # (V,)    next token dist
#
#   Q @ K.T  =  "does token i want to look at token j?"
#               (dot product = similarity between query and key)
#   weights  =  normalized attention scores (who looks at whom)
#   weights @ V  =  weighted mix of values from all attended tokens


def gpt(token_id: int, pos_id: int, keys: list[list[list[Value]]], values: list[list[list[Value]]]) -> list[Value]:
    tok_emb = state_dict["wte"][token_id]  # token embedding
    pos_emb = state_dict["wpe"][pos_id]  # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
    x = rmsnorm(x)  # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
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
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


# adam optimizer values
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
m: list[float] = [0.0] * len(params)  # first moment buffer
v: list[float] = [0.0] * len(params)  # second moment buffer

# train loop
num_steps = 1000
for step in range(num_steps):

    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    values: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)  # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0  # optimizer.zero_grad()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")


#
# inference
#


temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\ninference:")
for sample_idx in range(20):
    keys: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    values: list[list[list[Value]]] = [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([logit / temperature for logit in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
