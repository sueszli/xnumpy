# choose kernels from:
# - https://karpathy.github.io/2026/02/12/microgpt/
# - https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py


from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from plotnine import aes, element_line, element_rect, element_text, expand_limits, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual, scale_linetype_manual, scale_shape_manual, theme, theme_minimal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from kernels.add import add
from kernels.add_neon import add_neon
from kernels.cross_entropy import cross_entropy
from kernels.cross_entropy_neon import cross_entropy_neon
from kernels.dot import dot
from kernels.dot_neon import dot_neon
from kernels.embedding import embedding
from kernels.embedding_neon import embedding_neon
from kernels.matmul import matmul
from kernels.matmul_neon import matmul_neon
from kernels.matvec import matvec
from kernels.matvec_neon import matvec_neon
from kernels.relu import relu
from kernels.relu_neon import relu_neon
from kernels.rmsnorm import rmsnorm
from kernels.rmsnorm_neon import rmsnorm_neon
from kernels.saxpy import saxpy
from kernels.saxpy_neon import saxpy_neon
from kernels.softmax import _jit_max_neon, softmax
from kernels.softmax_neon import softmax_neon
from kernels.weighted_sum import weighted_sum
from kernels.weighted_sum_neon import weighted_sum_neon

REPEATS = 200


bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []


#
# matmul
#


matmul_sizes = [1 << 5, 1 << 6, 1 << 7, 1 << 8]


for n in tqdm(matmul_sizes, desc="matmul"):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    expected = A @ B
    flops = 2 * n**3

    # numpy
    t_np = bench(lambda A=A, B=B: A @ B)

    # exo auto-vectorized
    fn_exo = matmul(n, n, n)
    C_exo = np.zeros((n, n), dtype=np.float32)
    fn_exo(C_exo, A, B)
    assert np.allclose(C_exo, expected, atol=1e-3)
    t_exo = bench(lambda C=C_exo, A=A, B=B: fn_exo(C, A, B))

    # exo neon intrinsics
    fn_neon = matmul_neon(n, n, n)
    C_neon = np.zeros((n, n), dtype=np.float32)
    fn_neon(C_neon, A, B)
    assert np.allclose(C_neon, expected, atol=1e-3)
    t_neon = bench(lambda C=C_neon, A=A, B=B: fn_neon(C, A, B))

    rows.append(
        {
            "kernel": "matmul",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 1),
            "exo_gflops": round(flops / t_exo / 1e9, 1),
            "neon_gflops": round(flops / t_neon / 1e9, 1),
        }
    )


#
# matvec (linear projection) y[j] = sum_i w[j][i] * x[i]
#


matvec_sizes = [1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10]


for n in tqdm(matvec_sizes, desc="matvec"):
    W = np.random.randn(n, n).astype(np.float32)
    x = np.random.randn(n).astype(np.float32)
    expected = W @ x
    flops = 2 * n**2  # n muls + n adds per output, n outputs

    # numpy
    t_np = bench(lambda W=W, x=x: W @ x)

    # exo auto-vectorized
    fn_exo = matvec(n, n)
    y_exo = np.zeros(n, dtype=np.float32)
    fn_exo(y_exo, W, x)
    assert np.allclose(y_exo, expected, atol=1e-3), f"matvec exo wrong: max_diff={np.max(np.abs(y_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, y=y_exo, W=W, x=x: fn(y, W, x))

    # exo neon intrinsics (uses transposed w for contiguous neon access)
    WT = np.ascontiguousarray(W.T)
    fn_neon = matvec_neon(n, n)
    y_neon = np.zeros(n, dtype=np.float32)
    fn_neon(y_neon, WT, x)
    assert np.allclose(y_neon, expected, atol=1e-3), f"matvec neon wrong: max_diff={np.max(np.abs(y_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, y=y_neon, WT=WT, x=x: fn(y, WT, x))

    rows.append(
        {
            "kernel": "matvec",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 1),
            "exo_gflops": round(flops / t_exo / 1e9, 1),
            "neon_gflops": round(flops / t_neon / 1e9, 1),
        }
    )


#
# saxpy (y += a*x)
#


saxpy_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(saxpy_sizes, desc="saxpy"):
    x = np.random.randn(n).astype(np.float32)
    y_orig = np.random.randn(n).astype(np.float32)
    a_val = np.float32(2.5)
    a_arr = np.array([a_val], dtype=np.float32)
    expected = y_orig + a_val * x
    flops = 2 * n  # n multiplies + n adds

    # numpy
    t_np = bench(lambda y=y_orig.copy(), a=a_val, xv=x: np.add(a * xv, y, out=y))

    # exo auto-vectorized
    fn_exo = saxpy(n)
    y_test = y_orig.copy()
    fn_exo(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy exo wrong"
    t_exo = bench(lambda fn=fn_exo, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    # exo neon intrinsics
    fn_neon = saxpy_neon(n)
    y_test = y_orig.copy()
    fn_neon(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy neon wrong"
    t_neon = bench(lambda fn=fn_neon, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    rows.append(
        {
            "kernel": "saxpy",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# softmax (fused: exp(x-max) + sum + normalize)
#


softmax_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(softmax_sizes, desc="softmax"):
    inp = np.random.randn(n).astype(np.float32)
    out_np = np.empty(n, dtype=np.float32)
    tmp_np = np.empty(n, dtype=np.float32)

    # numpy reference: max + sub + exp + sum + div (pre-allocated temporaries)
    def numpy_softmax(x=inp, out=out_np, tmp=tmp_np):
        m = x.max()
        np.subtract(x, m, out=tmp)
        np.exp(tmp, out=out)
        s = out.sum()
        out *= 1.0 / s

    numpy_softmax()
    expected = out_np.copy()
    flops = 4 * n  # sub + exp + sum + div (4 ops per element, standard counting)

    t_np = bench(numpy_softmax)

    # exo auto-vectorized (jit max + fused exp/sum/normalize)
    fn_max, fn_core = softmax(n)
    out_exo = np.zeros(n, dtype=np.float32)
    mx = np.array([0.0], dtype=np.float32)
    fn_max(mx, inp)
    fn_core(out_exo, inp, mx)
    assert np.allclose(out_exo, expected, atol=1e-3), f"softmax exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"

    def bench_exo(fn_m=fn_max, fn_c=fn_core, out=out_exo, x=inp, mx=mx):
        fn_m(mx, x)
        fn_c(out, x, mx)

    t_exo = bench(bench_exo)

    # exo neon intrinsics (neon max + fused exp/sum/normalize with explicit neon)
    fn_neon = softmax_neon(n)
    fn_max_neon = _jit_max_neon(n)
    out_neon = np.zeros(n, dtype=np.float32)
    fn_max_neon(mx, inp)
    fn_neon(out_neon, inp, mx)
    assert np.allclose(out_neon, expected, atol=1e-3), f"softmax neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"

    def bench_neon(fn_m=fn_max_neon, fn=fn_neon, out=out_neon, x=inp, mx=mx):
        fn_m(mx, x)
        fn(out, x, mx)

    t_neon = bench(bench_neon)

    rows.append(
        {
            "kernel": "softmax",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# relu (y = max(0, x))
#


relu_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(relu_sizes, desc="relu"):
    inp = np.random.randn(n).astype(np.float32)
    expected = np.maximum(0, inp)
    flops = n  # 1 comparison per element

    # numpy
    t_np = bench(lambda x=inp: np.maximum(0, x))

    # exo auto-vectorized
    fn_exo = relu(n)
    out_exo = np.empty(n, dtype=np.float32)
    fn_exo(out_exo, inp)
    assert np.allclose(out_exo, expected, atol=1e-6), f"relu exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, x=inp: fn(out, x))

    # exo neon intrinsics
    fn_neon = relu_neon(n)
    out_neon = np.empty(n, dtype=np.float32)
    fn_neon(out_neon, inp)
    assert np.allclose(out_neon, expected, atol=1e-6), f"relu neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, x=inp: fn(out, x))

    rows.append(
        {
            "kernel": "relu",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# add (z = x + y)
#


add_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(add_sizes, desc="add"):
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    expected = x + y
    flops = n  # 1 add per element

    # numpy
    t_np = bench(lambda x=x, y=y: np.add(x, y))

    # exo auto-vectorized
    fn_exo = add(n)
    z_exo = np.empty(n, dtype=np.float32)
    fn_exo(z_exo, x, y)
    assert np.allclose(z_exo, expected, atol=1e-6), f"add exo wrong: max_diff={np.max(np.abs(z_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, z=z_exo, x=x, y=y: fn(z, x, y))

    # exo neon intrinsics
    fn_neon = add_neon(n)
    z_neon = np.empty(n, dtype=np.float32)
    fn_neon(z_neon, x, y)
    assert np.allclose(z_neon, expected, atol=1e-6), f"add neon wrong: max_diff={np.max(np.abs(z_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, z=z_neon, x=x, y=y: fn(z, x, y))

    rows.append(
        {
            "kernel": "add",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# cross-entropy loss: loss = -log(softmax(logits)[target])
# numerically stable: loss = -logits[target] + max + log(sum(exp(logits - max)))
#


ce_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(ce_sizes, desc="cross_entropy"):
    logits = np.random.randn(n).astype(np.float32)
    target = np.random.randint(0, n)

    # numpy reference
    def numpy_ce(x=logits, t=target):
        m = x.max()
        return -x[t] + m + np.log(np.sum(np.exp(x - m)))

    expected = numpy_ce()
    flops = 4 * n  # max + sub + exp + sum

    t_np = bench(numpy_ce)

    # exo auto-vectorized
    fn_max, fn_sum_exp = cross_entropy(n)
    mx = np.array([0.0], dtype=np.float32)
    sum_exp = np.array([0.0], dtype=np.float32)
    fn_max(mx, logits)
    fn_sum_exp(sum_exp, logits, mx)
    loss_exo = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_exo, expected, atol=1e-3), f"cross_entropy exo wrong: {loss_exo} vs {expected}"

    def bench_exo(fn_m=fn_max, fn_se=fn_sum_exp, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn_se(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_exo = bench(bench_exo)

    # exo neon intrinsics
    fn_max_neon = _jit_max_neon(n)
    fn_sum_exp_neon = cross_entropy_neon(n)
    fn_max_neon(mx, logits)
    fn_sum_exp_neon(sum_exp, logits, mx)
    loss_neon = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_neon, expected, atol=1e-3), f"cross_entropy neon wrong: {loss_neon} vs {expected}"

    def bench_neon(fn_m=fn_max_neon, fn=fn_sum_exp_neon, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_neon = bench(bench_neon)

    rows.append(
        {
            "kernel": "cross_entropy",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# rmsnorm: y[i] = x[i] * rsqrt(mean(x^2) + eps)
#


rmsnorm_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]
EPS = np.float32(1e-5)


for n in tqdm(rmsnorm_sizes, desc="rmsnorm"):
    inp = np.random.randn(n).astype(np.float32)
    expected = inp / np.sqrt(np.mean(inp**2) + EPS)
    flops = 3 * n  # n squares + n sum-adds + n scale-muls

    # numpy (dot for sum-of-squares, pre-allocated output)
    out_np = np.empty(n, dtype=np.float32)

    def numpy_rmsnorm(x=inp, out=out_np, nn=n, eps=EPS):
        s = np.dot(x, x)
        np.multiply(x, np.float32(1.0 / np.sqrt(s / nn + eps)), out=out)

    numpy_rmsnorm()
    assert np.allclose(out_np, expected, atol=1e-3), "rmsnorm numpy wrong"
    t_np = bench(numpy_rmsnorm)

    # exo auto-vectorized (sumsq kernel + python sqrt + scale kernel)
    fn_sumsq, fn_scale = rmsnorm(n)
    sumsq = np.array([0.0], dtype=np.float32)
    scale_arr = np.array([0.0], dtype=np.float32)
    out_exo = np.empty(n, dtype=np.float32)

    def run_exo(fn_sq=fn_sumsq, fn_sc=fn_scale, sq=sumsq, sc=scale_arr, out=out_exo, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    run_exo()
    assert np.allclose(out_exo, expected, atol=1e-3), f"rmsnorm exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(run_exo)

    # exo neon intrinsics
    fn_sumsq_neon, fn_scale_neon = rmsnorm_neon(n)
    out_neon = np.empty(n, dtype=np.float32)

    def run_neon(fn_sq=fn_sumsq_neon, fn_sc=fn_scale_neon, sq=sumsq, sc=scale_arr, out=out_neon, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    run_neon()
    assert np.allclose(out_neon, expected, atol=1e-3), f"rmsnorm neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(run_neon)

    rows.append(
        {
            "kernel": "rmsnorm",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# embedding lookup: y[i] = table[index][i]
#


embed_dims = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]
VOCAB_SIZE = 256  # only one row is accessed; small vocab keeps memory reasonable


for d in tqdm(embed_dims, desc="embedding"):
    table = np.random.randn(VOCAB_SIZE, d).astype(np.float32)
    index = np.random.randint(0, VOCAB_SIZE)
    expected = table[index].copy()
    ops = d  # 1 load+store per element

    # numpy
    out_np = np.empty(d, dtype=np.float32)
    t_np = bench(lambda out=out_np, t=table, idx=index: np.copyto(out, t[idx]))

    # exo auto-vectorized
    fn_exo = embedding(d)
    out_exo = np.empty(d, dtype=np.float32)
    fn_exo(out_exo, table[index])
    assert np.allclose(out_exo, expected, atol=1e-6), "embedding exo wrong"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, row=table[index]: fn(out, row))

    # exo neon intrinsics
    fn_neon = embedding_neon(d)
    out_neon = np.empty(d, dtype=np.float32)
    fn_neon(out_neon, table[index])
    assert np.allclose(out_neon, expected, atol=1e-6), "embedding neon wrong"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, row=table[index]: fn(out, row))

    rows.append(
        {
            "kernel": "embedding",
            "n": d,
            "numpy_gflops": round(ops / t_np / 1e9, 2),
            "exo_gflops": round(ops / t_exo / 1e9, 2),
            "neon_gflops": round(ops / t_neon / 1e9, 2),
        }
    )


#
# scaled dot product: score = sum(q[j] * k[j]) / sqrt(d)
#


dot_sizes = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18]


for n in tqdm(dot_sizes, desc="dot"):
    q = np.random.randn(n).astype(np.float32)
    k = np.random.randn(n).astype(np.float32)
    inv_sqrt_d = np.float32(1.0 / np.sqrt(n))
    expected = np.dot(q, k) * inv_sqrt_d
    flops = 2 * n  # n multiplies + n adds

    # numpy
    t_np = bench(lambda q=q, k=k, s=inv_sqrt_d: np.dot(q, k) * s)

    # exo auto-vectorized
    fn_exo = dot(n)
    result_exo = np.array([0.0], dtype=np.float32)
    fn_exo(result_exo, q, k)
    score_exo = result_exo[0] * inv_sqrt_d
    assert np.allclose(score_exo, expected, atol=1e-2), f"dot exo wrong: {score_exo} vs {expected}"
    t_exo = bench(lambda fn=fn_exo, r=result_exo, q=q, k=k, s=inv_sqrt_d: fn(r, q, k))

    # exo neon intrinsics
    fn_neon = dot_neon(n)
    result_neon = np.array([0.0], dtype=np.float32)
    fn_neon(result_neon, q, k)
    score_neon = result_neon[0] * inv_sqrt_d
    assert np.allclose(score_neon, expected, atol=1e-2), f"dot neon wrong: {score_neon} vs {expected}"
    t_neon = bench(lambda fn=fn_neon, r=result_neon, q=q, k=k, s=inv_sqrt_d: fn(r, q, k))

    rows.append(
        {
            "kernel": "dot",
            "n": n,
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


#
# weighted sum (attention output): out[j] = sum_t weights[t] * v[t][j]
#


ws_sizes = [(64, 64), (64, 128), (128, 128), (128, 256), (256, 256)]


for t_size, d_size in tqdm(ws_sizes, desc="weighted_sum"):
    weights = np.random.randn(t_size).astype(np.float32)
    V = np.random.randn(t_size, d_size).astype(np.float32)
    expected = weights @ V
    flops = 2 * t_size * d_size  # t*d muls + t*d adds

    # numpy
    t_np = bench(lambda w=weights, v=V: w @ v)

    # exo auto-vectorized
    fn_exo = weighted_sum(t_size, d_size)
    out_exo = np.zeros(d_size, dtype=np.float32)
    fn_exo(out_exo, weights, V)
    assert np.allclose(out_exo, expected, atol=1e-3), f"weighted_sum exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, w=weights, v=V: fn(out, w, v))

    # exo neon intrinsics
    fn_neon = weighted_sum_neon(t_size, d_size)
    out_neon = np.zeros(d_size, dtype=np.float32)
    fn_neon(out_neon, weights, V)
    assert np.allclose(out_neon, expected, atol=1e-3), f"weighted_sum neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, w=weights, v=V: fn(out, w, v))

    rows.append(
        {
            "kernel": "weighted_sum",
            "n": f"{t_size}x{d_size}",
            "numpy_gflops": round(flops / t_np / 1e9, 2),
            "exo_gflops": round(flops / t_exo / 1e9, 2),
            "neon_gflops": round(flops / t_neon / 1e9, 2),
        }
    )


def _plot(df: pl.DataFrame) -> None:
    pdf = df.unpivot(on=["exo_speedup", "neon_speedup"], index=["kernel", "n"], variable_name="variant", value_name="speedup").with_columns(pl.col("variant").replace({"exo_speedup": "Auto-vectorized", "neon_speedup": "NEON intrinsics"}), pl.col("speedup").clip(lower_bound=1.0)).to_pandas()

    seen = []
    for v in pdf["n"]:
        if v not in seen:
            seen.append(v)
    pdf["n"] = pd.Categorical(pdf["n"], categories=seen, ordered=True)

    out = Path(__file__).parent / "plots"
    out.mkdir(exist_ok=True)
    # fmt: off
    p = (
        ggplot(pdf, aes("n", "speedup", color="variant", linetype="variant", group="variant"))
        + geom_hline(yintercept=1, linetype="solid", color="#bbbbbb", size=0.8)
        + geom_line(size=1.4)
        + geom_point(aes(shape="variant"), size=2.8)
        + facet_wrap("~kernel", scales="free")
        + scale_color_manual(values=["#4C72B0", "#DD8452"])
        + scale_linetype_manual(values=["solid", "dashed"])
        + scale_shape_manual(values=["o", "^"])
        + expand_limits(y=1)
        + theme_minimal()
        + theme(
            figure_size=(20, 13),
            legend_position="top",
            legend_title=element_text(size=0),
            legend_text=element_text(size=11),
            plot_title=element_text(size=15, weight="bold"),
            plot_subtitle=element_text(size=11, color="#555555"),
            axis_title=element_text(size=10),
            axis_text_x=element_text(rotation=45, ha="right", size=8),
            strip_text=element_text(size=11, weight="bold"),
            panel_grid_minor=element_line(color="#eeeeee", size=0.3),
            panel_grid_major=element_line(color="#dddddd", size=0.5),
            panel_border=element_rect(color="#cccccc", size=0.5),
        )
        + labs(
            title="xnumpy Kernel Performance vs NumPy",
            subtitle="Speedup over NumPy. Baseline = 1x (same as NumPy). Values below 1x are clamped to the baseline.",
            x="Problem Size (number of elements)",
            y="Speedup over NumPy (1x = baseline)",
            color="", linetype="", shape="",
        )
    )
    p.save(str(out / "convergence.pdf"))
    # fmt: on


if __name__ == "__main__":
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("exo_gflops") / pl.col("numpy_gflops")).round(2).alias("exo_speedup"),
        (pl.col("neon_gflops") / pl.col("numpy_gflops")).round(2).alias("neon_speedup"),
    )
    with pl.Config(tbl_rows=-1):
        print(df)
    df.write_csv(Path(__file__).parent / "results.csv")
    _plot(df)
