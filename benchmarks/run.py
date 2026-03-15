# choose kernels from:
#
# - https://karpathy.github.io/2026/02/12/microgpt/
# - https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from plotnine import aes, annotate, element_line, element_rect, element_text, expand_limits, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual, scale_linetype_manual, scale_shape_manual, theme, theme_minimal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from kernels.adam_exo import adam_exo
from kernels.adam_jax import adam_jax
from kernels.adam_neon import adam_neon
from kernels.adam_numba import adam_numba
from kernels.adam_numpy import adam_numpy
from kernels.adam_torch import adam_torch
from kernels.add_exo import add_exo
from kernels.add_jax import add_jax
from kernels.add_neon import add_neon
from kernels.add_numba import add_numba
from kernels.add_numpy import add_numpy
from kernels.add_torch import add_torch
from kernels.cross_entropy_exo import cross_entropy_exo
from kernels.cross_entropy_jax import cross_entropy_jax
from kernels.cross_entropy_neon import cross_entropy_neon
from kernels.cross_entropy_numba import cross_entropy_numba
from kernels.cross_entropy_numpy import cross_entropy_numpy
from kernels.cross_entropy_torch import cross_entropy_torch
from kernels.dot_exo import dot_exo
from kernels.dot_jax import dot_jax
from kernels.dot_neon import dot_neon
from kernels.dot_numba import dot_numba
from kernels.dot_numpy import dot_numpy
from kernels.dot_torch import dot_torch
from kernels.embedding_exo import embedding_exo
from kernels.embedding_jax import embedding_jax
from kernels.embedding_neon import embedding_neon
from kernels.embedding_numba import embedding_numba
from kernels.embedding_numpy import embedding_numpy
from kernels.embedding_torch import embedding_torch
from kernels.matmul_exo import matmul_exo
from kernels.matmul_jax import matmul_jax
from kernels.matmul_neon import matmul_neon
from kernels.matmul_numba import matmul_numba
from kernels.matmul_numpy import matmul_numpy
from kernels.matmul_torch import matmul_torch
from kernels.matvec_exo import matvec_exo
from kernels.matvec_jax import matvec_jax
from kernels.matvec_neon import matvec_neon
from kernels.matvec_numba import matvec_numba
from kernels.matvec_numpy import matvec_numpy
from kernels.matvec_torch import matvec_torch
from kernels.relu_exo import relu_exo
from kernels.relu_jax import relu_jax
from kernels.relu_neon import relu_neon
from kernels.relu_numba import relu_numba
from kernels.relu_numpy import relu_numpy
from kernels.relu_torch import relu_torch
from kernels.rmsnorm_exo import rmsnorm_exo
from kernels.rmsnorm_jax import rmsnorm_jax
from kernels.rmsnorm_neon import rmsnorm_neon
from kernels.rmsnorm_numba import rmsnorm_numba
from kernels.rmsnorm_numpy import rmsnorm_numpy
from kernels.rmsnorm_torch import rmsnorm_torch
from kernels.saxpy_exo import saxpy_exo
from kernels.saxpy_jax import saxpy_jax
from kernels.saxpy_neon import saxpy_neon
from kernels.saxpy_numba import saxpy_numba
from kernels.saxpy_numpy import saxpy_numpy
from kernels.saxpy_torch import saxpy_torch
from kernels.softmax_exo import _jit_max_neon, softmax_exo
from kernels.softmax_jax import softmax_jax
from kernels.softmax_neon import softmax_neon
from kernels.softmax_numba import softmax_numba
from kernels.softmax_numpy import softmax_numpy
from kernels.softmax_torch import softmax_torch
from kernels.weighted_sum_exo import weighted_sum_exo
from kernels.weighted_sum_jax import weighted_sum_jax
from kernels.weighted_sum_neon import weighted_sum_neon
from kernels.weighted_sum_numba import weighted_sum_numba
from kernels.weighted_sum_numpy import weighted_sum_numpy
from kernels.weighted_sum_torch import weighted_sum_torch

REPEATS = 50


bench = lambda fn: min(timeit.repeat(fn, number=1, repeat=REPEATS))

np.random.seed(42)
rows: list[dict[str, object]] = []


def _gflops(flops: int, t: float, precision: int = 2) -> float:
    return round(flops / t / 1e9, precision)


def run_kernel(name: str, sizes: list, bench_one, precision: int = 2) -> None:
    for size in tqdm(sizes, desc=name):
        n_label, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch = bench_one(size)
        rows.append(
            {
                "kernel": name,
                "n": n_label,
                "numpy_gflops": _gflops(flops, t_np, precision),
                "exo_gflops": _gflops(flops, t_exo, precision),
                "neon_gflops": _gflops(flops, t_neon, precision),
                "numba_gflops": _gflops(flops, t_numba, precision),
                "jax_gflops": _gflops(flops, t_jax, precision),
                "torch_gflops": _gflops(flops, t_torch, precision),
            }
        )


def bench_matmul(n):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    flops = 2 * n**3

    fn_np = matmul_numpy(n, n, n)
    expected = np.empty((n, n), dtype=np.float32)
    fn_np(expected, A, B)
    t_np = bench(lambda fn=fn_np, C=expected, A=A, B=B: fn(C, A, B))

    fn_exo = matmul_exo(n, n, n)
    C_exo = np.zeros((n, n), dtype=np.float32)
    fn_exo(C_exo, A, B)
    assert np.allclose(C_exo, expected, atol=1e-3)
    t_exo = bench(lambda C=C_exo, A=A, B=B: fn_exo(C, A, B))

    fn_neon = matmul_neon(n, n, n)
    C_neon = np.zeros((n, n), dtype=np.float32)
    fn_neon(C_neon, A, B)
    assert np.allclose(C_neon, expected, atol=1e-3)
    t_neon = bench(lambda C=C_neon, A=A, B=B: fn_neon(C, A, B))

    fn_numba = matmul_numba(n, n, n)
    C_numba = np.zeros((n, n), dtype=np.float32)
    fn_numba(C_numba, A, B)
    assert np.allclose(C_numba, expected, atol=1e-3)
    t_numba = bench(lambda C=C_numba, A=A, B=B: fn_numba(C, A, B))

    fn_jax = matmul_jax(n, n, n)
    C_jax = np.zeros((n, n), dtype=np.float32)
    fn_jax(C_jax, A, B)
    assert np.allclose(C_jax, expected, atol=1e-3), f"matmul jax wrong: max_diff={np.max(np.abs(C_jax - expected))}"
    t_jax = bench(lambda C=C_jax, A=A, B=B: fn_jax(C, A, B))

    fn_torch = matmul_torch(n, n, n)
    C_torch = np.zeros((n, n), dtype=np.float32)
    fn_torch(C_torch, A, B)
    assert np.allclose(C_torch, expected, atol=1e-3), f"matmul torch wrong: max_diff={np.max(np.abs(C_torch - expected))}"
    t_torch = bench(lambda C=C_torch, A=A, B=B: fn_torch(C, A, B))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("matmul", [1 << 4, 1 << 5, 1 << 6, 1 << 7, 1 << 8], bench_matmul, precision=1)


def bench_matvec(n):
    W = np.random.randn(n, n).astype(np.float32)
    x = np.random.randn(n).astype(np.float32)
    flops = 2 * n**2

    fn_np = matvec_numpy(n, n)
    expected = np.empty(n, dtype=np.float32)
    fn_np(expected, W, x)
    t_np = bench(lambda fn=fn_np, y=expected, W=W, x=x: fn(y, W, x))

    fn_exo = matvec_exo(n, n)
    y_exo = np.zeros(n, dtype=np.float32)
    fn_exo(y_exo, W, x)
    assert np.allclose(y_exo, expected, atol=1e-3), f"matvec exo wrong: max_diff={np.max(np.abs(y_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, y=y_exo, W=W, x=x: fn(y, W, x))

    WT = np.ascontiguousarray(W.T)
    fn_neon = matvec_neon(n, n)
    y_neon = np.zeros(n, dtype=np.float32)
    fn_neon(y_neon, WT, x)
    assert np.allclose(y_neon, expected, atol=1e-3), f"matvec neon wrong: max_diff={np.max(np.abs(y_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, y=y_neon, WT=WT, x=x: fn(y, WT, x))

    fn_numba = matvec_numba(n, n)
    y_numba = np.zeros(n, dtype=np.float32)
    fn_numba(y_numba, W, x)
    assert np.allclose(y_numba, expected, atol=1e-3), f"matvec numba wrong: max_diff={np.max(np.abs(y_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, y=y_numba, W=W, x=x: fn(y, W, x))

    fn_jax = matvec_jax(n, n)
    y_jax = np.zeros(n, dtype=np.float32)
    fn_jax(y_jax, W, x)
    assert np.allclose(y_jax, expected, atol=1e-3), f"matvec jax wrong: max_diff={np.max(np.abs(y_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, y=y_jax, W=W, x=x: fn(y, W, x))

    fn_torch = matvec_torch(n, n)
    y_torch = np.zeros(n, dtype=np.float32)
    fn_torch(y_torch, W, x)
    assert np.allclose(y_torch, expected, atol=1e-3), f"matvec torch wrong: max_diff={np.max(np.abs(y_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, y=y_torch, W=W, x=x: fn(y, W, x))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("matvec", [1 << 5, 1 << 7, 1 << 9, 1 << 11], bench_matvec, precision=1)


def bench_saxpy(n):
    x = np.random.randn(n).astype(np.float32)
    y_orig = np.random.randn(n).astype(np.float32)
    a_val = np.float32(2.5)
    a_arr = np.array([a_val], dtype=np.float32)
    expected = y_orig + a_val * x
    flops = 2 * n

    fn_np = saxpy_numpy(n)
    t_np = bench(lambda fn=fn_np, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    fn_exo = saxpy_exo(n)
    y_test = y_orig.copy()
    fn_exo(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy exo wrong"
    t_exo = bench(lambda fn=fn_exo, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    fn_neon = saxpy_neon(n)
    y_test = y_orig.copy()
    fn_neon(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy neon wrong"
    t_neon = bench(lambda fn=fn_neon, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    fn_numba = saxpy_numba(n)
    y_test = y_orig.copy()
    fn_numba(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy numba wrong"
    t_numba = bench(lambda fn=fn_numba, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    fn_jax = saxpy_jax(n)
    y_test = y_orig.copy()
    fn_jax(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy jax wrong"
    t_jax = bench(lambda fn=fn_jax, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    fn_torch = saxpy_torch(n)
    y_test = y_orig.copy()
    fn_torch(y_test, x, a_arr)
    assert np.allclose(y_test, expected, atol=1e-4), "saxpy torch wrong"
    t_torch = bench(lambda fn=fn_torch, y=y_orig.copy(), x=x, a=a_arr: fn(y, x, a))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("saxpy", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_saxpy)


def bench_softmax(n):
    inp = np.random.randn(n).astype(np.float32)
    flops = 4 * n

    fn_np = softmax_numpy(n)
    expected = np.empty(n, dtype=np.float32)
    fn_np(expected, inp)
    t_np = bench(lambda fn=fn_np, out=expected, x=inp: fn(out, x))

    fn_max, fn_core = softmax_exo(n)
    out_exo = np.zeros(n, dtype=np.float32)
    mx = np.array([0.0], dtype=np.float32)
    fn_max(mx, inp)
    fn_core(out_exo, inp, mx)
    assert np.allclose(out_exo, expected, atol=1e-3), f"softmax exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"

    def _bench_exo(fn_m=fn_max, fn_c=fn_core, out=out_exo, x=inp, mx=mx):
        fn_m(mx, x)
        fn_c(out, x, mx)

    t_exo = bench(_bench_exo)

    fn_neon = softmax_neon(n)
    fn_max_neon = _jit_max_neon(n)
    out_neon = np.zeros(n, dtype=np.float32)
    fn_max_neon(mx, inp)
    fn_neon(out_neon, inp, mx)
    assert np.allclose(out_neon, expected, atol=1e-3), f"softmax neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"

    def _bench_neon(fn_m=fn_max_neon, fn=fn_neon, out=out_neon, x=inp, mx=mx):
        fn_m(mx, x)
        fn(out, x, mx)

    t_neon = bench(_bench_neon)

    fn_numba = softmax_numba(n)
    out_numba = np.zeros(n, dtype=np.float32)
    fn_numba(out_numba, inp)
    assert np.allclose(out_numba, expected, atol=1e-3), f"softmax numba wrong: max_diff={np.max(np.abs(out_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, out=out_numba, x=inp: fn(out, x))

    fn_jax = softmax_jax(n)
    out_jax = np.zeros(n, dtype=np.float32)
    fn_jax(out_jax, inp)
    assert np.allclose(out_jax, expected, atol=1e-3), f"softmax jax wrong: max_diff={np.max(np.abs(out_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, out=out_jax, x=inp: fn(out, x))

    fn_torch = softmax_torch(n)
    out_torch = np.zeros(n, dtype=np.float32)
    fn_torch(out_torch, inp)
    assert np.allclose(out_torch, expected, atol=1e-3), f"softmax torch wrong: max_diff={np.max(np.abs(out_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, out=out_torch, x=inp: fn(out, x))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("softmax", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_softmax)


def bench_relu(n):
    inp = np.random.randn(n).astype(np.float32)
    flops = n

    fn_np = relu_numpy(n)
    expected = np.empty(n, dtype=np.float32)
    fn_np(expected, inp)
    t_np = bench(lambda fn=fn_np, out=expected, x=inp: fn(out, x))

    fn_exo = relu_exo(n)
    out_exo = np.empty(n, dtype=np.float32)
    fn_exo(out_exo, inp)
    assert np.allclose(out_exo, expected, atol=1e-6), f"relu exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, x=inp: fn(out, x))

    fn_neon = relu_neon(n)
    out_neon = np.empty(n, dtype=np.float32)
    fn_neon(out_neon, inp)
    assert np.allclose(out_neon, expected, atol=1e-6), f"relu neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, x=inp: fn(out, x))

    fn_numba = relu_numba(n)
    out_numba = np.empty(n, dtype=np.float32)
    fn_numba(out_numba, inp)
    assert np.allclose(out_numba, expected, atol=1e-6), f"relu numba wrong: max_diff={np.max(np.abs(out_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, out=out_numba, x=inp: fn(out, x))

    fn_jax = relu_jax(n)
    out_jax = np.empty(n, dtype=np.float32)
    fn_jax(out_jax, inp)
    assert np.allclose(out_jax, expected, atol=1e-6), f"relu jax wrong: max_diff={np.max(np.abs(out_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, out=out_jax, x=inp: fn(out, x))

    fn_torch = relu_torch(n)
    out_torch = np.empty(n, dtype=np.float32)
    fn_torch(out_torch, inp)
    assert np.allclose(out_torch, expected, atol=1e-6), f"relu torch wrong: max_diff={np.max(np.abs(out_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, out=out_torch, x=inp: fn(out, x))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("relu", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_relu)


def bench_add(n):
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    flops = n

    fn_np = add_numpy(n)
    expected = np.empty(n, dtype=np.float32)
    fn_np(expected, x, y)
    t_np = bench(lambda fn=fn_np, z=expected, x=x, y=y: fn(z, x, y))

    fn_exo = add_exo(n)
    z_exo = np.empty(n, dtype=np.float32)
    fn_exo(z_exo, x, y)
    assert np.allclose(z_exo, expected, atol=1e-6), f"add exo wrong: max_diff={np.max(np.abs(z_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, z=z_exo, x=x, y=y: fn(z, x, y))

    fn_neon = add_neon(n)
    z_neon = np.empty(n, dtype=np.float32)
    fn_neon(z_neon, x, y)
    assert np.allclose(z_neon, expected, atol=1e-6), f"add neon wrong: max_diff={np.max(np.abs(z_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, z=z_neon, x=x, y=y: fn(z, x, y))

    fn_numba = add_numba(n)
    z_numba = np.empty(n, dtype=np.float32)
    fn_numba(z_numba, x, y)
    assert np.allclose(z_numba, expected, atol=1e-6), f"add numba wrong: max_diff={np.max(np.abs(z_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, z=z_numba, x=x, y=y: fn(z, x, y))

    fn_jax = add_jax(n)
    z_jax = np.empty(n, dtype=np.float32)
    fn_jax(z_jax, x, y)
    assert np.allclose(z_jax, expected, atol=1e-6), f"add jax wrong: max_diff={np.max(np.abs(z_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, z=z_jax, x=x, y=y: fn(z, x, y))

    fn_torch = add_torch(n)
    z_torch = np.empty(n, dtype=np.float32)
    fn_torch(z_torch, x, y)
    assert np.allclose(z_torch, expected, atol=1e-6), f"add torch wrong: max_diff={np.max(np.abs(z_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, z=z_torch, x=x, y=y: fn(z, x, y))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("add", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_add)


def bench_cross_entropy(n):
    logits = np.random.randn(n).astype(np.float32)
    target = np.random.randint(0, n)
    flops = 4 * n

    fn_max_np, fn_sum_exp_np = cross_entropy_numpy(n)
    mx_np = np.array([0.0], dtype=np.float32)
    se_np = np.array([0.0], dtype=np.float32)
    fn_max_np(mx_np, logits)
    fn_sum_exp_np(se_np, logits, mx_np)
    expected = -logits[target] + mx_np[0] + np.log(se_np[0])

    def _bench_np(fn_m=fn_max_np, fn_se=fn_sum_exp_np, x=logits, mx=mx_np, se=se_np, t=target):
        fn_m(mx, x)
        fn_se(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_np = bench(_bench_np)

    fn_max, fn_sum_exp = cross_entropy_exo(n)
    mx = np.array([0.0], dtype=np.float32)
    sum_exp = np.array([0.0], dtype=np.float32)
    fn_max(mx, logits)
    fn_sum_exp(sum_exp, logits, mx)
    loss_exo = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_exo, expected, atol=1e-3), f"cross_entropy exo wrong: {loss_exo} vs {expected}"

    def _bench_exo(fn_m=fn_max, fn_se=fn_sum_exp, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn_se(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_exo = bench(_bench_exo)

    fn_max_neon = _jit_max_neon(n)
    fn_sum_exp_neon = cross_entropy_neon(n)
    fn_max_neon(mx, logits)
    fn_sum_exp_neon(sum_exp, logits, mx)
    loss_neon = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_neon, expected, atol=1e-3), f"cross_entropy neon wrong: {loss_neon} vs {expected}"

    def _bench_neon(fn_m=fn_max_neon, fn=fn_sum_exp_neon, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_neon = bench(_bench_neon)

    fn_max_numba, fn_sum_exp_numba = cross_entropy_numba(n)
    fn_max_numba(mx, logits)
    fn_sum_exp_numba(sum_exp, logits, mx)
    loss_numba = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_numba, expected, atol=1e-3), f"cross_entropy numba wrong: {loss_numba} vs {expected}"

    def _bench_numba(fn_m=fn_max_numba, fn=fn_sum_exp_numba, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_numba = bench(_bench_numba)

    fn_max_jax, fn_sum_exp_jax = cross_entropy_jax(n)
    fn_max_jax(mx, logits)
    fn_sum_exp_jax(sum_exp, logits, mx)
    loss_jax = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_jax, expected, atol=1e-3), f"cross_entropy jax wrong: {loss_jax} vs {expected}"

    def _bench_jax(fn_m=fn_max_jax, fn=fn_sum_exp_jax, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_jax = bench(_bench_jax)

    fn_max_torch, fn_sum_exp_torch = cross_entropy_torch(n)
    fn_max_torch(mx, logits)
    fn_sum_exp_torch(sum_exp, logits, mx)
    loss_torch = -logits[target] + mx[0] + np.log(sum_exp[0])
    assert np.allclose(loss_torch, expected, atol=1e-3), f"cross_entropy torch wrong: {loss_torch} vs {expected}"

    def _bench_torch(fn_m=fn_max_torch, fn=fn_sum_exp_torch, x=logits, mx=mx, se=sum_exp, t=target):
        fn_m(mx, x)
        fn(se, x, mx)
        return -x[t] + mx[0] + np.log(se[0])

    t_torch = bench(_bench_torch)

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("cross_entropy", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_cross_entropy)


EPS = np.float32(1e-5)


def bench_rmsnorm(n):
    inp = np.random.randn(n).astype(np.float32)
    expected = inp / np.sqrt(np.mean(inp**2) + EPS)
    flops = 3 * n

    fn_sumsq_np, fn_scale_np = rmsnorm_numpy(n)
    sumsq_np = np.array([0.0], dtype=np.float32)
    scale_np = np.array([0.0], dtype=np.float32)
    out_np = np.empty(n, dtype=np.float32)

    def _run_np(fn_sq=fn_sumsq_np, fn_sc=fn_scale_np, sq=sumsq_np, sc=scale_np, out=out_np, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_np()
    assert np.allclose(out_np, expected, atol=1e-3), "rmsnorm numpy wrong"
    t_np = bench(_run_np)

    fn_sumsq, fn_scale = rmsnorm_exo(n)
    sumsq = np.array([0.0], dtype=np.float32)
    scale_arr = np.array([0.0], dtype=np.float32)
    out_exo = np.empty(n, dtype=np.float32)

    def _run_exo(fn_sq=fn_sumsq, fn_sc=fn_scale, sq=sumsq, sc=scale_arr, out=out_exo, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_exo()
    assert np.allclose(out_exo, expected, atol=1e-3), f"rmsnorm exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(_run_exo)

    fn_sumsq_neon, fn_scale_neon = rmsnorm_neon(n)
    out_neon = np.empty(n, dtype=np.float32)

    def _run_neon(fn_sq=fn_sumsq_neon, fn_sc=fn_scale_neon, sq=sumsq, sc=scale_arr, out=out_neon, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_neon()
    assert np.allclose(out_neon, expected, atol=1e-3), f"rmsnorm neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(_run_neon)

    fn_sumsq_numba, fn_scale_numba = rmsnorm_numba(n)
    out_numba = np.empty(n, dtype=np.float32)

    def _run_numba(fn_sq=fn_sumsq_numba, fn_sc=fn_scale_numba, sq=sumsq, sc=scale_arr, out=out_numba, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_numba()
    assert np.allclose(out_numba, expected, atol=1e-3), f"rmsnorm numba wrong: max_diff={np.max(np.abs(out_numba - expected))}"
    t_numba = bench(_run_numba)

    fn_sumsq_jax, fn_scale_jax = rmsnorm_jax(n)
    out_jax = np.empty(n, dtype=np.float32)

    def _run_jax(fn_sq=fn_sumsq_jax, fn_sc=fn_scale_jax, sq=sumsq, sc=scale_arr, out=out_jax, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_jax()
    assert np.allclose(out_jax, expected, atol=2e-3), f"rmsnorm jax wrong: max_diff={np.max(np.abs(out_jax - expected))}"
    t_jax = bench(_run_jax)

    fn_sumsq_torch, fn_scale_torch = rmsnorm_torch(n)
    out_torch = np.empty(n, dtype=np.float32)

    def _run_torch(fn_sq=fn_sumsq_torch, fn_sc=fn_scale_torch, sq=sumsq, sc=scale_arr, out=out_torch, x=inp, nn=n, eps=EPS):
        fn_sq(sq, x)
        sc[0] = np.float32(1.0 / np.sqrt(sq[0] / nn + eps))
        fn_sc(out, x, sc)

    _run_torch()
    assert np.allclose(out_torch, expected, atol=2e-3), f"rmsnorm torch wrong: max_diff={np.max(np.abs(out_torch - expected))}"
    t_torch = bench(_run_torch)

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("rmsnorm", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_rmsnorm)


VOCAB_SIZE = 16


def bench_embedding(d):
    table = np.random.randn(VOCAB_SIZE, d).astype(np.float32)
    index = np.random.randint(0, VOCAB_SIZE)
    ops = d

    fn_np = embedding_numpy(d)
    expected = np.empty(d, dtype=np.float32)
    fn_np(expected, table[index])
    t_np = bench(lambda fn=fn_np, out=expected, row=table[index]: fn(out, row))

    fn_exo = embedding_exo(d)
    out_exo = np.empty(d, dtype=np.float32)
    fn_exo(out_exo, table[index])
    assert np.allclose(out_exo, expected, atol=1e-6), "embedding exo wrong"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, row=table[index]: fn(out, row))

    fn_neon = embedding_neon(d)
    out_neon = np.empty(d, dtype=np.float32)
    fn_neon(out_neon, table[index])
    assert np.allclose(out_neon, expected, atol=1e-6), "embedding neon wrong"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, row=table[index]: fn(out, row))

    fn_numba = embedding_numba(d)
    out_numba = np.empty(d, dtype=np.float32)
    fn_numba(out_numba, table[index])
    assert np.allclose(out_numba, expected, atol=1e-6), "embedding numba wrong"
    t_numba = bench(lambda fn=fn_numba, out=out_numba, row=table[index]: fn(out, row))

    fn_jax = embedding_jax(d)
    out_jax = np.empty(d, dtype=np.float32)
    fn_jax(out_jax, table[index])
    assert np.allclose(out_jax, expected, atol=1e-6), "embedding jax wrong"
    t_jax = bench(lambda fn=fn_jax, out=out_jax, row=table[index]: fn(out, row))

    fn_torch = embedding_torch(d)
    out_torch = np.empty(d, dtype=np.float32)
    fn_torch(out_torch, table[index])
    assert np.allclose(out_torch, expected, atol=1e-6), "embedding torch wrong"
    t_torch = bench(lambda fn=fn_torch, out=out_torch, row=table[index]: fn(out, row))

    return d, ops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("embedding", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_embedding)


def bench_dot(n):
    q = np.random.randn(n).astype(np.float32)
    k = np.random.randn(n).astype(np.float32)
    inv_sqrt_d = np.float32(1.0 / np.sqrt(n))
    flops = 2 * n

    fn_np = dot_numpy(n)
    result_np = np.array([0.0], dtype=np.float32)
    fn_np(result_np, q, k)
    expected = result_np[0] * inv_sqrt_d
    t_np = bench(lambda fn=fn_np, r=result_np, q=q, k=k: fn(r, q, k))

    fn_exo = dot_exo(n)
    result_exo = np.array([0.0], dtype=np.float32)
    fn_exo(result_exo, q, k)
    score_exo = result_exo[0] * inv_sqrt_d
    assert np.allclose(score_exo, expected, atol=1e-2), f"dot exo wrong: {score_exo} vs {expected}"
    t_exo = bench(lambda fn=fn_exo, r=result_exo, q=q, k=k, s=inv_sqrt_d: fn(r, q, k))

    fn_neon = dot_neon(n)
    result_neon = np.array([0.0], dtype=np.float32)
    fn_neon(result_neon, q, k)
    score_neon = result_neon[0] * inv_sqrt_d
    assert np.allclose(score_neon, expected, atol=1e-2), f"dot neon wrong: {score_neon} vs {expected}"
    t_neon = bench(lambda fn=fn_neon, r=result_neon, q=q, k=k, s=inv_sqrt_d: fn(r, q, k))

    fn_numba = dot_numba(n)
    result_numba = np.array([0.0], dtype=np.float32)
    fn_numba(result_numba, q, k)
    score_numba = result_numba[0] * inv_sqrt_d
    assert np.allclose(score_numba, expected, atol=1e-2), f"dot numba wrong: {score_numba} vs {expected}"
    t_numba = bench(lambda fn=fn_numba, r=result_numba, q=q, k=k: fn(r, q, k))

    fn_jax = dot_jax(n)
    result_jax = np.array([0.0], dtype=np.float32)
    fn_jax(result_jax, q, k)
    score_jax = result_jax[0] * inv_sqrt_d
    assert np.allclose(score_jax, expected, atol=1e-2), f"dot jax wrong: {score_jax} vs {expected}"
    t_jax = bench(lambda fn=fn_jax, r=result_jax, q=q, k=k: fn(r, q, k))

    fn_torch = dot_torch(n)
    result_torch = np.array([0.0], dtype=np.float32)
    fn_torch(result_torch, q, k)
    score_torch = result_torch[0] * inv_sqrt_d
    assert np.allclose(score_torch, expected, atol=1e-2), f"dot torch wrong: {score_torch} vs {expected}"
    t_torch = bench(lambda fn=fn_torch, r=result_torch, q=q, k=k: fn(r, q, k))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("dot", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_dot)


def bench_weighted_sum(size):
    t_size, d_size = size
    weights = np.random.randn(t_size).astype(np.float32)
    V = np.random.randn(t_size, d_size).astype(np.float32)
    flops = 2 * t_size * d_size

    fn_np = weighted_sum_numpy(t_size, d_size)
    expected = np.empty(d_size, dtype=np.float32)
    fn_np(expected, weights, V)
    t_np = bench(lambda fn=fn_np, out=expected, w=weights, v=V: fn(out, w, v))

    fn_exo = weighted_sum_exo(t_size, d_size)
    out_exo = np.zeros(d_size, dtype=np.float32)
    fn_exo(out_exo, weights, V)
    assert np.allclose(out_exo, expected, atol=1e-3), f"weighted_sum exo wrong: max_diff={np.max(np.abs(out_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, out=out_exo, w=weights, v=V: fn(out, w, v))

    fn_neon = weighted_sum_neon(t_size, d_size)
    out_neon = np.zeros(d_size, dtype=np.float32)
    fn_neon(out_neon, weights, V)
    assert np.allclose(out_neon, expected, atol=1e-3), f"weighted_sum neon wrong: max_diff={np.max(np.abs(out_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, out=out_neon, w=weights, v=V: fn(out, w, v))

    fn_numba = weighted_sum_numba(t_size, d_size)
    out_numba = np.zeros(d_size, dtype=np.float32)
    fn_numba(out_numba, weights, V)
    assert np.allclose(out_numba, expected, atol=1e-3), f"weighted_sum numba wrong: max_diff={np.max(np.abs(out_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, out=out_numba, w=weights, v=V: fn(out, w, v))

    fn_jax = weighted_sum_jax(t_size, d_size)
    out_jax = np.zeros(d_size, dtype=np.float32)
    fn_jax(out_jax, weights, V)
    assert np.allclose(out_jax, expected, atol=1e-3), f"weighted_sum jax wrong: max_diff={np.max(np.abs(out_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, out=out_jax, w=weights, v=V: fn(out, w, v))

    fn_torch = weighted_sum_torch(t_size, d_size)
    out_torch = np.zeros(d_size, dtype=np.float32)
    fn_torch(out_torch, weights, V)
    assert np.allclose(out_torch, expected, atol=1e-3), f"weighted_sum torch wrong: max_diff={np.max(np.abs(out_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, out=out_torch, w=weights, v=V: fn(out, w, v))

    return f"{t_size}x{d_size}", flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("weighted_sum", [(32, 64), (64, 128), (128, 256), (256, 512)], bench_weighted_sum)


B1 = np.float32(0.9)
B2 = np.float32(0.999)
ADAM_EPS = np.float32(1e-8)
LR = np.float32(0.001)
STEP = 10

b1_arr = np.array([B1], dtype=np.float32)
b2_arr = np.array([B2], dtype=np.float32)
eps_arr = np.array([ADAM_EPS], dtype=np.float32)
lr_arr = np.array([LR], dtype=np.float32)
beta1_t_arr = np.array([1.0 - B1**STEP], dtype=np.float32)
beta2_t_arr = np.array([1.0 - B2**STEP], dtype=np.float32)


def bench_adam(n):
    param_orig = np.random.randn(n).astype(np.float32)
    grad = np.random.randn(n).astype(np.float32)
    m_orig = np.random.randn(n).astype(np.float32) * 0.1
    v_orig = np.abs(np.random.randn(n).astype(np.float32)) * 0.01
    flops = 10 * n

    fn_np = adam_numpy(n)

    p_exp = param_orig.copy()
    m_exp = m_orig.copy()
    v_exp = v_orig.copy()
    fn_np(p_exp, grad, m_exp, v_exp, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    expected = p_exp

    t_np = bench(lambda fn=fn_np, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    fn_exo = adam_exo(n)
    p_exo = param_orig.copy()
    m_exo = m_orig.copy()
    v_exo = v_orig.copy()
    fn_exo(p_exo, grad, m_exo, v_exo, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    assert np.allclose(p_exo, expected, atol=1e-3), f"adam exo wrong: max_diff={np.max(np.abs(p_exo - expected))}"
    t_exo = bench(lambda fn=fn_exo, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    fn_neon = adam_neon(n)
    p_neon = param_orig.copy()
    m_neon = m_orig.copy()
    v_neon = v_orig.copy()
    fn_neon(p_neon, grad, m_neon, v_neon, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    assert np.allclose(p_neon, expected, atol=1e-3), f"adam neon wrong: max_diff={np.max(np.abs(p_neon - expected))}"
    t_neon = bench(lambda fn=fn_neon, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    fn_numba = adam_numba(n)
    p_numba = param_orig.copy()
    m_numba = m_orig.copy()
    v_numba = v_orig.copy()
    fn_numba(p_numba, grad, m_numba, v_numba, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    assert np.allclose(p_numba, expected, atol=1e-3), f"adam numba wrong: max_diff={np.max(np.abs(p_numba - expected))}"
    t_numba = bench(lambda fn=fn_numba, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    fn_jax = adam_jax(n)
    p_jax = param_orig.copy()
    m_jax = m_orig.copy()
    v_jax = v_orig.copy()
    fn_jax(p_jax, grad, m_jax, v_jax, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    assert np.allclose(p_jax, expected, atol=1e-3), f"adam jax wrong: max_diff={np.max(np.abs(p_jax - expected))}"
    t_jax = bench(lambda fn=fn_jax, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    fn_torch = adam_torch(n)
    p_torch = param_orig.copy()
    m_torch = m_orig.copy()
    v_torch = v_orig.copy()
    fn_torch(p_torch, grad, m_torch, v_torch, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr)
    assert np.allclose(p_torch, expected, atol=1e-3), f"adam torch wrong: max_diff={np.max(np.abs(p_torch - expected))}"
    t_torch = bench(lambda fn=fn_torch, p=param_orig.copy(), g=grad, mm=m_orig.copy(), vv=v_orig.copy(): fn(p, g, mm, vv, b1_arr, b2_arr, eps_arr, lr_arr, beta1_t_arr, beta2_t_arr))

    return n, flops, t_np, t_exo, t_neon, t_numba, t_jax, t_torch


run_kernel("adam", [1 << 8, 1 << 12, 1 << 16, 1 << 18, 1 << 20], bench_adam)


def _plot(df: pl.DataFrame) -> None:
    long = df.unpivot(on=["exo_speedup", "neon_speedup", "numba_speedup", "jax_speedup", "torch_speedup"], index=["kernel", "n"], variable_name="variant", value_name="speedup").with_columns(pl.col("variant").replace({"exo_speedup": "Auto-vectorized", "neon_speedup": "NEON intrinsics", "numba_speedup": "Numba JIT", "jax_speedup": "JAX JIT", "torch_speedup": "torch.compile"}))

    last_n = df.group_by("kernel").agg(pl.col("n").last().alias("last_n"))
    best = df.join(last_n, on="kernel").filter(pl.col("n") == pl.col("last_n")).with_columns(pl.max_horizontal("exo_speedup", "neon_speedup", "numba_speedup", "jax_speedup", "torch_speedup").alias("best")).sort("best", descending=True)
    kernel_order = best["kernel"].to_list()

    pdf = long.to_pandas()

    seen = []
    for v in pdf["n"]:
        if v not in seen:
            seen.append(v)
    pdf["n"] = pd.Categorical(pdf["n"], categories=seen, ordered=True)
    pdf["kernel"] = pd.Categorical(pdf["kernel"], categories=kernel_order, ordered=True)
    variant_order = ["Auto-vectorized", "NEON intrinsics", "Numba JIT", "JAX JIT", "torch.compile"]
    pdf["variant"] = pd.Categorical(pdf["variant"], categories=variant_order, ordered=True)

    out = Path(__file__).parent / "plots"
    out.mkdir(exist_ok=True)
    n_kernels = len(kernel_order)
    # fmt: off
    p = (
        ggplot(pdf, aes("n", "speedup", color="variant", linetype="variant", group="variant"))
        + annotate("rect", xmin=float("-inf"), xmax=float("inf"), ymin=float("-inf"), ymax=1, fill="#e74c3c", alpha=0.07)
        + geom_hline(yintercept=1, linetype="solid", color="#e74c3c", size=0.8)
        + geom_line(size=1.4)
        + geom_point(aes(shape="variant"), size=2.8)
        + facet_wrap("~kernel", scales="free", ncol=1)
        + scale_color_manual(values=["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3"])
        + scale_linetype_manual(values=["solid", "solid", "dotted", "dotted", "dotted"])
        + scale_shape_manual(values=["o", "D", "s", "^", "v"])
        + expand_limits(y=1)
        + theme_minimal()
        + theme(
            figure_size=(8, 5 * n_kernels),
            panel_spacing_y=0.15,
            legend_position="top",
            legend_title=element_text(size=0),
            legend_text=element_text(size=13),
            plot_title=element_text(size=18, weight="bold", margin={"b": 0}),
            axis_title=element_text(size=13),
            axis_text_x=element_text(rotation=45, ha="right", size=11),
            axis_text_y=element_text(size=11),
            strip_text=element_text(size=13, weight="bold"),
            panel_grid_minor=element_line(color="#eeeeee", size=0.3),
            panel_grid_major=element_line(color="#dddddd", size=0.5),
            panel_border=element_rect(color="#cccccc", size=0.5),
        )
        + labs(
            title="exojit vs NumPy",
            x="Problem size",
            y="Speedup (x)",
            color="", linetype="", shape="",
        )
    )
    p.save(str(out / "convergence.pdf"), limitsize=False)
    # fmt: on


if __name__ == "__main__":
    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("exo_gflops") / pl.col("numpy_gflops")).round(2).alias("exo_speedup"),
        (pl.col("neon_gflops") / pl.col("numpy_gflops")).round(2).alias("neon_speedup"),
        (pl.col("numba_gflops") / pl.col("numpy_gflops")).round(2).alias("numba_speedup"),
        (pl.col("jax_gflops") / pl.col("numpy_gflops")).round(2).alias("jax_speedup"),
        (pl.col("torch_gflops") / pl.col("numpy_gflops")).round(2).alias("torch_speedup"),
    )
    with pl.Config(tbl_rows=-1):
        print(df)
    df.write_csv(Path(__file__).parent / "results.csv")
    _plot(df)
