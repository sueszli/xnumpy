# /// script
# requires-python = ">=3.11"
# dependencies = ["xdsl-exo"]
# [tool.uv.sources]
# xdsl-exo = { path = "../.." }
# ///

from __future__ import annotations

import ctypes
import time

import numpy as np
from exo import *
from exo.stdlib.scheduling import *

from xdsl_exo.main import compile_procs
from xdsl_exo.patches_llvmlite import jit_compile

N = 256
FLOPS = 2 * N * N * N
WARMUP = 3
REPEATS = 20


@proc
def v0_naive(C: f32[256, 256] @ DRAM, A: f32[256, 256] @ DRAM, B: f32[256, 256] @ DRAM):
    for i in seq(0, 256):
        for j in seq(0, 256):
            C[i, j] = 0.0
            for k in seq(0, 256):
                C[i, j] += A[i, k] * B[k, j]


v1_reorder = rename(v0_naive, "v1_reorder")
v1_reorder = fission(v1_reorder, v1_reorder.find("for k in _: _").before(), n_lifts=2)
v1_reorder = reorder_loops(v1_reorder, "j k")

v2_k_tile = rename(v0_naive, "v2_k_tile")
v2_k_tile = fission(v2_k_tile, v2_k_tile.find("for k in _: _").before(), n_lifts=2)
v2_k_tile = reorder_loops(v2_k_tile, "j k")
v2_k_tile = divide_loop(v2_k_tile, "k", 32, ["ko", "ki"], perfect=True)

v3_unroll = rename(v0_naive, "v3_unroll")
v3_unroll = fission(v3_unroll, v3_unroll.find("for k in _: _").before(), n_lifts=2)
v3_unroll = reorder_loops(v3_unroll, "j k")
v3_unroll = divide_loop(v3_unroll, "j #1", 4, ["jo", "ji"], perfect=True)
v3_unroll = unroll_loop(v3_unroll, "ji")


@proc
def v4_2d_tile(C: f32[256, 256] @ DRAM, A: f32[256, 256] @ DRAM, B: f32[256, 256] @ DRAM):
    for io in seq(0, 8):
        for jo in seq(0, 8):
            for ii in seq(0, 32):
                for ji in seq(0, 32):
                    C[32 * io + ii, 32 * jo + ji] = 0.0
            for ko in seq(0, 8):
                for ii in seq(0, 32):
                    for ki in seq(0, 32):
                        for ji in seq(0, 32):
                            C[32 * io + ii, 32 * jo + ji] += A[32 * io + ii, 32 * ko + ki] * B[32 * ko + ki, 32 * jo + ji]


KERNELS = [v0_naive, v1_reorder, v2_k_tile, v3_unroll, v4_2d_tile]

_FN_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
_cache: dict[str, ctypes.CFUNCTYPE] = {}


def _get_fn(p):
    name = p.name()
    if name not in _cache:
        engine = jit_compile(compile_procs([p]))
        fn = _FN_TYPE(engine.get_function_address(name))
        fn._engine = engine
        _cache[name] = fn
    return _cache[name]


def jit_matmul(p, A, B):
    C = np.zeros((N, N), dtype=np.float32)
    _get_fn(p)(C.ctypes.data, A.ctypes.data, B.ctypes.data)
    return C


def python_matmul(A, B):
    C = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for k in range(N):
            a_ik = A[i][k]
            for j in range(N):
                C[i][j] += a_ik * B[k][j]
    return C


def bench(label, fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


np.random.seed(42)
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
expected = A @ B

for k in KERNELS:
    assert np.allclose(jit_matmul(k, A, B), expected, atol=0.5), f"{k.name()} wrong"
    bench(k.name(), lambda k=k: jit_matmul(k, A, B))

t0 = time.perf_counter()
C_py = python_matmul(A.tolist(), B.tolist())
t_py = time.perf_counter() - t0
assert np.allclose(np.array(C_py, dtype=np.float32), expected, atol=0.5)
