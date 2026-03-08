# /// script
# requires-python = ">=3.11"
# dependencies = ["xnumpy"]
# [tool.uv.sources]
# xnumpy = { path = ".." }
# ///

from __future__ import annotations

import platform
import time

import numpy as np
from exo import *
from exo.stdlib.scheduling import *

from xnumpy.main import compile_procs
from xnumpy.patches_exo import NEON
from xnumpy.patches_llvmlite import emit_assembly, jit_compile

WARMUP = 5
REPEATS = 50
BATCH = 1000

_cache: dict = {}


def jit(p):
    name = p.name()
    if name not in _cache:
        _cache[name] = jit_compile(compile_procs(p))
    return _cache[name]


def bench(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


#
# neon intrinsics
#


@instr("neon_loadu_f32x4({dst_data}, {src_data});")
def neon_loadu_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_storeu_f32x4({dst_data}, {src_data});")
def neon_storeu_f32x4(dst: [f32][4] @ DRAM, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


#
# saxpy (y += a*x, N=1024, f32)
#


@proc
def v0_naive(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    for i in seq(0, 1024):
        y[i] += a[0] * x[i]


@proc
def v1_vectorized(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 256):
        x_vec: f32[4] @ NEON
        y_vec: f32[4] @ NEON
        neon_loadu_f32x4(x_vec, x[4 * i : 4 * i + 4])
        neon_loadu_f32x4(y_vec, y[4 * i : 4 * i + 4])
        neon_fmadd_f32x4(y_vec, a_vec, x_vec)
        neon_storeu_f32x4(y[4 * i : 4 * i + 4], y_vec)


@proc
def v2_unrolled_4x(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 64):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(x1, x[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(x2, x[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(x3, x[16 * i + 12 : 16 * i + 16])
        neon_loadu_f32x4(y0, y[16 * i + 0 : 16 * i + 4])
        neon_loadu_f32x4(y1, y[16 * i + 4 : 16 * i + 8])
        neon_loadu_f32x4(y2, y[16 * i + 8 : 16 * i + 12])
        neon_loadu_f32x4(y3, y[16 * i + 12 : 16 * i + 16])
        neon_fmadd_f32x4(y0, a_vec, x0)
        neon_fmadd_f32x4(y1, a_vec, x1)
        neon_fmadd_f32x4(y2, a_vec, x2)
        neon_fmadd_f32x4(y3, a_vec, x3)
        neon_storeu_f32x4(y[16 * i + 0 : 16 * i + 4], y0)
        neon_storeu_f32x4(y[16 * i + 4 : 16 * i + 8], y1)
        neon_storeu_f32x4(y[16 * i + 8 : 16 * i + 12], y2)
        neon_storeu_f32x4(y[16 * i + 12 : 16 * i + 16], y3)


@proc
def v3_unrolled_8x(y: f32[1024] @ DRAM, x: f32[1024] @ DRAM, a: f32[1] @ DRAM):
    a_vec: f32[4] @ NEON
    neon_broadcast_f32x4(a_vec, a[0:1])
    for i in seq(0, 32):
        x0: f32[4] @ NEON
        x1: f32[4] @ NEON
        x2: f32[4] @ NEON
        x3: f32[4] @ NEON
        x4: f32[4] @ NEON
        x5: f32[4] @ NEON
        x6: f32[4] @ NEON
        x7: f32[4] @ NEON
        y0: f32[4] @ NEON
        y1: f32[4] @ NEON
        y2: f32[4] @ NEON
        y3: f32[4] @ NEON
        y4: f32[4] @ NEON
        y5: f32[4] @ NEON
        y6: f32[4] @ NEON
        y7: f32[4] @ NEON
        neon_loadu_f32x4(x0, x[32 * i + 0 : 32 * i + 4])
        neon_loadu_f32x4(x1, x[32 * i + 4 : 32 * i + 8])
        neon_loadu_f32x4(x2, x[32 * i + 8 : 32 * i + 12])
        neon_loadu_f32x4(x3, x[32 * i + 12 : 32 * i + 16])
        neon_loadu_f32x4(x4, x[32 * i + 16 : 32 * i + 20])
        neon_loadu_f32x4(x5, x[32 * i + 20 : 32 * i + 24])
        neon_loadu_f32x4(x6, x[32 * i + 24 : 32 * i + 28])
        neon_loadu_f32x4(x7, x[32 * i + 28 : 32 * i + 32])
        neon_loadu_f32x4(y0, y[32 * i + 0 : 32 * i + 4])
        neon_loadu_f32x4(y1, y[32 * i + 4 : 32 * i + 8])
        neon_loadu_f32x4(y2, y[32 * i + 8 : 32 * i + 12])
        neon_loadu_f32x4(y3, y[32 * i + 12 : 32 * i + 16])
        neon_loadu_f32x4(y4, y[32 * i + 16 : 32 * i + 20])
        neon_loadu_f32x4(y5, y[32 * i + 20 : 32 * i + 24])
        neon_loadu_f32x4(y6, y[32 * i + 24 : 32 * i + 28])
        neon_loadu_f32x4(y7, y[32 * i + 28 : 32 * i + 32])
        neon_fmadd_f32x4(y0, a_vec, x0)
        neon_fmadd_f32x4(y1, a_vec, x1)
        neon_fmadd_f32x4(y2, a_vec, x2)
        neon_fmadd_f32x4(y3, a_vec, x3)
        neon_fmadd_f32x4(y4, a_vec, x4)
        neon_fmadd_f32x4(y5, a_vec, x5)
        neon_fmadd_f32x4(y6, a_vec, x6)
        neon_fmadd_f32x4(y7, a_vec, x7)
        neon_storeu_f32x4(y[32 * i + 0 : 32 * i + 4], y0)
        neon_storeu_f32x4(y[32 * i + 4 : 32 * i + 8], y1)
        neon_storeu_f32x4(y[32 * i + 8 : 32 * i + 12], y2)
        neon_storeu_f32x4(y[32 * i + 12 : 32 * i + 16], y3)
        neon_storeu_f32x4(y[32 * i + 16 : 32 * i + 20], y4)
        neon_storeu_f32x4(y[32 * i + 20 : 32 * i + 24], y5)
        neon_storeu_f32x4(y[32 * i + 24 : 32 * i + 28], y6)
        neon_storeu_f32x4(y[32 * i + 28 : 32 * i + 32], y7)


SAXPY_KERNELS = [v0_naive, v1_vectorized, v2_unrolled_4x, v3_unrolled_8x]


#
# matmul (256x256, f32)
#


@proc
def mm_naive(C: f32[256, 256] @ DRAM, A: f32[256, 256] @ DRAM, B: f32[256, 256] @ DRAM):
    for i in seq(0, 256):
        for j in seq(0, 256):
            C[i, j] = 0.0
            for k in seq(0, 256):
                C[i, j] += A[i, k] * B[k, j]


mm_reorder = rename(mm_naive, "mm_reorder")
mm_reorder = fission(mm_reorder, mm_reorder.find("for k in _: _").before(), n_lifts=2)
mm_reorder = reorder_loops(mm_reorder, "j k")

mm_k_tile = rename(mm_naive, "mm_k_tile")
mm_k_tile = fission(mm_k_tile, mm_k_tile.find("for k in _: _").before(), n_lifts=2)
mm_k_tile = reorder_loops(mm_k_tile, "j k")
mm_k_tile = divide_loop(mm_k_tile, "k", 32, ["ko", "ki"], perfect=True)

mm_unroll = rename(mm_naive, "mm_unroll")
mm_unroll = fission(mm_unroll, mm_unroll.find("for k in _: _").before(), n_lifts=2)
mm_unroll = reorder_loops(mm_unroll, "j k")
mm_unroll = divide_loop(mm_unroll, "j #1", 4, ["jo", "ji"], perfect=True)
mm_unroll = unroll_loop(mm_unroll, "ji")


@proc
def mm_2d_tile(C: f32[256, 256] @ DRAM, A: f32[256, 256] @ DRAM, B: f32[256, 256] @ DRAM):
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


MM_KERNELS = [mm_naive, mm_reorder, mm_k_tile, mm_unroll, mm_2d_tile]


#
# run
#

np.random.seed(42)

sx = np.random.randn(1024).astype(np.float32)
sy_orig = np.random.randn(1024).astype(np.float32)
sa_val = np.float32(2.5)
sa_arr = np.array([sa_val], dtype=np.float32)
s_expected = sy_orig + sa_val * sx

saxpy_results: list[tuple[str, float]] = []
for k in SAXPY_KERNELS:
    fns = jit(k)
    fn, fn_repeat = fns[k.name()], fns[f"{k.name()}_repeat"]
    y_test = sy_orig.copy()
    fn(y_test.ctypes.data, sx.ctypes.data, sa_arr.ctypes.data)
    assert np.allclose(y_test, s_expected, atol=1e-5), f"{k.name()} wrong"
    y_b = sy_orig.copy()
    yp, xp, ap = y_b.ctypes.data, sx.ctypes.data, sa_arr.ctypes.data
    t = bench(lambda f=fn_repeat, yp=yp, xp=xp, ap=ap: f(yp, xp, ap, BATCH))
    saxpy_results.append((k.name(), t / BATCH))

y_np = sy_orig.copy()


def _np_saxpy(y=y_np, a=sa_val, xv=sx):
    for _ in range(BATCH):
        y += a * xv


t_np = bench(_np_saxpy)
saxpy_results.append(("numpy (y+=a*x)", t_np / BATCH))

if platform.machine() in ("aarch64", "arm64"):
    asm = emit_assembly(compile_procs(v1_vectorized))
    assert any(p in asm for p in ["fmla.4s", ".4s", "ldp\tq", "stp\tq"])

np_time = saxpy_results[-1][1]
for name, t in saxpy_results:
    print(f"{name:<20s} {t*1e9:7.1f}ns {np_time/t:5.1f}x numpy")

mA = np.random.randn(256, 256).astype(np.float32)
mB = np.random.randn(256, 256).astype(np.float32)
m_expected = mA @ mB

for k in MM_KERNELS:
    fn = jit(k)[k.name()]
    C = np.zeros((256, 256), dtype=np.float32)
    fn(C.ctypes.data, mA.ctypes.data, mB.ctypes.data)
    assert np.allclose(C, m_expected, atol=0.5), f"{k.name()} wrong"
    t = bench(
        lambda fn=fn: fn(
            np.zeros((256, 256), dtype=np.float32).ctypes.data,
            mA.ctypes.data,
            mB.ctypes.data,
        )
    )
    print(f"{k.name():<20s} {t*1e3:7.2f}ms {2*256**3/t/1e9:5.1f} GFLOP/s")
