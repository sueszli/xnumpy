from __future__ import annotations

import platform

import numpy as np
import pytest
from exo import *

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON

pytestmark = pytest.mark.skipif(platform.machine() not in ("arm64", "aarch64"), reason="NEON requires aarch64")


#
# f32x4 intrinsic declarations
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
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("vec_add_f32x4({dst_data}, {a_data}, {b_data});")
def vec_add_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("vec_mul_f32x4({dst_data}, {a_data}, {b_data});")
def vec_mul_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("vec_neg_f32x4({dst_data}, {src_data});")
def vec_neg_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = -src[i]


@instr("vec_copy_f32x4({dst_data}, {src_data});")
def vec_copy_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]


@instr("vec_add_red_f32x4({dst_data}, {src_data});")
def vec_add_red_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] += src[i]


#
# f64x2 intrinsic declarations
#


@instr("neon_loadu_f64x2({dst_data}, {src_data});")
def neon_loadu_f64x2(dst: [f64][2] @ NEON, src: [f64][2] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[i]


@instr("neon_storeu_f64x2({dst_data}, {src_data});")
def neon_storeu_f64x2(dst: [f64][2] @ DRAM, src: [f64][2] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[i]


@instr("neon_fmadd_f64x2({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f64x2(dst: [f64][2] @ NEON, a: [f64][2] @ NEON, b: [f64][2] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] += a[i] * b[i]


@instr("neon_broadcast_f64x2({dst_data}, {src_data});")
def neon_broadcast_f64x2(dst: [f64][2] @ NEON, src: [f64][1] @ DRAM):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 2):
        dst[i] = src[0]


@instr("vec_add_f64x2({dst_data}, {a_data}, {b_data});")
def vec_add_f64x2(dst: [f64][2] @ NEON, a: [f64][2] @ NEON, b: [f64][2] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] = a[i] + b[i]


@instr("vec_mul_f64x2({dst_data}, {a_data}, {b_data});")
def vec_mul_f64x2(dst: [f64][2] @ NEON, a: [f64][2] @ NEON, b: [f64][2] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 2):
        dst[i] = a[i] * b[i]


#
# test procs
#


@proc
def neon_vec_add_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM, y: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    c: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    neon_loadu_f32x4(b, y[0:4])
    vec_add_f32x4(c, a, b)
    neon_storeu_f32x4(out[0:4], c)


@proc
def neon_vec_mul_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM, y: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    c: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    neon_loadu_f32x4(b, y[0:4])
    vec_mul_f32x4(c, a, b)
    neon_storeu_f32x4(out[0:4], c)


@proc
def neon_vec_neg_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    vec_neg_f32x4(b, a)
    neon_storeu_f32x4(out[0:4], b)


@proc
def neon_vec_copy_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    vec_copy_f32x4(b, a)
    neon_storeu_f32x4(out[0:4], b)


@proc
def neon_vec_fma_f32(dst: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    vd: f32[4] @ NEON
    va: f32[4] @ NEON
    vb: f32[4] @ NEON
    neon_loadu_f32x4(vd, dst[0:4])
    neon_loadu_f32x4(va, a[0:4])
    neon_loadu_f32x4(vb, b[0:4])
    neon_fmadd_f32x4(vd, va, vb)
    neon_storeu_f32x4(dst[0:4], vd)


@proc
def neon_vec_add_red_f32(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
    vd: f32[4] @ NEON
    vs: f32[4] @ NEON
    neon_loadu_f32x4(vd, dst[0:4])
    neon_loadu_f32x4(vs, src[0:4])
    vec_add_red_f32x4(vd, vs)
    neon_storeu_f32x4(dst[0:4], vd)


@proc
def neon_saxpy_f32(y: f32[16] @ DRAM, a: f32[1] @ DRAM, x: f32[16] @ DRAM):
    va: f32[4] @ NEON
    neon_broadcast_f32x4(va, a[0:1])
    for j in seq(0, 4):
        vx: f32[4] @ NEON
        vy: f32[4] @ NEON
        neon_loadu_f32x4(vx, x[4 * j : 4 * j + 4])
        neon_loadu_f32x4(vy, y[4 * j : 4 * j + 4])
        neon_fmadd_f32x4(vy, va, vx)
        neon_storeu_f32x4(y[4 * j : 4 * j + 4], vy)


@proc
def neon_vec_add_f64(out: f64[2] @ DRAM, x: f64[2] @ DRAM, y: f64[2] @ DRAM):
    a: f64[2] @ NEON
    b: f64[2] @ NEON
    c: f64[2] @ NEON
    neon_loadu_f64x2(a, x[0:2])
    neon_loadu_f64x2(b, y[0:2])
    vec_add_f64x2(c, a, b)
    neon_storeu_f64x2(out[0:2], c)


@proc
def neon_vec_mul_f64(out: f64[2] @ DRAM, x: f64[2] @ DRAM, y: f64[2] @ DRAM):
    a: f64[2] @ NEON
    b: f64[2] @ NEON
    c: f64[2] @ NEON
    neon_loadu_f64x2(a, x[0:2])
    neon_loadu_f64x2(b, y[0:2])
    vec_mul_f64x2(c, a, b)
    neon_storeu_f64x2(out[0:2], c)


@proc
def neon_vec_fma_f64(dst: f64[2] @ DRAM, a: f64[2] @ DRAM, b: f64[2] @ DRAM):
    vd: f64[2] @ NEON
    va: f64[2] @ NEON
    vb: f64[2] @ NEON
    neon_loadu_f64x2(vd, dst[0:2])
    neon_loadu_f64x2(va, a[0:2])
    neon_loadu_f64x2(vb, b[0:2])
    neon_fmadd_f64x2(vd, va, vb)
    neon_storeu_f64x2(dst[0:2], vd)


@proc
def neon_broadcast_store_f64(out: f64[2] @ DRAM, s: f64[1] @ DRAM):
    v: f64[2] @ NEON
    neon_broadcast_f64x2(v, s[0:1])
    neon_storeu_f64x2(out[0:2], v)


#
# jit helper
#


def _jit_call(proc_obj, **kwargs):
    fns = compile_jit(proc_obj)
    fn = fns[proc_obj._loopir_proc.name]

    args, bufs = [], {}
    for arg in proc_obj._loopir_proc.args:
        name = str(arg.name)
        val = kwargs[name]
        if hasattr(arg.type, "basetype"):
            dtype = {"f32": np.float32, "f64": np.float64}[str(arg.type.basetype())]
            arr = np.array(val, dtype=dtype)
            bufs[name] = arr
            args.append(arr.ctypes.data)
        else:
            args.append(int(val))

    fn(*args)
    return bufs


#
# f32x4 tests
#


def test_neon_add_f32():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    bufs = _jit_call(neon_vec_add_f32, out=[0.0] * 4, x=x, y=y)
    np.testing.assert_allclose(bufs["out"], np.array(x) + np.array(y))


def test_neon_mul_f32():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [5.0, 6.0, 7.0, 8.0]
    bufs = _jit_call(neon_vec_mul_f32, out=[0.0] * 4, x=x, y=y)
    np.testing.assert_allclose(bufs["out"], np.array(x) * np.array(y))


def test_neon_neg_f32():
    x = [1.0, -2.0, 0.0, 3.5]
    bufs = _jit_call(neon_vec_neg_f32, out=[0.0] * 4, x=x)
    np.testing.assert_allclose(bufs["out"], -np.array(x, dtype=np.float32))


def test_neon_copy_f32():
    x = [42.0, -1.0, 0.0, 3.14]
    bufs = _jit_call(neon_vec_copy_f32, out=[0.0] * 4, x=x)
    np.testing.assert_allclose(bufs["out"], x)


def test_neon_fma_f32():
    dst = [1.0, 2.0, 3.0, 4.0]
    a = [2.0, 3.0, 4.0, 5.0]
    b = [10.0, 10.0, 10.0, 10.0]
    bufs = _jit_call(neon_vec_fma_f32, dst=list(dst), a=a, b=b)
    expected = np.array(dst) + np.array(a) * np.array(b)
    np.testing.assert_allclose(bufs["dst"], expected)


def test_neon_add_red_f32():
    dst = [1.0, 2.0, 3.0, 4.0]
    src = [10.0, 20.0, 30.0, 40.0]
    bufs = _jit_call(neon_vec_add_red_f32, dst=list(dst), src=src)
    np.testing.assert_allclose(bufs["dst"], np.array(dst) + np.array(src))


def test_neon_saxpy_f32():
    y = [float(i) for i in range(16)]
    x = [float(i * 2) for i in range(16)]
    a = [3.0]
    y_orig = np.array(y, dtype=np.float32)
    bufs = _jit_call(neon_saxpy_f32, y=list(y), a=a, x=x)
    expected = y_orig + 3.0 * np.array(x, dtype=np.float32)
    np.testing.assert_allclose(bufs["y"], expected)


#
# f64x2 tests
#


def test_neon_add_f64():
    x = [1.0, 2.0]
    y = [10.0, 20.0]
    bufs = _jit_call(neon_vec_add_f64, out=[0.0] * 2, x=x, y=y)
    np.testing.assert_allclose(bufs["out"], np.array(x) + np.array(y))


def test_neon_mul_f64():
    x = [3.0, 4.0]
    y = [5.0, 6.0]
    bufs = _jit_call(neon_vec_mul_f64, out=[0.0] * 2, x=x, y=y)
    np.testing.assert_allclose(bufs["out"], np.array(x) * np.array(y))


def test_neon_fma_f64():
    dst = [1.0, 2.0]
    a = [3.0, 4.0]
    b = [5.0, 6.0]
    bufs = _jit_call(neon_vec_fma_f64, dst=list(dst), a=a, b=b)
    expected = np.array(dst) + np.array(a) * np.array(b)
    np.testing.assert_allclose(bufs["dst"], expected)


def test_neon_broadcast_f64():
    bufs = _jit_call(neon_broadcast_store_f64, out=[0.0] * 2, s=[42.0])
    np.testing.assert_allclose(bufs["out"], [42.0, 42.0])
