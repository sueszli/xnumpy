# REQUIRES: aarch64
# RUN: uv run xnumpy --asm %s | filecheck %s

from __future__ import annotations

from exo import *

from xnumpy.patches_exo import NEON

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
# test_add_f32: vec_add_f32x4 produces fadd.4s
#

# CHECK-LABEL: _test_add_f32:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  ldr	q1, [x2]
# CHECK-NEXT:  fadd.4s	v0, v0, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_add_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM, y: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    c: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    neon_loadu_f32x4(b, y[0:4])
    vec_add_f32x4(c, a, b)
    neon_storeu_f32x4(out[0:4], c)


#
# test_add_f64: vec_add_f64x2 produces fadd.2d
#

# CHECK-LABEL: _test_add_f64:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  ldr	q1, [x2]
# CHECK-NEXT:  fadd.2d	v0, v0, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_add_f64(out: f64[2] @ DRAM, x: f64[2] @ DRAM, y: f64[2] @ DRAM):
    a: f64[2] @ NEON
    b: f64[2] @ NEON
    c: f64[2] @ NEON
    neon_loadu_f64x2(a, x[0:2])
    neon_loadu_f64x2(b, y[0:2])
    vec_add_f64x2(c, a, b)
    neon_storeu_f64x2(out[0:2], c)


#
# test_add_red_f32: vec_add_red accumulates with fadd.4s
#

# CHECK-LABEL: _test_add_red_f32:
# CHECK-NEXT:  ldr	q0, [x0]
# CHECK-NEXT:  ldr	q1, [x1]
# CHECK-NEXT:  fadd.4s	v0, v0, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_add_red_f32(dst: f32[4] @ DRAM, src: f32[4] @ DRAM):
    vd: f32[4] @ NEON
    vs: f32[4] @ NEON
    neon_loadu_f32x4(vd, dst[0:4])
    neon_loadu_f32x4(vs, src[0:4])
    vec_add_red_f32x4(vd, vs)
    neon_storeu_f32x4(dst[0:4], vd)


#
# test_broadcast_f64: neon_broadcast_f64x2 produces ld1r.2d
#

# CHECK-LABEL: _test_broadcast_f64:
# CHECK-NEXT:  ld1r.2d	{ v0 }, [x1]
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_broadcast_f64(out: f64[2] @ DRAM, s: f64[1] @ DRAM):
    v: f64[2] @ NEON
    neon_broadcast_f64x2(v, s[0:1])
    neon_storeu_f64x2(out[0:2], v)


#
# test_copy_f32: vec_copy_f32x4 is just ldr+str
#

# CHECK-LABEL: _test_copy_f32:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_copy_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    vec_copy_f32x4(b, a)
    neon_storeu_f32x4(out[0:4], b)


#
# test_fma_f32: neon_fmadd_f32x4 produces fmla.4s
#

# CHECK-LABEL: _test_fma_f32:
# CHECK-NEXT:  ldr	q0, [x0]
# CHECK-NEXT:  ldr	q1, [x1]
# CHECK-NEXT:  ldr	q2, [x2]
# CHECK-NEXT:  fmla.4s	v0, v2, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_fma_f32(dst: f32[4] @ DRAM, a: f32[4] @ DRAM, b: f32[4] @ DRAM):
    vd: f32[4] @ NEON
    va: f32[4] @ NEON
    vb: f32[4] @ NEON
    neon_loadu_f32x4(vd, dst[0:4])
    neon_loadu_f32x4(va, a[0:4])
    neon_loadu_f32x4(vb, b[0:4])
    neon_fmadd_f32x4(vd, va, vb)
    neon_storeu_f32x4(dst[0:4], vd)


#
# test_fma_f64: neon_fmadd_f64x2 produces fmla.2d
#

# CHECK-LABEL: _test_fma_f64:
# CHECK-NEXT:  ldr	q0, [x0]
# CHECK-NEXT:  ldr	q1, [x1]
# CHECK-NEXT:  ldr	q2, [x2]
# CHECK-NEXT:  fmla.2d	v0, v2, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_fma_f64(dst: f64[2] @ DRAM, a: f64[2] @ DRAM, b: f64[2] @ DRAM):
    vd: f64[2] @ NEON
    va: f64[2] @ NEON
    vb: f64[2] @ NEON
    neon_loadu_f64x2(vd, dst[0:2])
    neon_loadu_f64x2(va, a[0:2])
    neon_loadu_f64x2(vb, b[0:2])
    neon_fmadd_f64x2(vd, va, vb)
    neon_storeu_f64x2(dst[0:2], vd)


#
# test_loop_add_f32: 4-iteration unrolled fadd.4s loop
#

# CHECK-LABEL: _test_loop_add_f32:
# CHECK:       fadd.4s
# CHECK:       fadd.4s
# CHECK:       fadd.4s
# CHECK:       fadd.4s
# CHECK:       ret


@proc
def test_loop_add_f32(out: f32[16] @ DRAM, x: f32[16] @ DRAM, y: f32[16] @ DRAM):
    for j in seq(0, 4):
        a: f32[4] @ NEON
        b: f32[4] @ NEON
        c: f32[4] @ NEON
        neon_loadu_f32x4(a, x[4 * j : 4 * j + 4])
        neon_loadu_f32x4(b, y[4 * j : 4 * j + 4])
        vec_add_f32x4(c, a, b)
        neon_storeu_f32x4(out[4 * j : 4 * j + 4], c)


#
# test_mul_f32: vec_mul_f32x4 produces fmul.4s
#

# CHECK-LABEL: _test_mul_f32:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  ldr	q1, [x2]
# CHECK-NEXT:  fmul.4s	v0, v0, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_mul_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM, y: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    c: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    neon_loadu_f32x4(b, y[0:4])
    vec_mul_f32x4(c, a, b)
    neon_storeu_f32x4(out[0:4], c)


#
# test_mul_f64: vec_mul_f64x2 produces fmul.2d
#

# CHECK-LABEL: _test_mul_f64:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  ldr	q1, [x2]
# CHECK-NEXT:  fmul.2d	v0, v0, v1
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_mul_f64(out: f64[2] @ DRAM, x: f64[2] @ DRAM, y: f64[2] @ DRAM):
    a: f64[2] @ NEON
    b: f64[2] @ NEON
    c: f64[2] @ NEON
    neon_loadu_f64x2(a, x[0:2])
    neon_loadu_f64x2(b, y[0:2])
    vec_mul_f64x2(c, a, b)
    neon_storeu_f64x2(out[0:2], c)


#
# test_neg_f32: vec_neg_f32x4 produces fneg.4s
#

# CHECK-LABEL: _test_neg_f32:
# CHECK-NEXT:  ldr	q0, [x1]
# CHECK-NEXT:  fneg.4s	v0, v0
# CHECK-NEXT:  str	q0, [x0]
# CHECK-NEXT:  ret


@proc
def test_neg_f32(out: f32[4] @ DRAM, x: f32[4] @ DRAM):
    a: f32[4] @ NEON
    b: f32[4] @ NEON
    neon_loadu_f32x4(a, x[0:4])
    vec_neg_f32x4(b, a)
    neon_storeu_f32x4(out[0:4], b)


#
# test_saxpy_f32: broadcast scalar + fmla.4s with lane-indexed operand
#

# CHECK-LABEL: _test_saxpy_f32:
# CHECK-NEXT:  ldr	s0, [x1]
# CHECK:       fmla.4s	v{{[0-9]+}}, v{{[0-9]+}}, v0[0]
# CHECK:       ret


@proc
def test_saxpy_f32(y: f32[4] @ DRAM, a: f32[1] @ DRAM, x: f32[4] @ DRAM):
    va: f32[4] @ NEON
    vx: f32[4] @ NEON
    vy: f32[4] @ NEON
    neon_broadcast_f32x4(va, a[0:1])
    neon_loadu_f32x4(vx, x[0:4])
    neon_loadu_f32x4(vy, y[0:4])
    neon_fmadd_f32x4(vy, va, vx)
    neon_storeu_f32x4(y[0:4], vy)


#
# test_saxpy_loop: broadcast + 4x fmla.4s in loop
#

# CHECK-LABEL: _test_saxpy_loop:
# CHECK:       ldr	s{{[0-9]+}}, [x{{[0-9]+}}]
# CHECK:       fmla.4s
# CHECK:       fmla.4s
# CHECK:       fmla.4s
# CHECK:       fmla.4s
# CHECK:       ret


@proc
def test_saxpy_loop(y: f32[16] @ DRAM, a: f32[1] @ DRAM, x: f32[16] @ DRAM):
    va: f32[4] @ NEON
    neon_broadcast_f32x4(va, a[0:1])
    for j in seq(0, 4):
        vx: f32[4] @ NEON
        vy: f32[4] @ NEON
        neon_loadu_f32x4(vx, x[4 * j : 4 * j + 4])
        neon_loadu_f32x4(vy, y[4 * j : 4 * j + 4])
        neon_fmadd_f32x4(vy, va, vx)
        neon_storeu_f32x4(y[4 * j : 4 * j + 4], vy)
