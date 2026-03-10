from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import sqrt
from exo.stdlib.scheduling import rename

from xnumpy.main import compile_jit
from xnumpy.patches_exo import NEON


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


@instr("neon_broadcast_f32x4({dst_data}, {src_data});")
def neon_broadcast_f32x4(dst: [f32][4] @ NEON, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[0]


@instr("neon_mul_f32x4({dst_data}, {a_data}, {b_data});")
def neon_mul_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] * b[i]


@instr("neon_div_f32x4({dst_data}, {a_data}, {b_data});")
def neon_div_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] / b[i]


@instr("neon_fmadd_f32x4({dst_data}, {a_data}, {b_data});")
def neon_fmadd_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] += a[i] * b[i]


@instr("neon_add_f32x4({dst_data}, {a_data}, {b_data});")
def neon_add_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] + b[i]


@instr("neon_sub_f32x4({dst_data}, {a_data}, {b_data});")
def neon_sub_f32x4(dst: [f32][4] @ NEON, a: [f32][4] @ NEON, b: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(a, 0) == 1
    assert stride(b, 0) == 1
    for i in seq(0, 4):
        dst[i] = a[i] - b[i]


@instr("neon_sqrt_f32x4({dst_data}, {src_data});")
def neon_sqrt_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = sqrt(src[i])


@instr("neon_square_f32x4({dst_data}, {src_data});")
def neon_square_f32x4(dst: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i] * src[i]


@instr("neon_add_acc_f32x4({acc_data}, {src_data});")
def neon_add_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] += src[i]


@instr("neon_sub_acc_f32x4({acc_data}, {src_data});")
def neon_sub_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] - src[i]


@instr("neon_mul_acc_f32x4({acc_data}, {src_data});")
def neon_mul_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] * src[i]


@instr("neon_div_acc_f32x4({acc_data}, {src_data});")
def neon_div_acc_f32x4(acc: [f32][4] @ NEON, src: [f32][4] @ NEON):
    assert stride(acc, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        acc[i] = acc[i] / src[i]


@cache
def adam_neon(n: int) -> Callable[..., None]:
    assert n % 16 == 0
    n16 = n // 16

    @proc
    def _adam_neon(param: f32[n] @ DRAM, grad: f32[n] @ DRAM, m: f32[n] @ DRAM, v: f32[n] @ DRAM, b1: f32[1] @ DRAM, b2: f32[1] @ DRAM, eps: f32[1] @ DRAM, lr: f32[1] @ DRAM, beta1_t: f32[1] @ DRAM, beta2_t: f32[1] @ DRAM):
        # scalars
        one: f32[1] @ DRAM
        one[0] = 1.0

        b1_v: f32[4] @ NEON
        b2_v: f32[4] @ NEON
        eps_v: f32[4] @ NEON
        lr_v: f32[4] @ NEON
        inv_beta1_t_v: f32[4] @ NEON
        inv_beta2_t_v: f32[4] @ NEON
        inv_b1_v: f32[4] @ NEON
        inv_b2_v: f32[4] @ NEON

        neon_broadcast_f32x4(b1_v, b1[0:1])
        neon_broadcast_f32x4(b2_v, b2[0:1])
        neon_broadcast_f32x4(eps_v, eps[0:1])
        neon_broadcast_f32x4(lr_v, lr[0:1])

        # inv_beta1_t = 1.0 / beta1_t
        ib1t: f32[1] @ DRAM
        ib1t[0] = one[0] / beta1_t[0]
        neon_broadcast_f32x4(inv_beta1_t_v, ib1t[0:1])

        ib2t: f32[1] @ DRAM
        ib2t[0] = one[0] / beta2_t[0]
        neon_broadcast_f32x4(inv_beta2_t_v, ib2t[0:1])

        ib1: f32[1] @ DRAM
        ib1[0] = one[0] - b1[0]
        neon_broadcast_f32x4(inv_b1_v, ib1[0:1])

        ib2: f32[1] @ DRAM
        ib2[0] = one[0] - b2[0]
        neon_broadcast_f32x4(inv_b2_v, ib2[0:1])

        for i in seq(0, n16):
            g0: f32[4] @ NEON
            g1: f32[4] @ NEON
            g2: f32[4] @ NEON
            g3: f32[4] @ NEON
            neon_loadu_f32x4(g0, grad[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(g1, grad[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(g2, grad[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(g3, grad[16 * i + 12 : 16 * i + 16])

            m0: f32[4] @ NEON
            m1: f32[4] @ NEON
            m2: f32[4] @ NEON
            m3: f32[4] @ NEON
            neon_loadu_f32x4(m0, m[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(m1, m[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(m2, m[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(m3, m[16 * i + 12 : 16 * i + 16])

            neon_mul_acc_f32x4(m0, b1_v)
            neon_mul_acc_f32x4(m1, b1_v)
            neon_mul_acc_f32x4(m2, b1_v)
            neon_mul_acc_f32x4(m3, b1_v)
            neon_fmadd_f32x4(m0, inv_b1_v, g0)
            neon_fmadd_f32x4(m1, inv_b1_v, g1)
            neon_fmadd_f32x4(m2, inv_b1_v, g2)
            neon_fmadd_f32x4(m3, inv_b1_v, g3)

            neon_storeu_f32x4(m[16 * i + 0 : 16 * i + 4], m0)
            neon_storeu_f32x4(m[16 * i + 4 : 16 * i + 8], m1)
            neon_storeu_f32x4(m[16 * i + 8 : 16 * i + 12], m2)
            neon_storeu_f32x4(m[16 * i + 12 : 16 * i + 16], m3)

            v0: f32[4] @ NEON
            v1: f32[4] @ NEON
            v2: f32[4] @ NEON
            v3: f32[4] @ NEON
            neon_loadu_f32x4(v0, v[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(v1, v[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(v2, v[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(v3, v[16 * i + 12 : 16 * i + 16])

            neon_mul_acc_f32x4(v0, b2_v)
            neon_mul_acc_f32x4(v1, b2_v)
            neon_mul_acc_f32x4(v2, b2_v)
            neon_mul_acc_f32x4(v3, b2_v)

            gsq0: f32[4] @ NEON
            gsq1: f32[4] @ NEON
            gsq2: f32[4] @ NEON
            gsq3: f32[4] @ NEON
            neon_square_f32x4(gsq0, g0)
            neon_square_f32x4(gsq1, g1)
            neon_square_f32x4(gsq2, g2)
            neon_square_f32x4(gsq3, g3)

            neon_fmadd_f32x4(v0, inv_b2_v, gsq0)
            neon_fmadd_f32x4(v1, inv_b2_v, gsq1)
            neon_fmadd_f32x4(v2, inv_b2_v, gsq2)
            neon_fmadd_f32x4(v3, inv_b2_v, gsq3)

            neon_storeu_f32x4(v[16 * i + 0 : 16 * i + 4], v0)
            neon_storeu_f32x4(v[16 * i + 4 : 16 * i + 8], v1)
            neon_storeu_f32x4(v[16 * i + 8 : 16 * i + 12], v2)
            neon_storeu_f32x4(v[16 * i + 12 : 16 * i + 16], v3)

            m_hat0: f32[4] @ NEON
            m_hat1: f32[4] @ NEON
            m_hat2: f32[4] @ NEON
            m_hat3: f32[4] @ NEON
            neon_mul_f32x4(m_hat0, m0, inv_beta1_t_v)
            neon_mul_f32x4(m_hat1, m1, inv_beta1_t_v)
            neon_mul_f32x4(m_hat2, m2, inv_beta1_t_v)
            neon_mul_f32x4(m_hat3, m3, inv_beta1_t_v)

            v_hat0: f32[4] @ NEON
            v_hat1: f32[4] @ NEON
            v_hat2: f32[4] @ NEON
            v_hat3: f32[4] @ NEON
            neon_mul_f32x4(v_hat0, v0, inv_beta2_t_v)
            neon_mul_f32x4(v_hat1, v1, inv_beta2_t_v)
            neon_mul_f32x4(v_hat2, v2, inv_beta2_t_v)
            neon_mul_f32x4(v_hat3, v3, inv_beta2_t_v)

            v_hat_sq0: f32[4] @ NEON
            v_hat_sq1: f32[4] @ NEON
            v_hat_sq2: f32[4] @ NEON
            v_hat_sq3: f32[4] @ NEON
            neon_sqrt_f32x4(v_hat_sq0, v_hat0)
            neon_sqrt_f32x4(v_hat_sq1, v_hat1)
            neon_sqrt_f32x4(v_hat_sq2, v_hat2)
            neon_sqrt_f32x4(v_hat_sq3, v_hat3)

            neon_add_acc_f32x4(v_hat_sq0, eps_v)
            neon_add_acc_f32x4(v_hat_sq1, eps_v)
            neon_add_acc_f32x4(v_hat_sq2, eps_v)
            neon_add_acc_f32x4(v_hat_sq3, eps_v)

            upd0: f32[4] @ NEON
            upd1: f32[4] @ NEON
            upd2: f32[4] @ NEON
            upd3: f32[4] @ NEON
            neon_div_f32x4(upd0, m_hat0, v_hat_sq0)
            neon_div_f32x4(upd1, m_hat1, v_hat_sq1)
            neon_div_f32x4(upd2, m_hat2, v_hat_sq2)
            neon_div_f32x4(upd3, m_hat3, v_hat_sq3)

            neon_mul_acc_f32x4(upd0, lr_v)
            neon_mul_acc_f32x4(upd1, lr_v)
            neon_mul_acc_f32x4(upd2, lr_v)
            neon_mul_acc_f32x4(upd3, lr_v)

            p0: f32[4] @ NEON
            p1: f32[4] @ NEON
            p2: f32[4] @ NEON
            p3: f32[4] @ NEON
            neon_loadu_f32x4(p0, param[16 * i + 0 : 16 * i + 4])
            neon_loadu_f32x4(p1, param[16 * i + 4 : 16 * i + 8])
            neon_loadu_f32x4(p2, param[16 * i + 8 : 16 * i + 12])
            neon_loadu_f32x4(p3, param[16 * i + 12 : 16 * i + 16])

            neon_sub_acc_f32x4(p0, upd0)
            neon_sub_acc_f32x4(p1, upd1)
            neon_sub_acc_f32x4(p2, upd2)
            neon_sub_acc_f32x4(p3, upd3)

            neon_storeu_f32x4(param[16 * i + 0 : 16 * i + 4], p0)
            neon_storeu_f32x4(param[16 * i + 4 : 16 * i + 8], p1)
            neon_storeu_f32x4(param[16 * i + 8 : 16 * i + 12], p2)
            neon_storeu_f32x4(param[16 * i + 12 : 16 * i + 16], p3)

    name = f"_adam_neon_{n}"
    p = rename(_adam_neon, name)
    return compile_jit(p)[name]
