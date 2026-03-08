from __future__ import annotations

from collections.abc import Callable

from exo import *
from exo.stdlib.scheduling import divide_loop, replace, simplify

from xnumpy.backends import compile_jit
from xnumpy.library.kernels.neon import neon_vadd_f32x4, neon_vmul_f32x4, neon_vneg_f32x4, neon_vsub_f32x4

#
# vector-vector ops  (out = a op b)
#


@proc
def _add(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + b[i]


@proc
def _sub(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] - b[i]


@proc
def _mul(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] * b[i]


@proc
def _neg(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM):
    for i in seq(0, N):
        out[i] = -a[i]


#
# scalar broadcast ops  (out = a op scalar)
#


@proc
def _sadd(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] + s[0]


@proc
def _ssub(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] - s[0]


@proc
def _smul(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        out[i] = a[i] * s[0]


@proc
def _srsub(N: size, out: f32[N] @ DRAM, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    # reverse subtract: out = scalar - a  (for rsub where scalar is the lhs)
    for i in seq(0, N):
        out[i] = s[0] - a[i]


#
# in-place vector-vector ops  (a op= b)
#


@proc
def _iadd(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] + b[i]


@proc
def _isub(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] - b[i]


@proc
def _imul(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] * b[i]


#
# in-place scalar broadcast ops  (a op= scalar)
#


@proc
def _isadd(N: size, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] + s[0]


@proc
def _issub(N: size, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] - s[0]


@proc
def _ismul(N: size, a: f32[N] @ DRAM, s: f32[1] @ DRAM):
    for i in seq(0, N):
        a[i] = a[i] * s[0]


#
# reduction
#


@proc
def _sum(N: size, out: f32[1] @ DRAM, a: f32[N] @ DRAM):
    # out[0] = sum(a)  (sequential accumulation, no vectorized reduction yet)
    out[0] = 0.0
    for i in seq(0, N):
        out[0] += a[i]


#
# scheduling
#


def _schedule_vec(intrinsic):
    def transform(p):
        # tile loop by 4 and replace inner body with NEON intrinsic
        p = divide_loop(p, "i", 4, ["io", "ii"], perfect=True)
        p = replace(p, "for ii in _: _", intrinsic)
        p = simplify(p)
        return p

    return transform


def _make(generic, prefix, vec=None):
    # factory: partial-eval N, optionally apply NEON schedule if n divisible by 4
    def kernel(n):
        return compile_jit(
            generic.partial_eval(N=n),
            f"_{prefix}_{n}",
            schedule=_schedule_vec(vec) if vec and n % 4 == 0 else None,
        )

    return kernel


#
# public api
#

add = _make(_add, "add", neon_vadd_f32x4)
sub = _make(_sub, "sub", neon_vsub_f32x4)
mul = _make(_mul, "mul", neon_vmul_f32x4)
neg = _make(_neg, "neg", neon_vneg_f32x4)

scalar_add = _make(_sadd, "sadd")
scalar_sub = _make(_ssub, "ssub")
scalar_mul = _make(_smul, "smul")
scalar_rsub = _make(_srsub, "srsub")

iadd = _make(_iadd, "iadd", neon_vadd_f32x4)
isub = _make(_isub, "isub", neon_vsub_f32x4)
imul = _make(_imul, "imul", neon_vmul_f32x4)

iscalar_add = _make(_isadd, "isadd")
iscalar_sub = _make(_issub, "issub")
iscalar_mul = _make(_ismul, "ismul")


def sum_reduce(n: int) -> Callable[..., None]:
    return compile_jit(_sum.partial_eval(N=n), f"_sum_{n}")
