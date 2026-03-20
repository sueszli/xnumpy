from __future__ import annotations

from exo import *
from exo.libs.externs import select, sqrt

from exojit.patches_exo import Stack


@proc
def fill(M: size, N: size, x: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    # x[i, j] = value
    for i in seq(0, M):
        for j in seq(0, N):
            x[i, j] = value[0]


@proc
def add(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM):
    # out += x
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] += x[i, j]


@proc
def matmul_right_t(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[N, K] @ DRAM, zero: f64[1] @ DRAM):
    # out = x @ w^t
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[j, k]
            out[i, j] = acc


@proc
def matmul(M: size, N: size, K: size, out: f64[M, N] @ DRAM, x: f64[M, K] @ DRAM, w: f64[K, N] @ DRAM, zero: f64[1] @ DRAM):
    # out = x @ w
    for i in seq(0, M):
        for j in seq(0, N):
            acc: f64 @ Stack
            acc = zero[0]
            for k in seq(0, K):
                acc += x[i, k] * w[k, j]
            out[i, j] = acc


@proc
def matmul_left_t(M: size, N: size, K: size, out: f64[N, K] @ DRAM, x: f64[M, N] @ DRAM, w: f64[M, K] @ DRAM, zero: f64[1] @ DRAM):
    # out = x^t @ w
    for j in seq(0, N):
        for k in seq(0, K):
            acc: f64 @ Stack
            acc = zero[0]
            for i in seq(0, M):
                acc += x[i, j] * w[i, k]
            out[j, k] = acc


@proc
def rmsnorm(M: size, N: size, out: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM, one: f64[1] @ DRAM, inv_n: f64[1] @ DRAM, eps: f64[1] @ DRAM):
    # out = x / sqrt(mean(x^2) + eps)
    for i in seq(0, M):
        sumsq: f64 @ Stack
        scale: f64 @ Stack
        sumsq = zero[0]
        for j in seq(0, N):
            sumsq += x[i, j] * x[i, j]
        scale = one[0] / sqrt(sumsq * inv_n[0] + eps[0])
        rms[i, 0] = scale
        for j in seq(0, N):
            out[i, j] = x[i, j] * scale


@proc
def rmsnorm_bwd(M: size, N: size, out: f64[M, N] @ DRAM, dx: f64[M, N] @ DRAM, x_pre: f64[M, N] @ DRAM, rms: f64[M, 1] @ DRAM, zero: f64[1] @ DRAM, inv_n: f64[1] @ DRAM):
    # out = dnorm + dx residual
    for i in seq(0, M):
        dot: f64 @ Stack
        scale: f64 @ Stack
        corr: f64 @ Stack
        dot = zero[0]
        scale = rms[i, 0]
        for j in seq(0, N):
            dot += out[i, j] * x_pre[i, j]
        corr = scale * scale * scale * inv_n[0] * dot
        for j in seq(0, N):
            out[i, j] = out[i, j] * scale - x_pre[i, j] * corr + dx[i, j]


@proc
def relu(M: size, N: size, out: f64[M, N] @ DRAM, x: f64[M, N] @ DRAM, zero: f64[1] @ DRAM):
    # out = max(x, 0)
    for i in seq(0, M):
        for j in seq(0, N):
            out[i, j] = select(zero[0], x[i, j], x[i, j], zero[0])


@proc
def fill3(M: size, N: size, a: f64[M, N] @ DRAM, b: f64[M, N] @ DRAM, c: f64[M, N] @ DRAM, value: f64[1] @ DRAM):
    # fill three tensors with one scalar
    fill(M, N, a, value)
    fill(M, N, b, value)
    fill(M, N, c, value)
