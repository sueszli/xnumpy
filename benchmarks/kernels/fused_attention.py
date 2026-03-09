from __future__ import annotations

from collections.abc import Callable
from functools import cache

from exo import *
from exo.libs.externs import select
from exo.stdlib.scheduling import rename, simplify

from xnumpy.main import compile_jit
from xnumpy.patches_exo import Stack


@proc
def _fused_attention(SEQ: size, D: size, out: f32[SEQ, D] @ DRAM, q: f32[SEQ, D] @ DRAM, k: f32[SEQ, D] @ DRAM, v: f32[SEQ, D] @ DRAM, scale: f32[1] @ DRAM):
    # all temporaries on stack (alloca) for register promotion
    score: f32 @ Stack
    m_prev: f32 @ Stack
    m_new: f32 @ Stack
    l_prev: f32 @ Stack
    l_new: f32 @ Stack
    exp_diff: f32 @ Stack
    exp_score: f32 @ Stack
    inv_l: f32 @ Stack
    
    t_diff: f32 @ Stack
    y_d: f32 @ Stack
    e5_d: f32 @ Stack
    e4_d: f32 @ Stack
    e3_d: f32 @ Stack
    e2_d: f32 @ Stack
    e1_d: f32 @ Stack
    s1_d: f32 @ Stack
    s2_d: f32 @ Stack
    s3_d: f32 @ Stack
    s4_d: f32 @ Stack
    s5_d: f32 @ Stack

    t_score: f32 @ Stack
    y_s: f32 @ Stack
    e5_s: f32 @ Stack
    e4_s: f32 @ Stack
    e3_s: f32 @ Stack
    e2_s: f32 @ Stack
    e1_s: f32 @ Stack
    s1_s: f32 @ Stack
    s2_s: f32 @ Stack
    s3_s: f32 @ Stack
    s4_s: f32 @ Stack
    s5_s: f32 @ Stack

    for i in seq(0, SEQ):
        # j = 0
        score = 0.0
        for d in seq(0, D):
            score += q[i, d] * k[0, d]
        score = score * scale[0]
        
        m_new = score
        l_new = 1.0
        for d in seq(0, D):
            out[i, d] = v[0, d]

        # j = 1 to SEQ-1
        for j in seq(1, SEQ):
            score = 0.0
            for d in seq(0, D):
                score += q[i, d] * k[j, d]
            score = score * scale[0]
            
            m_prev = m_new
            m_new = select(m_new, score, score, m_new)
            
            # exp(m_prev - m_new)
            t_diff = m_prev - m_new
            y_d = t_diff * 0.03125
            e5_d = y_d * 0.008333333 + 0.041666667
            e4_d = e5_d * y_d + 0.166666667
            e3_d = e4_d * y_d + 0.5
            e2_d = e3_d * y_d + 1.0
            e1_d = e2_d * y_d + 1.0
            s1_d = e1_d * e1_d
            s2_d = s1_d * s1_d
            s3_d = s2_d * s2_d
            s4_d = s3_d * s3_d
            exp_diff = s4_d * s4_d
            
            # exp(score - m_new)
            t_score = score - m_new
            y_s = t_score * 0.03125
            e5_s = y_s * 0.008333333 + 0.041666667
            e4_s = e5_s * y_s + 0.166666667
            e3_s = e4_s * y_s + 0.5
            e2_s = e3_s * y_s + 1.0
            e1_s = e2_s * y_s + 1.0
            s1_s = e1_s * e1_s
            s2_s = s1_s * s1_s
            s3_s = s2_s * s2_s
            s4_s = s3_s * s3_s
            exp_score = s4_s * s4_s
            
            l_prev = l_new
            l_new = l_prev * exp_diff + exp_score
            
            for d in seq(0, D):
                out[i, d] = out[i, d] * exp_diff + exp_score * v[j, d]
                
        inv_l = 1.0 / l_new
        for d in seq(0, D):
            out[i, d] = out[i, d] * inv_l


@cache
def fused_attention(seq_len: int, d: int) -> Callable[..., None]:
    p = _fused_attention.partial_eval(SEQ=seq_len, D=d)
    p = simplify(p)
    name = f"_fused_attention_{seq_len}_{d}"
    return compile_jit(rename(p, name))[name]
