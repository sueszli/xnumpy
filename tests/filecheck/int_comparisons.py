# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @int_comparisons(%offset_pointer : !llvm.ptr, {{.*}} : i64, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.icmp "eq" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb0, ^bb1
# CHECK:        {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb2, ^bb3
# CHECK:        {{.*}} = llvm.icmp "sgt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb4, ^bb5
@proc
def int_comparisons(out: i32[1] @ DRAM, a: index, b: index):
    assert a >= 0
    assert b >= 0
    if a == b:
        out[0] = 1
    if a < b:
        out[0] = 2
    if a > b:
        out[0] = 3
