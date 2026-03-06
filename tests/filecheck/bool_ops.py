# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import *


# CHECK:      func.func @bool_ops(%offset_pointer : !llvm.ptr, {{.*}} : i64, {{.*}} : i64, {{.*}} : i64) {
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}} : i64
# CHECK-NEXT:   {{.*}} = llvm.and {{.*}}, {{.*}} : i1
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb0, ^bb1
# CHECK:        {{.*}} = llvm.or {{.*}}, {{.*}} : i1
# CHECK-NEXT:   cf.cond_br {{.*}}, ^bb2, ^bb3
@proc
def bool_ops(out: f32[1] @ DRAM, a: index, b: index, c: index):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    if a < b and b < c:
        out[0] = 1.0
    if a < b or b < c:
        out[0] = 2.0
