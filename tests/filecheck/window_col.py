# RUN: uv run exojit --mlir %s | filecheck %s

# CHECK: builtin.module {
# CHECK-NEXT:   llvm.func @set_col({{.*}}: !llvm.ptr) {
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}}: i64)
# CHECK-NEXT:   {{.*}}({{.*}}: i64):
# CHECK-NEXT:     {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}}: i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0.000000e+00 : f32) : f32
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = "llvm.getelementptr"({{.*}}, {{.*}}) <{rawConstantIndices = array<i32: -2147483648>, elem_type = f32, noWrapFlags = 0 : i32, inbounds}> : (!llvm.ptr, i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"({{.*}}, {{.*}}) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}}: i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}}: i64)
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @window_col({{.*}}: !llvm.ptr) {
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}}: i64)
# CHECK-NEXT:   {{.*}}({{.*}}: i64):
# CHECK-NEXT:     {{.*}} = llvm.icmp "slt" {{.*}}, {{.*}}: i64
# CHECK-NEXT:     llvm.cond_br {{.*}}, {{.*}}, {{.*}}
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(0) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(1) : i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.mlir.constant(4) : i64
# CHECK-NEXT:     {{.*}} = llvm.mul {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr to i64
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}}: i64
# CHECK-NEXT:     {{.*}} = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
# CHECK-NEXT:     "llvm.call"({{.*}}) <{callee = @set_col, fastmathFlags = #llvm.fastmath<none>, CConv = #llvm.cconv<ccc>, op_bundle_sizes = array<i32>, operandSegmentSizes = array<i32: 1, 0>, TailCallKind = #llvm.tailcallkind<none>}> : (!llvm.ptr) -> ()
# CHECK-NEXT:     {{.*}} = llvm.add {{.*}}, {{.*}}: i64
# CHECK-NEXT:     llvm.br {{.*}}({{.*}}: i64)
# CHECK-NEXT:   {{.*}}:
# CHECK-NEXT:     llvm.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }


from __future__ import annotations

from exo import *


@proc
def set_col(col: [f32][4] @ DRAM):
    for i in seq(0, 4):
        col[i] = 0.0


@proc
def window_col(A: f32[4, 4] @ DRAM):
    for j in seq(0, 4):
        set_col(A[:, j])
