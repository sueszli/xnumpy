# RUN: uv run xdsl-exo -o - %s | filecheck %s

from __future__ import annotations

from exo import DRAM, proc


# CHECK: builtin.module {
# CHECK-NEXT:   func.func @fixed_matmul(%offset_pointer : !llvm.ptr, %offset_pointer_1 : !llvm.ptr, %offset_pointer_2 : !llvm.ptr) {
# CHECK-NEXT:     %0 = arith.constant 0 : i64
# CHECK-NEXT:     %1 = arith.constant 16 : i64
# CHECK-NEXT:     %2 = arith.constant 1 : i64
# CHECK-NEXT:     %3 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:     cf.br ^bb0(%0 : i64)
# CHECK-NEXT:   ^bb0(%4 : i64):
# CHECK-NEXT:     %5 = arith.cmpi slt, %4, %1 : i64
# CHECK-NEXT:     cf.cond_br %5, ^bb1(%0 : i64), ^bb2
# CHECK-NEXT:   ^bb1(%6 : i64):
# CHECK-NEXT:     %7 = arith.cmpi slt, %6, %1 : i64
# CHECK-NEXT:     cf.cond_br %7, ^bb3, ^bb4
# CHECK-NEXT:   ^bb3:
# CHECK-NEXT:     %8 = arith.index_cast %4 : i64 to index
# CHECK-NEXT:     %9 = arith.index_cast %6 : i64 to index
# CHECK-NEXT:     %pointer_dim_stride = arith.constant 16 : index
# CHECK-NEXT:     %pointer_dim_offset = arith.muli %8, %pointer_dim_stride : index
# CHECK-NEXT:     %pointer_dim_stride_1 = arith.addi %pointer_dim_offset, %9 : index
# CHECK-NEXT:     %bytes_per_element = arith.constant 4 : index
# CHECK-NEXT:     %scaled_pointer_offset = arith.muli %pointer_dim_stride_1, %bytes_per_element : index
# CHECK-NEXT:     %offset_pointer_3 = arith.index_cast %scaled_pointer_offset : index to i64
# CHECK-NEXT:     %offset_pointer_4 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:     %offset_pointer_5 = arith.addi %offset_pointer_4, %offset_pointer_3 : i64
# CHECK-NEXT:     %offset_pointer_6 = "llvm.inttoptr"(%offset_pointer_5) : (i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"(%3, %offset_pointer_6) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     cf.br ^bb5(%0 : i64)
# CHECK-NEXT:   ^bb5(%10 : i64):
# CHECK-NEXT:     %11 = arith.cmpi slt, %10, %1 : i64
# CHECK-NEXT:     cf.cond_br %11, ^bb6, ^bb7
# CHECK-NEXT:   ^bb6:
# CHECK-NEXT:     %12 = arith.index_cast %10 : i64 to index
# CHECK-NEXT:     %pointer_dim_stride_2 = arith.constant 16 : index
# CHECK-NEXT:     %pointer_dim_offset_1 = arith.muli %8, %pointer_dim_stride_2 : index
# CHECK-NEXT:     %pointer_dim_stride_3 = arith.addi %pointer_dim_offset_1, %12 : index
# CHECK-NEXT:     %bytes_per_element_1 = arith.constant 4 : index
# CHECK-NEXT:     %scaled_pointer_offset_1 = arith.muli %pointer_dim_stride_3, %bytes_per_element_1 : index
# CHECK-NEXT:     %offset_pointer_7 = arith.index_cast %scaled_pointer_offset_1 : index to i64
# CHECK-NEXT:     %offset_pointer_8 = "llvm.ptrtoint"(%offset_pointer_1) : (!llvm.ptr) -> i64
# CHECK-NEXT:     %offset_pointer_9 = arith.addi %offset_pointer_8, %offset_pointer_7 : i64
# CHECK-NEXT:     %offset_pointer_10 = "llvm.inttoptr"(%offset_pointer_9) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %13 = "llvm.load"(%offset_pointer_10) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %pointer_dim_stride_4 = arith.constant 16 : index
# CHECK-NEXT:     %pointer_dim_offset_2 = arith.muli %12, %pointer_dim_stride_4 : index
# CHECK-NEXT:     %pointer_dim_stride_5 = arith.addi %pointer_dim_offset_2, %9 : index
# CHECK-NEXT:     %bytes_per_element_2 = arith.constant 4 : index
# CHECK-NEXT:     %scaled_pointer_offset_2 = arith.muli %pointer_dim_stride_5, %bytes_per_element_2 : index
# CHECK-NEXT:     %offset_pointer_11 = arith.index_cast %scaled_pointer_offset_2 : index to i64
# CHECK-NEXT:     %offset_pointer_12 = "llvm.ptrtoint"(%offset_pointer_2) : (!llvm.ptr) -> i64
# CHECK-NEXT:     %offset_pointer_13 = arith.addi %offset_pointer_12, %offset_pointer_11 : i64
# CHECK-NEXT:     %offset_pointer_14 = "llvm.inttoptr"(%offset_pointer_13) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %14 = "llvm.load"(%offset_pointer_14) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %15 = arith.mulf %13, %14 : f32
# CHECK-NEXT:     %pointer_dim_stride_6 = arith.constant 16 : index
# CHECK-NEXT:     %pointer_dim_offset_3 = arith.muli %8, %pointer_dim_stride_6 : index
# CHECK-NEXT:     %pointer_dim_stride_7 = arith.addi %pointer_dim_offset_3, %9 : index
# CHECK-NEXT:     %bytes_per_element_3 = arith.constant 4 : index
# CHECK-NEXT:     %scaled_pointer_offset_3 = arith.muli %pointer_dim_stride_7, %bytes_per_element_3 : index
# CHECK-NEXT:     %offset_pointer_15 = arith.index_cast %scaled_pointer_offset_3 : index to i64
# CHECK-NEXT:     %offset_pointer_16 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:     %offset_pointer_17 = arith.addi %offset_pointer_16, %offset_pointer_15 : i64
# CHECK-NEXT:     %offset_pointer_18 = "llvm.inttoptr"(%offset_pointer_17) : (i64) -> !llvm.ptr
# CHECK-NEXT:     %16 = "llvm.load"(%offset_pointer_18) <{ordering = 0 : i64}> : (!llvm.ptr) -> f32
# CHECK-NEXT:     %17 = arith.addf %16, %15 : f32
# CHECK-NEXT:     %pointer_dim_stride_8 = arith.constant 16 : index
# CHECK-NEXT:     %pointer_dim_offset_4 = arith.muli %8, %pointer_dim_stride_8 : index
# CHECK-NEXT:     %pointer_dim_stride_9 = arith.addi %pointer_dim_offset_4, %9 : index
# CHECK-NEXT:     %bytes_per_element_4 = arith.constant 4 : index
# CHECK-NEXT:     %scaled_pointer_offset_4 = arith.muli %pointer_dim_stride_9, %bytes_per_element_4 : index
# CHECK-NEXT:     %offset_pointer_19 = arith.index_cast %scaled_pointer_offset_4 : index to i64
# CHECK-NEXT:     %offset_pointer_20 = "llvm.ptrtoint"(%offset_pointer) : (!llvm.ptr) -> i64
# CHECK-NEXT:     %offset_pointer_21 = arith.addi %offset_pointer_20, %offset_pointer_19 : i64
# CHECK-NEXT:     %offset_pointer_22 = "llvm.inttoptr"(%offset_pointer_21) : (i64) -> !llvm.ptr
# CHECK-NEXT:     "llvm.store"(%17, %offset_pointer_22) <{ordering = 0 : i64}> : (f32, !llvm.ptr) -> ()
# CHECK-NEXT:     %18 = arith.addi %10, %2 : i64
# CHECK-NEXT:     cf.br ^bb5(%18 : i64)
# CHECK-NEXT:   ^bb7:
# CHECK-NEXT:     %19 = arith.addi %6, %2 : i64
# CHECK-NEXT:     cf.br ^bb1(%19 : i64)
# CHECK-NEXT:   ^bb4:
# CHECK-NEXT:     %20 = arith.addi %4, %2 : i64
# CHECK-NEXT:     cf.br ^bb0(%20 : i64)
# CHECK-NEXT:   ^bb2:
# CHECK-NEXT:     func.return
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @malloc(i64) -> !llvm.ptr
# CHECK-NEXT:   llvm.func @free(!llvm.ptr)
# CHECK-NEXT: }
@proc
def fixed_matmul(C: f32[16, 16] @ DRAM, A: f32[16, 16] @ DRAM, B: f32[16, 16] @ DRAM):
    for i in seq(0, 16):
        for j in seq(0, 16):
            C[i, j] = 0.0
            for k in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]
