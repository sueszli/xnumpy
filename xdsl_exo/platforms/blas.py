from functools import reduce

from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, MemRefType, ModuleOp, StringAttr, UnrealizedConversionCastOp, VectorType, f32, f64, i32, i64
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo
from xdsl_exo.dialects import llvm as llvm_extra


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        if op.mem.data != "VEC_AVX2":
            return

        assert isinstance(op.result.type, MemRefType), op.result.type

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(
                    IntegerAttr(
                        reduce(lambda x, y: x * y, op.result.type.get_shape()),
                        i64,
                    )
                ),
                alloc_op := llvm.AllocaOp(const_op.result, op.result.type.element_type),
                UnrealizedConversionCastOp.get(alloc_op.res, op.result.type),
            )
        )


class ConvertFreeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.DeallocOp, rewriter: PatternRewriter):
        if not isinstance(op.memref.type, MemRefType) or not isinstance(op.memref.type.memory_space, StringAttr):
            return
        if op.memref.type.memory_space.data != "VEC_AVX2":
            return

        rewriter.erase_op(op)


class InlineBLASAllocPass(ModulePass):
    name = "inline-blas-alloc"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [ConvertAllocOp(), ConvertFreeOp()],
            )
        ).rewrite_module(m)


class ConvertSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ExternOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "select":
            return

        assert len(op.arguments) == 4
        assert op.arguments[0].type == op.arguments[1].type, f"{op.arguments[0].type} != {op.arguments[1].type}"
        assert op.arguments[2].type == op.arguments[3].type, f"{op.arguments[2].type} != {op.arguments[3].type}"
        assert op.arguments[2].type == op.result.type, f"{op.arguments[2].type} != {op.result.type}"

        rewriter.replace_matched_op(
            (
                cmp_op := arith.CmpfOp(op.arguments[0], op.arguments[1], "olt"),
                arith.SelectOp(
                    cmp_op.results[0],
                    op.arguments[2],
                    op.arguments[3],
                ),
            )
        )


class ConvertVecAbsF32x8(RewritePattern):
    """
    def vec_abs_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                fabs_op := llvm_extra.FAbsOp(
                    load_op.dereferenced_value,
                    VectorType(f32, [8]),
                ),
                llvm.StoreOp(
                    fabs_op.result,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAbsF32x8Pfx(RewritePattern):
    """
    def vec_abs_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                fabs_op := llvm_extra.FAbsOp(
                    load_op.dereferenced_value,
                    VectorType(f32, [8]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                ),
                llvm_extra.MaskedStoreOp(
                    fabs_op.result,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAbsF64x4(RewritePattern):
    """
    def vec_abs_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                fabs_op := llvm_extra.FAbsOp(
                    load_op.dereferenced_value,
                    VectorType(f64, [4]),
                ),
                llvm.StoreOp(
                    fabs_op.result,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAbsF64x4Pfx(RewritePattern):
    """
    def vec_abs_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_and_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = select(0.0, src[i], src[i], -src[i])
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_abs_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                fabs_op := llvm_extra.FAbsOp(
                    load_op.dereferenced_value,
                    VectorType(f64, [4]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                ),
                llvm_extra.MaskedStoreOp(
                    fabs_op.result,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddRedF32x8(RewritePattern):
    """
    def vec_add_red_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[0],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm.StoreOp(
                    add_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAddRedF32x8Pfx(RewritePattern):
    """
    def vec_add_red_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                          src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({dst_data}, {src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm_extra.MaskedStoreOp(
                    add_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddRedF64x4(RewritePattern):
    """
    def vec_add_red_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[0],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm.StoreOp(
                    add_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAddRedF64x4Pfx(RewritePattern):
    """
    def vec_add_red_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                          src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({dst_data}, {src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_red_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm_extra.MaskedStoreOp(
                    add_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecCopyF32x8(RewritePattern):
    """
    def vec_copy_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecCopyF32x8Pfx(RewritePattern):
    """
    def vec_copy_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                       src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecCopyF64x4(RewritePattern):
    """
    def vec_copy_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecCopyF64x4Pfx(RewritePattern):
    """
    def vec_copy_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                       src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = {src_data};
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_copy_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


# Note: Alignment seems wrong - should be 1 here.
class ConvertVecLoadF32x8(RewritePattern):
    """
    def vec_load_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ DRAM):
    # @instr {dst_data} = _mm256_loadu_ps(&{src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecLoadF32x8Pfx(RewritePattern):
    """
    def vec_load_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                       src: [f32][8] @ DRAM):
    # @instr {dst_data} = _mm256_maskload_ps(&{src_data}, mm256_prefix_mask_epi32({m}));
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(op.arguments[2], VectorType(f32, [8])),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


# Note: Same as above.
class ConvertVecLoadF64x4(RewritePattern):
    """
    def vec_load_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ DRAM):
    # @instr {dst_data} = _mm256_loadu_pd(&{src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecLoadF64x4Pfx(RewritePattern):
    """
    def vec_load_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                       src: [f64][4] @ DRAM):
    # @instr {dst_data} = _mm256_maskload_pd(&{src_data}, mm256_prefix_mask_epi64x({m}));
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_load_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                sgext_op := arith.ExtSIOp(op.arguments[0], i64),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[sgext_op.result],
                    result_types=[VectorType(i64, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecReduceAddSclF32x8(RewritePattern):
    """
    def vec_reduce_add_scl_f32x8(dst: f32 @ DRAM, src: [f32][8] @ VEC_AVX2):
    # @instr *{dst_data} = mm256_reduce_add_ps({src_data});
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_reduce_add_scl_f32x8":
            return

        assert len(op.arguments) == 2

        # dst: f32 @ DRAM
        assert op.arguments[0].type == f32, op.arguments[0].type
        assert isinstance(acc_load_op := op.arguments[0].owner, llvm.LoadOp), op.arguments[0].owner

        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                reduce_op := vector.ReductionOp(
                    load_op.dereferenced_value,
                    vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]),
                    acc=op.arguments[0],
                ),
                llvm.StoreOp(reduce_op.dest, acc_load_op.ptr),
            )
        )


class ConvertVecReduceAddSclF64x4(RewritePattern):
    """
    def vec_reduce_add_scl_f64x4(dst: f64 @ DRAM, src: [f64][4] @ VEC_AVX2):
    # @instr *{dst_data} = mm256_reduce_add_pd({src_data});
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst += src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_reduce_add_scl_f64x4":
            return

        assert len(op.arguments) == 2

        # dst: f64 @ DRAM
        assert op.arguments[0].type == f64, op.arguments[0].type
        assert isinstance(acc_load_op := op.arguments[0].owner, llvm.LoadOp), op.arguments[0].owner

        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                reduce_op := vector.ReductionOp(
                    load_op.dereferenced_value,
                    vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]),
                    acc=op.arguments[0],
                ),
                llvm.StoreOp(reduce_op.dest, acc_load_op.ptr),
            )
        )


class ConvertVecZeroF32x8(RewritePattern):
    """
    def vec_zero_f32x8(dst: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_ps();
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f32x8":
            return

        assert len(op.arguments) == 1
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f32, [8]), [0.0] * 8)),
                llvm.StoreOp(
                    const_op.result,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecZeroF64x4(RewritePattern):
    """
    def vec_zero_f64x4(dst: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_pd();
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f64x4":
            return

        assert len(op.arguments) == 1
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f64, [4]), [0.0] * 4)),
                llvm.StoreOp(
                    const_op.result,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAddF32x8(RewritePattern):
    """
    def vec_add_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                  src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f32x8":
            return

        assert len(op.arguments) == 3
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                add_op := llvm.FAddOp(
                    load0_op.dereferenced_value,
                    load1_op.dereferenced_value,
                ),
                llvm.StoreOp(
                    add_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAddF32x8Pfx(RewritePattern):
    """
    def vec_add_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_ps({src1_data}, {src2_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f32x8_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm_extra.MaskedStoreOp(
                    add_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecAddF64x4(RewritePattern):
    """
    def vec_add_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                  src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f64x4":
            return

        assert len(op.arguments) == 3
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm.StoreOp(
                    add_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecAddF64x4Pfx(RewritePattern):
    """
    def vec_add_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_add_pd({src1_data}, {src2_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src1[i] + src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_add_f64x4_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                add_op := llvm.FAddOp(load0_op.dereferenced_value, load1_op.dereferenced_value),
                llvm_extra.MaskedStoreOp(
                    add_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecBrdcstSclF32x8(RewritePattern):
    """
    def vec_brdcst_scl_f32x8(dst: [f32][8] @ VEC_AVX2, src: f32 @ DRAM):
    # @instr {dst_data} = _mm256_set1_ps(*{src_data});
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: f32 @ DRAM
        assert op.arguments[1].type == f32, op.arguments[1].type

        rewriter.replace_matched_op(
            (
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[1]],
                    result_types=[VectorType(f32, [8])],
                ),
                llvm.StoreOp(
                    broadcast_op.vector,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecBrdcstSclF32x8Pfx(RewritePattern):
    """
    def vec_brdcst_scl_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                             src: f32 @ DRAM):
    # @instr {dst_data} = _mm256_set1_ps(*{src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: f32 @ DRAM
        assert op.arguments[2].type == f32, op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[2]],
                    result_types=[VectorType(f32, [8])],
                ),
                llvm_extra.MaskedStoreOp(
                    broadcast_op.vector,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecBrdcstSclF64x4(RewritePattern):
    """
    def vec_brdcst_scl_f64x4(dst: [f64][4] @ VEC_AVX2, src: f64 @ DRAM):
    # @instr {dst_data} = _mm256_set1_pd(*{src_data});
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: f64 @ DRAM
        assert op.arguments[1].type == f64, op.arguments[1].type

        rewriter.replace_matched_op(
            (
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[1]],
                    result_types=[VectorType(f64, [4])],
                ),
                llvm.StoreOp(
                    broadcast_op.vector,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecBrdcstSclF64x4Pfx(RewritePattern):
    """
    def vec_brdcst_scl_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                             src: f64 @ DRAM):
    # @instr {dst_data} = _mm256_set1_pd(*{src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_brdcst_scl_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: f64 @ DRAM
        assert op.arguments[2].type == f64, op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                broadcast_op := vector.BroadcastOp(
                    operands=[op.arguments[2]],
                    result_types=[VectorType(f64, [4])],
                ),
                llvm_extra.MaskedStoreOp(
                    broadcast_op.vector,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd2F32x8(RewritePattern):
    """
    def vec_fmadd2_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                     src2: [f32][8] @ VEC_AVX2, src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f32x8":
            return

        assert len(op.arguments) == 4
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        load0_op.dereferenced_value,
                        load1_op.dereferenced_value,
                        load2_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmadd2F32x8Pfx(RewritePattern):
    """
    def vec_fmadd2_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                         src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2,
                         src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f32x8_pfx":
            return

        assert len(op.arguments) == 5
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, llvm.LLVMPointerType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[4],
                    VectorType(f32, [8]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        load0_op.dereferenced_value,
                        load1_op.dereferenced_value,
                        load2_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd2F64x4(RewritePattern):
    """
    def vec_fmadd2_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                     src2: [f64][4] @ VEC_AVX2, src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f64x4":
            return

        assert len(op.arguments) == 4
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load0_op.dereferenced_value,
                        # rhs
                        load1_op.dereferenced_value,
                        # acc
                        load2_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmadd2F64x4Pfx(RewritePattern):
    """
    def vec_fmadd2_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                         src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2,
                         src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src3[i] + src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd2_f64x4_pfx":
            return

        assert len(op.arguments) == 5
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, llvm.LLVMPointerType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(op.arguments[4], VectorType(f64, [4])),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load0_op.dereferenced_value,
                        # rhs
                        load1_op.dereferenced_value,
                        # acc
                        load2_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecStoreF32x8(RewritePattern):
    """
    def vec_store_f32x8(dst: [f32][8] @ DRAM, src: [f32][8] @ VEC_AVX2):
    # @instr _mm256_storeu_ps(&{dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ DRAM
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecStoreF32x8Pfx(RewritePattern):
    """
    def vec_store_f32x8_pfx(m: size, dst: [f32][8] @ DRAM,
                        src: [f32][8] @ VEC_AVX2):
    # @instr _mm256_maskstore_ps(&{dst_data}, mm256_prefix_mask_epi32({m}), {src_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ DRAM
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecStoreF64x4(RewritePattern):
    """
    def vec_store_f64x4(dst: [f64][4] @ DRAM, src: [f64][4] @ VEC_AVX2):
    # @instr _mm256_storeu_pd(&{dst_data}, {src_data});
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ DRAM
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                llvm.StoreOp(
                    load_op.dereferenced_value,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecStoreF64x4Pfx(RewritePattern):
    """
    def vec_store_f64x4_pfx(m: size, dst: [f64][4] @ DRAM,
                        src: [f64][4] @ VEC_AVX2):
    # @instr _mm256_maskstore_pd(&{dst_data}, mm256_prefix_mask_epi64x({m}), {src_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_store_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ DRAM
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                llvm_extra.MaskedStoreOp(
                    load_op.dereferenced_value,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmaddRedF32x8(RewritePattern):
    """
    def vec_fmadd_red_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                        src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        dst[i] += src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd_red_f32x8":
            return

        assert len(op.arguments) == 3
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[0],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load0_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmaddRedF32x8Pfx(RewritePattern):
    """
    def vec_fmadd_red_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                            src1: [f32][8] @ VEC_AVX2,
                            src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] += src1[i] * src2[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd_red_f32x8_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load0_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmaddRedF64x4(RewritePattern):
    """
    def vec_fmadd_red_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                        src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        dst[i] += src1[i] * src2[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd_red_f64x4":
            return

        assert len(op.arguments) == 3
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load0_op := llvm.LoadOp(
                    op.arguments[0],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(op.arguments[2], VectorType(f64, [4])),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load0_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmaddRedF64x4Pfx(RewritePattern):
    """
    def vec_fmadd_red_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                            src1: [f64][4] @ VEC_AVX2,
                            src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] += src1[i] * src2[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd_red_f64x4_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load0_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load0_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd1F32x8(RewritePattern):
    """
    def vec_fmadd1_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                     src2: [f32][8] @ VEC_AVX2, src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        dst[i] = src1[i] * src2[i] + src3[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd1_f32x8":
            return

        assert len(op.arguments) == 4
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load3_op := llvm.LoadOp(op.arguments[3], VectorType(f32, [8])),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load3_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmadd1F32x8Pfx(RewritePattern):
    """
    def vec_fmadd1_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                         src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2,
                         src3: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {src3_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src1[i] * src2[i] + src3[i]


    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd1_f32x8_pfx":
            return

        assert len(op.arguments) == 5
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type
        # src3: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, llvm.LLVMPointerType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                load3_op := llvm.LoadOp(op.arguments[4], VectorType(f32, [8])),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load3_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f32, [8]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecFmadd1F64x4(RewritePattern):
    """
    def vec_fmadd1_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                     src2: [f64][4] @ VEC_AVX2, src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        dst[i] = src1[i] * src2[i] + src3[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd1_f64x4":
            return

        assert len(op.arguments) == 4
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(op.arguments[2], VectorType(f64, [4])),
                load3_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load3_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm.StoreOp(
                    fma_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecFmadd1F64x4Pfx(RewritePattern):
    """
    def vec_fmadd1_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                         src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2,
                         src3: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {src3_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(src3, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src1[i] * src2[i] + src3[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_fmadd1_f64x4_pfx":
            return

        assert len(op.arguments) == 5
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type
        # src3: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[4].type, llvm.LLVMPointerType), op.arguments[4].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                load3_op := llvm.LoadOp(op.arguments[4], VectorType(f64, [4])),
                fma_op := vector.FMAOp(
                    operands=[
                        # lhs
                        load1_op.dereferenced_value,
                        # rhs
                        load2_op.dereferenced_value,
                        # acc
                        load3_op.dereferenced_value,
                    ],
                    result_types=[
                        VectorType(f64, [4]),
                    ],
                ),
                llvm_extra.MaskedStoreOp(
                    fma_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecMulF32x8(RewritePattern):
    """
    def vec_mul_f32x8(dst: [f32][8] @ VEC_AVX2, src1: [f32][8] @ VEC_AVX2,
                  src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_mul_ps({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        dst[i] = src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_mul_f32x8":
            return

        assert len(op.arguments) == 3
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                mul_op := llvm.FMulOp(
                    load1_op.dereferenced_value,
                    load2_op.dereferenced_value,
                ),
                llvm.StoreOp(
                    mul_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecMulF32x8Pfx(RewritePattern):
    """
    def vec_mul_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src1: [f32][8] @ VEC_AVX2, src2: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_mul_ps({src1_data}, {src2_data});
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_mul_f32x8_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f32, [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f32, [8]),
                ),
                mul_op := llvm.FMulOp(
                    load1_op.dereferenced_value,
                    load2_op.dereferenced_value,
                ),
                llvm_extra.MaskedStoreOp(
                    mul_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecMulF64x4(RewritePattern):
    """
    def vec_mul_f64x4(dst: [f64][4] @ VEC_AVX2, src1: [f64][4] @ VEC_AVX2,
                  src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_mul_pd({src1_data}, {src2_data});
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        dst[i] = src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_mul_f64x4":
            return

        assert len(op.arguments) == 3
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(op.arguments[2], VectorType(f64, [4])),
                mul_op := llvm.FMulOp(
                    load1_op.dereferenced_value,
                    load2_op.dereferenced_value,
                ),
                llvm.StoreOp(
                    mul_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecMulF64x4Pfx(RewritePattern):
    """
    def vec_mul_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src1: [f64][4] @ VEC_AVX2, src2: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_mul_pd({src1_data}, {src2_data});
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = src1[i] * src2[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_mul_f64x4_pfx":
            return

        assert len(op.arguments) == 4
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src1: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type
        # src2: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[3].type, llvm.LLVMPointerType), op.arguments[3].type

        rewriter.replace_matched_op(
            (
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[3],
                    VectorType(f64, [4]),
                ),
                mul_op := llvm.FMulOp(
                    load1_op.dereferenced_value,
                    load2_op.dereferenced_value,
                ),
                llvm_extra.MaskedStoreOp(mul_op.res, op.arguments[1], mask_op.res),
            )
        )


class ConvertVecNegF32x8(RewritePattern):
    """
    def vec_neg_f32x8(dst: [f32][8] @ VEC_AVX2, src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_xor_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        dst[i] = -src[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_neg_f32x8":
            return

        assert len(op.arguments) == 2
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f32, [8]), [0.0] * 8)),
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f32, [8]),
                ),
                # missing llvm.FNeg
                neg_op := llvm.FSubOp(
                    zero_vec_op.result,
                    load_op.dereferenced_value,
                ),
                llvm.StoreOp(
                    neg_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecNegF32x8Pfx(RewritePattern):
    """
    def vec_neg_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2,
                      src: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_xor_ps({src_data}, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    assert m <= 8
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = -src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_neg_f32x8_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f32, [8]), [0.0] * 8)),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(op.arguments[2], VectorType(f32, [8])),
                # same as above
                neg_op := llvm.FSubOp(
                    zero_vec_op.result,
                    load_op.dereferenced_value,
                ),
                llvm_extra.MaskedStoreOp(
                    neg_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecNegF64x4(RewritePattern):
    """
    def vec_neg_f64x4(dst: [f64][4] @ VEC_AVX2, src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_xor_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL)));
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        dst[i] = -src[i]
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_neg_f64x4":
            return

        assert len(op.arguments) == 2
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[0].type, llvm.LLVMPointerType), op.arguments[0].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f64, [4]), [0.0] * 4)),
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(f64, [4]),
                ),
                # same as above
                neg_op := llvm.FSubOp(
                    zero_vec_op.result,
                    load_op.dereferenced_value,
                ),
                llvm.StoreOp(
                    neg_op.res,
                    op.arguments[0],
                ),
            )
        )


class ConvertVecNegF64x4Pfx(RewritePattern):
    """
    def vec_neg_f64x4_pfx(m: size, dst: [f64][4] @ VEC_AVX2,
                      src: [f64][4] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_xor_pd({src_data}, _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000LL)));
    assert m <= 4
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1
    for i in seq(0, 4):
        if i < m:
            dst[i] = -src[i]

    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_neg_f64x4_pfx":
            return

        assert len(op.arguments) == 3
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type
        # src: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[2].type, llvm.LLVMPointerType), op.arguments[2].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f64, [4]), [0.0] * 4)),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                load_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(f64, [4]),
                ),
                # same as above
                neg_op := llvm.FSubOp(
                    zero_vec_op.result,
                    load_op.dereferenced_value,
                ),
                llvm_extra.MaskedStoreOp(
                    neg_op.res,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecZeroF32x8Pfx(RewritePattern):
    """
    def vec_zero_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_ps();
    assert m <= 8
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f32x8_pfx":
            return

        assert len(op.arguments) == 2
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f32][8] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f32, [8]), [0.0] * 8)),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [8]), [0, 1, 2, 3, 4, 5, 6, 7]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i64, [8])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                llvm_extra.MaskedStoreOp(
                    zero_vec_op.result,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class ConvertVecZeroF64x4Pfx(RewritePattern):
    """
    def vec_zero_f32x8_pfx(m: size, dst: [f32][8] @ VEC_AVX2):
    # @instr {dst_data} = _mm256_setzero_ps();
    assert m <= 8
    assert stride(dst, 0) == 1
    for i in seq(0, 8):
        if i < m:
            dst[i] = 0.0
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "vec_zero_f64x4_pfx":
            return

        assert len(op.arguments) == 2
        # m: size
        assert op.arguments[0].type == i64, op.arguments[0].type
        # dst: [f64][4] @ VEC_AVX2
        assert isinstance(op.arguments[1].type, llvm.LLVMPointerType), op.arguments[1].type

        rewriter.replace_matched_op(
            (
                zero_vec_op := arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(VectorType(f64, [4]), [0.0] * 4)),
                indices_op := arith.ConstantOp(
                    DenseIntOrFPElementsAttr.create_dense_int(VectorType(i64, [4]), [0, 1, 2, 3]),
                ),
                broadcast_thresh_op := vector.BroadcastOp(
                    operands=[op.arguments[0]],
                    result_types=[VectorType(i32, [4])],
                ),
                mask_op := llvm.ICmpOp(
                    indices_op.result,
                    broadcast_thresh_op.vector,
                    IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64),
                ),
                llvm_extra.MaskedStoreOp(
                    zero_vec_op.result,
                    op.arguments[1],
                    mask_op.res,
                ),
            )
        )


class InlineBLASPass(ModulePass):
    name = "inline-blas"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSelect(),
                    ConvertVecAbsF32x8(),
                    ConvertVecAbsF32x8Pfx(),
                    ConvertVecAbsF64x4(),
                    ConvertVecAbsF64x4Pfx(),
                    ConvertVecAddRedF32x8(),
                    ConvertVecAddRedF32x8Pfx(),
                    ConvertVecAddRedF64x4(),
                    ConvertVecAddRedF64x4Pfx(),
                    ConvertVecCopyF32x8(),
                    ConvertVecCopyF32x8Pfx(),
                    ConvertVecCopyF64x4(),
                    ConvertVecCopyF64x4Pfx(),
                    ConvertVecLoadF32x8(),
                    ConvertVecLoadF32x8Pfx(),
                    ConvertVecLoadF64x4(),
                    ConvertVecLoadF64x4Pfx(),
                    ConvertVecReduceAddSclF32x8(),
                    ConvertVecReduceAddSclF64x4(),
                    ConvertVecZeroF32x8(),
                    ConvertVecZeroF64x4(),
                    ConvertVecAddF32x8(),
                    ConvertVecAddF32x8Pfx(),
                    ConvertVecAddF64x4(),
                    ConvertVecAddF64x4Pfx(),
                    ConvertVecBrdcstSclF32x8(),
                    ConvertVecBrdcstSclF32x8Pfx(),
                    ConvertVecBrdcstSclF64x4(),
                    ConvertVecBrdcstSclF64x4Pfx(),
                    ConvertVecFmadd2F32x8(),
                    ConvertVecFmadd2F32x8Pfx(),
                    ConvertVecFmadd2F64x4(),
                    ConvertVecFmadd2F64x4Pfx(),
                    ConvertVecStoreF32x8(),
                    ConvertVecStoreF32x8Pfx(),
                    ConvertVecStoreF64x4(),
                    ConvertVecStoreF64x4Pfx(),
                    ConvertVecFmaddRedF32x8(),
                    ConvertVecFmaddRedF32x8Pfx(),
                    ConvertVecFmaddRedF64x4(),
                    ConvertVecFmaddRedF64x4Pfx(),
                    ConvertVecFmadd1F32x8(),
                    ConvertVecFmadd1F32x8Pfx(),
                    ConvertVecFmadd1F64x4(),
                    ConvertVecFmadd1F64x4Pfx(),
                    ConvertVecMulF32x8(),
                    ConvertVecMulF32x8Pfx(),
                    ConvertVecMulF64x4(),
                    ConvertVecMulF64x4Pfx(),
                    ConvertVecNegF32x8(),
                    ConvertVecNegF32x8Pfx(),
                    ConvertVecNegF64x4(),
                    ConvertVecNegF64x4Pfx(),
                    ConvertVecZeroF32x8Pfx(),
                    ConvertVecZeroF64x4Pfx(),
                ]
            )
        ).rewrite_module(m)
