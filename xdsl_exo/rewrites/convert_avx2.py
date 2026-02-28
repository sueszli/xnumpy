from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref, vector
from xdsl.dialects.builtin import Float32Type, IndexType, IntegerAttr, MemRefType, ModuleOp, VectorType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo


class ConvertMM256StoreuPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_storeu_ps":
            return

        # preconditions
        assert len(op.arguments) == 2
        assert isinstance(op.arguments[0].type, MemRefType)

        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(Float32Type(), [8]),
                ),
                llvm.StoreOp(
                    load_op.result,
                    op.arguments[0],
                ),
            )
        )


class ConvertMM256FmaddPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_fmadd_ps":
            return

        assert len(op.arguments) == 3
        assert isinstance(op.arguments[0].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := llvm.LoadOp(
                    op.arguments[0],
                    VectorType(Float32Type(), [8]),
                ),
                load1_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(Float32Type(), [8]),
                ),
                load2_op := llvm.LoadOp(
                    op.arguments[2],
                    VectorType(Float32Type(), [8]),
                ),
                fma_op := vector.FMAOp(
                    operands=[
                        load1_op.dereferenced_value,
                        load2_op.dereferenced_value,
                        load0_op.dereferenced_value,
                    ],
                    result_types=[VectorType(Float32Type(), [8])],
                ),
                llvm.StoreOp(fma_op.res, op.arguments[0], [zero_op.result]),
            )
        )


class ConvertMM256BroadcastSsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_broadcast_ss":
            return

        assert len(op.arguments) == 2
        assert isinstance(op.arguments[0].type, MemRefType)
        assert isinstance(op.arguments[1].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                scalar_load_op := memref.LoadOp.get(
                    op.arguments[1],
                    [zero_op.result],
                ),
                broadcast_op := vector.BroadcastOp(
                    operands=[scalar_load_op.results[0]],
                    result_types=[VectorType(Float32Type(), [8])],
                ),
                llvm.StoreOp(
                    broadcast_op.results[0],
                    op.arguments[0],
                    [zero_op.result],
                ),
            )
        )


class ConvertMM256LoaduPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.InstrOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_loadu_ps":
            return

        assert len(op.arguments) == 2
        assert isinstance(op.arguments[0].type, MemRefType)
        assert isinstance(op.arguments[1].type, MemRefType)

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := llvm.LoadOp(
                    op.arguments[1],
                    VectorType(Float32Type(), [8]),
                ),
                llvm.StoreOp(
                    load_op.result,
                    op.arguments[1],
                    [zero_op.result],
                ),
            )
        )


class ConvertAVX2Pass(ModulePass):
    name = "convert-avx2"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertMM256StoreuPsOp(),
                    ConvertMM256FmaddPsOp(),
                    ConvertMM256BroadcastSsOp(),
                    ConvertMM256LoaduPsOp(),
                ]
            )
        ).rewrite_module(m)
