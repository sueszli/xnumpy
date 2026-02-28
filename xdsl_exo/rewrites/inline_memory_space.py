from typing import cast

from xdsl.context import Context
from xdsl.dialects import memref
from xdsl.dialects.builtin import MemRefType, ModuleOp, NoneAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        # convert tensor allocs only
        if not isinstance(op.result.type, MemRefType):
            return

        assert isinstance(op.result.type.memory_space, StringAttr), "memory space should be a string"

        # only convert DRAM allocs
        if cast(StringAttr, op.result.type.memory_space).data != "DRAM":
            return

        rewriter.replace_matched_op(
            memref.AllocOp.get(
                op.result.type.element_type,
                shape=op.result.type.shape,
                layout=op.result.type.layout,
                memory_space=op.mem,
            )
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        # convert tensor writes only
        if len(op.indices) <= 1:
            return

        input_type = cast(MemRefType, op.input.type)
        output_type = cast(MemRefType, op.result.type)

        # if no input memory space is specified, we cannot convert
        if isinstance(input_type.memory_space, NoneAttr):
            return

        if not isinstance(output_type.memory_space, NoneAttr) and input_type.memory_space.data == output_type.memory_space.data:
            return

        rewriter.replace_matched_op(
            exo.WindowOp(
                op.input,
                op.indices,
                MemRefType(
                    op.result.type.element_type,
                    op.result.type.shape,
                    op.result.type.layout,
                    StringAttr(input_type.memory_space.data),
                ),
            )
        )


class InlineMemorySpacePass(ModulePass):
    name = "inline-memory-space"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ConvertWindowOp()])).rewrite_module(m)
