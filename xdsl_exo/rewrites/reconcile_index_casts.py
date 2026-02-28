from xdsl.context import Context
from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern


class ReconcileIndexCasts(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.IndexCastOp, rewriter: PatternRewriter):
        if len(op.result.uses) == 0:
            rewriter.erase_matched_op()
            return

        # replace x -> y -> x cast with x
        if not isinstance(op.input.owner, arith.IndexCastOp):
            return

        if op.input.owner.input.type != op.result.type:
            return

        rewriter.replace_matched_op((), (op.input.owner.input,))


class ReconcileIndexCastsPass(ModulePass):
    name = "reconcile-index-casts"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ReconcileIndexCasts()]),
            walk_reverse=True,
        ).rewrite_module(m)
