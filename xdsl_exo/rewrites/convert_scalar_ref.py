from xdsl.context import Context
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import FunctionType, IndexType, IntegerAttr, MemRefType, ModuleOp, NoneAttr, i64
from xdsl.ir import BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo


class ConvertRedundantReads(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # replace redundant reads
        if isinstance(op.input.type, MemRefType) and op.result.type == op.input.type and len(op.indices) == 0:
            rewriter.replace_matched_op((), (op.input,))
            return

        # convert scalar reads only
        if isinstance(op.input.type, MemRefType):
            return

        # replace index -> x with index cast
        if isinstance(op.input.type, IndexType) and not isinstance(op.result.type, IndexType):
            rewriter.replace_matched_op(arith.IndexCastOp(op.input, op.result.type))

        if op.input.type == op.result.type and isinstance(op.input, BlockArgument):
            # replace x -> x with x
            rewriter.replace_matched_op((), (op.input,))


class ConvertAllocToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        if isinstance(op.result.type, MemRefType):
            return

        rewriter.replace_matched_op(exo.AllocOp(op.mem.data, MemRefType(op.result.type, [1], NoneAttr(), op.mem)))


class ConvertReadToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        if not isinstance(op.input.type, MemRefType) or op.input.type.get_shape() != (1,) or len(op.indices) != 0:
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, i64)),
                exo.ReadOp(op.input, [zero_op.result], [1], op.result.type),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertAssignToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        if not isinstance(op.input.type, MemRefType) or op.input.type.get_shape() != (1,) or len(op.indices) != 0:
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, i64)),
                exo.AssignOp(op.value, op.input, [zero_op.result], [1]),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertReduceToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReduceOp, rewriter: PatternRewriter):
        if not isinstance(op.input.type, MemRefType) or op.input.type.get_shape() != (1,) or len(op.indices) != 0:
            return

        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, i64)),
                exo.ReduceOp(op.value, op.input, [zero_op.result], [1]),
            )
        )
        zero_op.result.name_hint = "c0"


class ConvertScalarFuncArgsToTensor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        for idx, arg in enumerate(op.args):
            # ignore tensor types
            if isinstance(arg.type, MemRefType):
                continue

            mutated = any(isinstance(use.operation, exo.AssignOp) and use.operation.input == arg or isinstance(use.operation, exo.ReduceOp) and use.operation.input == arg for use in arg.uses)

            # ignore unmutated scalar types, these can stay as is
            if not mutated:
                continue

            func_type = FunctionType.from_lists(
                (
                    *(arg.type for arg in op.args[:idx]),
                    MemRefType(arg.type, [1], NoneAttr()),
                    *(arg.type for arg in op.args[idx + 1 :]),
                ),
                op.function_type.outputs,
            )

            # rewrite function signature
            body = op.detach_region(op.body)
            new_arg = rewriter.insert_block_argument(body.block, idx, (MemRefType(arg.type, [1], NoneAttr())))
            rewriter.replace_all_uses_with(
                arg,
                new_arg,
            )
            rewriter.erase_block_argument(arg, idx)

            rewriter.replace_matched_op(func.FuncOp(op.sym_name.data, func_type, body, op.sym_visibility))


class ConvertScalarRefPass(ModulePass):
    name = "convert-scalar-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertRedundantReads(),
                    ConvertAllocToTensor(),
                    ConvertReadToTensor(),
                    ConvertAssignToTensor(),
                    ConvertReduceToTensor(),
                    ConvertScalarFuncArgsToTensor(),
                ]
            ),
        ).rewrite_module(m)
