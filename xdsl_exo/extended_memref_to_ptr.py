# monkey patch. merge this upstream

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, memref, ptr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.convert_memref_to_ptr import ConvertLoadPattern, ConvertStorePattern, ConvertSubviewPattern
from xdsl.utils.hints import isa


@dataclass
class ConvertCastOp(RewritePattern):
    """Converts memref.cast to identity (forwards source)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, memref.MemRefType)
        rewriter.replace_matched_op((), (op.source,))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
    """Converts memref.reinterpret_cast to ptr cast."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            (
                ptr_cast := ptr.ToPtrOp(op.source),
                builtin.UnrealizedConversionCastOp.get([ptr_cast.res], [op.result.type]),
            )
        )


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    """
    monkey patched ConvertMemRefToPtr
    """

    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStorePattern(),
                    ConvertLoadPattern(),
                    ConvertSubviewPattern(),
                    ConvertCastOp(),
                    ConvertReinterpretCastOp(),
                ]
            )
        ).rewrite_module(op)
