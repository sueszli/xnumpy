"""
to be merged into upstream xDSL
"""

from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import builtin, memref, ptr
from xdsl.dialects.builtin import I1, AnyFloatConstr, IntegerAttr, VectorType, i32
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import Attribute, IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.convert_memref_to_ptr import ConvertLoadPattern, ConvertStorePattern, ConvertSubviewPattern
from xdsl.utils.hints import isa


@irdl_op_definition
class FAbsOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyFloatConstr | VectorType.constr(AnyFloatConstr))

    name = "llvm.intr.fabs"

    input = operand_def(T)
    result = result_def(T)

    assembly_format = "`(` operands `)` attr-dict `:` functional-type(operands, results)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, input: Operation | SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    name = "llvm.intr.masked.store"

    value = operand_def(AnyFloatConstr | VectorType.constr(AnyFloatConstr))
    data = operand_def(LLVMPointerType)
    mask = operand_def(I1 | VectorType[I1])
    alignment = prop_def(IntegerAttr[i32])

    assembly_format = "$value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($data)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(
        self,
        value: Operation | SSAValue,
        data: Operation | SSAValue,
        mask: Operation | SSAValue,
        alignment: int = 32,
    ):
        super().__init__(
            operands=[value, data, mask],
            result_types=[],
            properties={
                "alignment": IntegerAttr(alignment, 32),
            },
        )


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [FAbsOp, MaskedStoreOp],
    [],
)


#
# ConvertMemRefToPtr extensions
#


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
    Extended ConvertMemRefToPtr — adds memref.cast and memref.reinterpret_cast
    support on top of upstream ConvertStorePattern, ConvertLoadPattern,
    ConvertSubviewPattern. To be upstreamed into xDSL's convert_memref_to_ptr.
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
