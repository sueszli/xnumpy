from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, memref, ptr, scf
from xdsl.dialects.builtin import DYNAMIC_INDEX, I1, AnyFloatConstr, IndexType, IntegerAttr, StringAttr, VectorType, i1, i32, i64
from xdsl.dialects.llvm import FastMathAttr, LLVMPointerType
from xdsl.ir import BlockArgument, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import AnyAttr, Attribute, IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def, traits_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.traits import Pure, SameOperandsAndResultType
from xdsl.transforms.convert_memref_to_ptr import ConvertLoadPattern, ConvertStorePattern, ConvertSubviewPattern, get_offset_pointer
from xdsl.utils.hints import isa


@irdl_op_definition
class FAbsOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/commit/5f30cfdd78d8dbaddb70b15358c406ab63524b5b
    T: ClassVar = VarConstraint("T", AnyFloatConstr | VectorType.constr(AnyFloatConstr))

    name = "llvm.intr.fabs"

    input = operand_def(T)
    result = result_def(T)

    assembly_format = "`(` operands `)` attr-dict `:` functional-type(operands, results)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, input: Operation | SSAValue, result_type: Attribute):
        super().__init__(operands=[input], result_types=[result_type])


@irdl_op_definition
class FNegOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/pull/5697
    T: ClassVar = VarConstraint("T", AnyFloatConstr | VectorType.constr(AnyFloatConstr))

    name = "llvm.fneg"

    arg = operand_def(T)
    res = result_def(T)

    fastmathFlags = prop_def(FastMathAttr, default_value=FastMathAttr(None))

    traits = traits_def(Pure(), SameOperandsAndResultType())

    assembly_format = "$arg attr-dict `:` type($arg)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, arg: Operation | SSAValue, fast_math: FastMathAttr | None = None):
        if fast_math is None:
            fast_math = FastMathAttr(None)
        super().__init__(operands=[arg], result_types=[SSAValue.get(arg).type], properties={"fastmathFlags": fast_math})


@irdl_op_definition
class FCmpOp(IRDLOperation):
    name = "llvm.fcmp"

    T: ClassVar = VarConstraint("T", AnyFloatConstr)

    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(I1)
    predicate = prop_def(StringAttr)

    traits = traits_def(Pure())

    assembly_format = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs)"

    def __init__(self, lhs: Operation | SSAValue, rhs: Operation | SSAValue, predicate: str):
        super().__init__(operands=[lhs, rhs], result_types=[i1], properties={"predicate": StringAttr(predicate)})


@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "llvm.select"

    T: ClassVar = VarConstraint("T", AnyAttr())

    cond = operand_def(I1)
    lhs = operand_def(T)
    rhs = operand_def(T)
    res = result_def(T)

    traits = traits_def(Pure())

    assembly_format = "$cond `,` $lhs `,` $rhs attr-dict `:` type($cond) `,` type($res)"

    def __init__(self, cond: Operation | SSAValue, lhs: Operation | SSAValue, rhs: Operation | SSAValue):
        super().__init__(operands=[cond, lhs, rhs], result_types=[SSAValue.get(lhs).type])


@irdl_op_definition
class MaskedStoreOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/commit/726e2c40df108e700fc9eab071555adc4fff8b75
    name = "llvm.intr.masked.store"

    value = operand_def(AnyFloatConstr | VectorType.constr(AnyFloatConstr))
    data = operand_def(LLVMPointerType)
    mask = operand_def(I1 | VectorType[I1])
    alignment = prop_def(IntegerAttr[i32])

    assembly_format = "$value `,` $data `,` $mask attr-dict `:` type($value) `,` type($mask) `into` type($data)"

    irdl_options = (ParsePropInAttrDict(),)

    def __init__(self, value: Operation | SSAValue, data: Operation | SSAValue, mask: Operation | SSAValue, alignment: int = 32):
        super().__init__(operands=[value, data, mask], result_types=[], properties={"alignment": IntegerAttr(alignment, 32)})


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [
        FAbsOp,
        FCmpOp,
        SelectOp,
        MaskedStoreOp,
    ],
    [],
)


#
# ConvertMemRefToPtr extensions
#


def _loop_ub_as_i64(index: SSAValue) -> SSAValue | None:
    # exo emits `%i_idx = arith.index_cast(%i_i64)` before using the IV as an index. unwrap it.
    iv = index.op.input if isinstance(index, OpResult) and isinstance(index.op, arith.IndexCastOp) else index
    if not isinstance(iv, BlockArgument) or iv.index != 0 or not isinstance(for_op := iv.block.parent_op(), scf.ForOp):
        return None
    ub = for_op.ub
    return ub if ub.type == i64 else None


def _get_dynamic_target_ptr(memref_val: SSAValue, memref_type: builtin.MemRefType, indices: list[SSAValue], rewriter: PatternRewriter) -> SSAValue:
    shape = memref_type.get_shape()
    ins = rewriter.insert_op
    iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result

    def to_i64(val: SSAValue) -> SSAValue:
        # unwrap arith.index_cast(i64 → index) to recover the original i64
        if isinstance(val, OpResult) and isinstance(val.op, arith.IndexCastOp) and val.op.input.type == i64:
            return val.op.input
        return val

    def dim_size(i: int) -> SSAValue:  # static -> constant; dynamic -> scf.for ub
        if shape[i] != DYNAMIC_INDEX:
            return iconst(shape[i])
        ub = _loop_ub_as_i64(indices[i])
        assert ub is not None, f"dynamic dim {i}: index is not an scf.for induction variable"
        return ub

    # strides[rank-1] = 1
    # strides[i] = strides[i+1]*dim[i+1]  (row-major, right-to-left)
    strides: list[SSAValue] = [iconst(1)] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size(i + 1))).res

    # flat offset = sum(indices[i] * strides[i])
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        term = ins(llvm.MulOp(to_i64(idx), stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

    # pointer arithmetic
    ptr_base = ins(ptr.ToPtrOp(memref_val)).res
    if flat is None:
        return ptr_base
    bytes_flat = ins(llvm.MulOp(flat, iconst(memref_type.element_type.size))).res
    bytes_flat_idx = ins(arith.IndexCastOp(bytes_flat, IndexType())).result
    return get_offset_pointer(ptr_base, bytes_flat_idx, rewriter)


@dataclass
class DynamicConvertStorePattern(RewritePattern):
    # like ConvertStorePattern but for dynamic shapes.
    # dim sizes recovered from scf.for upper bound.
    # (can't use memref.dim because types are erased later by RewriteMemRefTypes)
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if memref_type.has_static_shape() or not isa(memref_type.layout, builtin.NoneAttr):
            return
        target_ptr = _get_dynamic_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, ptr.StoreOp(target_ptr, op.value))


@dataclass
class DynamicConvertLoadPattern(RewritePattern):
    # like ConvertLoadPattern but for dynamic shapes.
    # same as above.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if memref_type.has_static_shape() or not isa(memref_type.layout, builtin.NoneAttr):
            return
        target_ptr = _get_dynamic_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, ptr.LoadOp(target_ptr, memref_type.element_type))


@dataclass
class ConvertCmpiPattern(RewritePattern):
    # arith.cmpi uses predicate ints 0-9 identical to llvm.icmp's ICmpPredicateFlag ints
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(llvm.ICmpOp(op.lhs, op.rhs, IntegerAttr(op.predicate.value.data, i64)))


@dataclass
class ConvertCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.CastOp, rewriter: PatternRewriter, /):
        assert isa(op.source.type, builtin.MemRefType)
        rewriter.replace_matched_op((), (op.source,))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
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
    # https://github.com/xdslproject/xdsl/pull/5692
    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DynamicConvertStorePattern(),
                    DynamicConvertLoadPattern(),
                    ConvertStorePattern(),
                    ConvertLoadPattern(),
                    ConvertSubviewPattern(),
                    ConvertCastOp(),
                    ConvertReinterpretCastOp(),
                ]
            )
        ).rewrite_module(op)
