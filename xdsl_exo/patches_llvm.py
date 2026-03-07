from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, memref, scf
from xdsl.dialects.builtin import DYNAMIC_INDEX, I1, AnyFloatConstr, IntegerAttr, MemRefType, StringAttr, UnrealizedConversionCastOp, i1, i64
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import BlockArgument, Dialect, Operation, OpResult, SSAValue
from xdsl.irdl import AnyAttr, IRDLOperation, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def, traits_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.traits import Pure
from xdsl.transforms.convert_memref_to_ptr import ConvertCastOp
from xdsl.utils.hints import isa


@irdl_op_definition
class FCmpOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/pull/5706
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
    # https://github.com/xdslproject/xdsl/pull/5707
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


LLVMIntrinsics = Dialect(
    "llvm.intr",
    [
        FCmpOp,
        SelectOp,
    ],
    [],
)


#
# memref lowering: direct emission of llvm ops (no ptr/arith dialects)
#


def _unwrap_i64(val: SSAValue) -> SSAValue:
    # peek through unrealized_cast(x:i64 -> index) to recover the original i64
    if isinstance(val, OpResult) and isinstance(val.op, UnrealizedConversionCastOp):
        inputs = list(val.op.operands)
        if len(inputs) == 1 and inputs[0].type == i64:
            return inputs[0]
    return val


def _loop_ub_as_i64(index: SSAValue) -> SSAValue | None:
    # exo emits unrealized_cast(iv:i64 -> index) before using a loop IV as a memref index; unwrap it
    if isinstance(index, OpResult) and isinstance(index.op, UnrealizedConversionCastOp):
        inputs = list(index.op.operands)
        iv = inputs[0] if len(inputs) == 1 else index
    else:
        iv = index
    if not isinstance(iv, BlockArgument) or iv.index != 0 or not isinstance(for_op := iv.block.parent_op(), scf.ForOp):
        return None
    ub = for_op.ub
    return ub if ub.type == i64 else None


def _get_target_ptr(memref_val: SSAValue, memref_type: builtin.MemRefType, indices: list[SSAValue], rewriter: PatternRewriter) -> SSAValue:
    # compute an llvm.ptr to memref_val[indices]; emits pure llvm ops, no ptr/arith dialect
    shape = memref_type.get_shape()
    ins = rewriter.insert_op
    iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result

    def dim_size(i: int) -> SSAValue:
        if shape[i] != DYNAMIC_INDEX:
            return iconst(shape[i])
        ub = _loop_ub_as_i64(indices[i])
        assert ub is not None, f"dynamic dim {i}: index is not an scf.for induction variable"
        return ub

    # row-major strides: stride[rank-1]=1, stride[i]=stride[i+1]*dim[i+1]
    strides: list[SSAValue] = [iconst(1)] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size(i + 1))).res

    # flat element offset = sum(index_i * stride_i)
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        term = ins(llvm.MulOp(_unwrap_i64(idx), stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

    # cast memref -> llvm.ptr; ReconcileUnrealizedCastsPass folds the pair after RewriteMemRefTypes
    base_ptr = ins(UnrealizedConversionCastOp.get([memref_val], [LLVMPointerType()])).results[0]
    if flat is None:
        return base_ptr

    byte_offset = ins(llvm.MulOp(flat, iconst(memref_type.element_type.size))).res
    ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
    target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
    return ins(llvm.IntToPtrOp(target_int)).output


@dataclass
class ConvertLoadPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, llvm.LoadOp(target_ptr, op.res.type))


@dataclass
class ConvertStorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, llvm.StoreOp(op.value, target_ptr))


@dataclass
class ConvertSubviewPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        assert isa(src_type := op.source.type, builtin.MemRefType)
        if not isa(src_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        src_shape = src_type.get_shape()
        assert all(d != DYNAMIC_INDEX for d in src_shape), "dynamic source dims in subview not supported"

        ins = rewriter.insert_op
        iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result

        # merge static_offsets (constants) and dynamic offsets (SSA values) into one list
        all_offsets: list[SSAValue] = []
        dyn_iter = iter(op.offsets)
        for soff in op.static_offsets.iter_values():
            if soff == DYNAMIC_INDEX:
                all_offsets.append(next(dyn_iter))
            else:
                all_offsets.append(iconst(soff))

        # row-major strides: stride[rank-1]=1, stride[i]=stride[i+1]*dim[i+1]
        strides: list[SSAValue] = [iconst(1)] * len(src_shape)
        for i in range(len(src_shape) - 2, -1, -1):
            strides[i] = ins(llvm.MulOp(strides[i + 1], iconst(src_shape[i + 1]))).res

        # flat element offset = sum(offset_i * stride_i)
        flat: SSAValue | None = None
        for offset, stride in zip(all_offsets, strides):
            term = ins(llvm.MulOp(_unwrap_i64(offset), stride)).res
            flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

        base_ptr = ins(UnrealizedConversionCastOp.get([op.source], [LLVMPointerType()])).results[0]
        if flat is not None:
            byte_offset = ins(llvm.MulOp(flat, iconst(src_type.element_type.size))).res
            ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
            target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
            result_ptr = ins(llvm.IntToPtrOp(target_int)).output
        else:
            result_ptr = base_ptr

        # wrap as result MemRefType so downstream loads/stores see the correct shape for stride computation
        rewriter.replace_op(op, UnrealizedConversionCastOp.get([result_ptr], [op.result.type]))


@dataclass
class ConvertCmpiPattern(RewritePattern):
    # arith.cmpi uses predicate ints 0-9 identical to llvm.icmp's ICmpPredicateFlag ints
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CmpiOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(llvm.ICmpOp(op.lhs, op.rhs, IntegerAttr(op.predicate.value.data, i64)))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /):
        # both types are MemRefType; after RewriteMemRefTypes both become llvm.ptr -> identity cast -> Reconcile removes it
        rewriter.replace_matched_op(UnrealizedConversionCastOp.get([op.source], [op.result.type]))


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSubviewPattern(),
                    ConvertStorePattern(),
                    ConvertLoadPattern(),
                    ConvertCastOp(),
                    ConvertReinterpretCastOp(),
                ]
            )
        ).rewrite_module(op)


#
# erase MemRefType on all values (runs after load/store/subview patterns consumed shape info)
#


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()


#
# arith.addi:i64 -> llvm.add (loop increment emitted by ConvertScfToCf)
#


@dataclass
class ConvertArithAddiI64(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddiOp, rewriter: PatternRewriter, /):
        if op.result.type != i64:
            return
        rewriter.replace_matched_op(llvm.AddOp(op.lhs, op.rhs))
