from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from xdsl.context import Context
from xdsl.dialects import builtin, llvm, memref
from xdsl.dialects.builtin import DYNAMIC_INDEX, I1, AnyFloatConstr, IntegerAttr, IntegerType, MemRefType, StringAttr, UnrealizedConversionCastOp, i1, i64
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.ir import Block, BlockArgument, Operation, OpResult, SSAValue
from xdsl.irdl import AnyAttr, AttrSizedOperandSegments, IRDLOperation, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def, successor_def, traits_def, var_operand_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.traits import IsTerminator, Pure
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


@irdl_op_definition
class BrOp(IRDLOperation):
    name = "llvm.br"
    arguments = var_operand_def()
    successor = successor_def()
    traits = traits_def(IsTerminator())
    assembly_format = "$successor (`(` $arguments^ `:` type($arguments) `)`)? attr-dict"

    def __init__(self, dest: Block, *ops: Operation | SSAValue):
        super().__init__(operands=[[op for op in ops]], successors=[dest])


@irdl_op_definition
class CondBrOp(IRDLOperation):
    # https://github.com/xdslproject/xdsl/pull/5710
    name = "llvm.cond_br"
    cond = operand_def(IntegerType(1))
    then_arguments = var_operand_def()
    else_arguments = var_operand_def()
    irdl_options = (AttrSizedOperandSegments(as_property=True),)
    then_block = successor_def()
    else_block = successor_def()
    traits = traits_def(IsTerminator())
    assembly_format = "$cond `,` $then_block (`(` $then_arguments^ `:` type($then_arguments) `)`)? `,`" " $else_block (`(` $else_arguments^ `:` type($else_arguments) `)`)? attr-dict"

    def __init__(self, cond: Operation | SSAValue, then_block: Block, then_ops: Sequence[Operation | SSAValue], else_block: Block, else_ops: Sequence[Operation | SSAValue]):
        super().__init__(operands=[cond, then_ops, else_ops], successors=[then_block, else_block])


# `memref` -> `llvm.ptr` lowering: replace structured memory ops with raw pointer arithmetic
#
# Standard MLIR lowers memref through a "descriptor" struct (base ptr, offset, sizes, strides).
# We skip that and go straight to flat pointer math because Exo only emits statically-shaped,
# row-major memrefs with no affine maps. The descriptor is unnecessary overhead.
#
# Pipeline order matters:
# ----------------------
#     1. ExtendedConvertMemRefToPtr   — rewrite load/store/subview while shape info is still on the MemRefType
#     2. RewriteMemRefTypes           — erase MemRefType -> llvm.ptr everywhere
#     3. reconcile-unrealized-casts   — clean up identity casts left behind
#
# Example (ConvertLoadPattern):
# -----------------------------
#     memref.load %buf[%i, %j] : memref<4x8xf32>
#     =>
#     %stride = llvm.mul %1, 8          ; stride[0] = dim[1] = 8
#     %off0   = llvm.mul %i, %stride    ; i * 8
#     %off1   = llvm.mul %j, %1         ; j * 1
#     %flat   = llvm.add %off0, %off1   ; i*8 + j
#     %bytes  = llvm.mul %flat, 4       ; * sizeof(f32)
#     %ptr    = llvm.inttoptr ...       ; base + bytes
#     %val    = llvm.load %ptr : f32


def _unwrap_i64(val: SSAValue) -> SSAValue:
    # peek through unrealized_cast(x:i64 -> index) to recover the original i64
    if isinstance(val, OpResult) and isinstance(val.op, UnrealizedConversionCastOp):
        inputs = list(val.op.operands)
        if len(inputs) == 1 and inputs[0].type == i64:
            return inputs[0]
    return val


def _loop_upper_bound_as_i64(index: SSAValue) -> SSAValue | None:
    # for dynamic dims: walk index -> block_arg#0 -> find llvm.icmp in the loop header -> extract the bound
    # e.g. `icmp slt %iv, %N` => return %N as the dim size
    if isinstance(index, OpResult) and isinstance(index.op, UnrealizedConversionCastOp):
        inputs = list(index.op.operands)
        iv = inputs[0] if len(inputs) == 1 else index
    else:
        iv = index
    if not isinstance(iv, BlockArgument) or iv.index != 0:
        return None
    for op in iv.block.ops:
        if isinstance(op, llvm.ICmpOp):
            if op.lhs == iv:
                return op.rhs if op.rhs.type == i64 else None
            if op.rhs == iv:
                return op.lhs if op.lhs.type == i64 else None
    return None


def _iconst(ins, n: int) -> SSAValue:
    return ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result


def _offset_ptr(
    base: SSAValue,
    indices: Sequence[SSAValue],
    rank: int,
    dim_size_fn,
    elem_size: int,
    ins,
) -> SSAValue:
    # flat_byte_offset = sum(index_i * stride_i) * elem_size, then base_ptr + flat_byte_offset

    # row-major strides: stride[last]=1, stride[i]=stride[i+1]*dim[i+1]
    strides: list[SSAValue] = [_iconst(ins, 1)] * rank
    for i in range(rank - 2, -1, -1):
        strides[i] = ins(llvm.MulOp(strides[i + 1], dim_size_fn(i + 1))).res

    # flat element offset = sum(index_i * stride_i)
    flat: SSAValue | None = None
    for idx, stride in zip(indices, strides):
        term = ins(llvm.MulOp(_unwrap_i64(idx), stride)).res
        flat = term if flat is None else ins(llvm.AddOp(flat, term)).res

    # cast base memref -> llvm.ptr, then add byte offset via ptr-to-int round-trip
    base_ptr = ins(UnrealizedConversionCastOp.get([base], [LLVMPointerType()])).results[0]
    if flat is None:
        return base_ptr

    byte_offset = ins(llvm.MulOp(flat, _iconst(ins, elem_size))).res
    ptr_int = ins(llvm.PtrToIntOp(base_ptr)).output
    target_int = ins(llvm.AddOp(ptr_int, byte_offset)).res
    return ins(llvm.IntToPtrOp(target_int)).output


def _get_target_ptr(memref_val: SSAValue, memref_type: builtin.MemRefType, indices: list[SSAValue], rewriter: PatternRewriter) -> SSAValue:
    # compute &memref_val[indices] as an llvm.ptr; uses static shape when available, falls back to loop bound for dynamic dims
    shape = memref_type.get_shape()
    ins = rewriter.insert_op

    def dim_size(i: int) -> SSAValue:
        if shape[i] != DYNAMIC_INDEX:
            return _iconst(ins, shape[i])
        ub = _loop_upper_bound_as_i64(indices[i])
        assert ub is not None
        return ub

    return _offset_ptr(memref_val, indices, len(shape), dim_size, memref_type.element_type.size, ins)


@dataclass
class ConvertLoadPattern(RewritePattern):
    # memref.load %buf[%i, %j] => ptr arithmetic + llvm.load
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.LoadOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, llvm.LoadOp(target_ptr, op.res.type))


@dataclass
class ConvertStorePattern(RewritePattern):
    # memref.store %val, %buf[%i, %j] => ptr arithmetic + llvm.store
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.StoreOp, rewriter: PatternRewriter, /):
        assert isa(memref_type := op.memref.type, builtin.MemRefType)
        if not isa(memref_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        target_ptr = _get_target_ptr(op.memref, memref_type, list(op.indices), rewriter)
        rewriter.replace_op(op, llvm.StoreOp(op.value, target_ptr))


@dataclass
class ConvertSubviewPattern(RewritePattern):
    # memref.subview %buf[offsets] => ptr to the start of the slice
    #
    # subview carries both static offsets (baked into the op) and dynamic offsets (SSA values).
    # MLIR encodes "this offset is dynamic" by setting the static value to DYNAMIC_INDEX (-1);
    # the actual SSA value then comes from the op.offsets list in order.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        assert isa(src_type := op.source.type, builtin.MemRefType)
        if not isa(src_type.layout, builtin.NoneAttr):
            return  # skip affine map layouts
        src_shape = src_type.get_shape()
        assert all(d != DYNAMIC_INDEX for d in src_shape), "dynamic source dims in subview not supported"

        ins = rewriter.insert_op

        # merge static_offsets (constants) and dynamic offsets (SSA values) into one list
        all_offsets: list[SSAValue] = []
        dyn_iter = iter(op.offsets)
        for soff in op.static_offsets.iter_values():
            if soff == DYNAMIC_INDEX:
                all_offsets.append(next(dyn_iter))
            else:
                all_offsets.append(_iconst(ins, soff))

        result_ptr = _offset_ptr(op.source, all_offsets, len(src_shape), lambda i: _iconst(ins, src_shape[i]), src_type.element_type.size, ins)

        # wrap result as MemRefType so downstream load/store patterns still see the right shape for stride computation
        rewriter.replace_op(op, UnrealizedConversionCastOp.get([result_ptr], [op.result.type]))


@dataclass
class ConvertReinterpretCastOp(RewritePattern):
    # reinterpret_cast just changes the memref metadata (shape/strides) without moving data.
    # after RewriteMemRefTypes erases all MemRefType -> llvm.ptr, both sides become the same type,
    # so this turns into an identity cast that reconcile-unrealized-casts will remove.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.ReinterpretCastOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(UnrealizedConversionCastOp.get([op.source], [op.result.type]))


@dataclass(frozen=True)
class ExtendedConvertMemRefToPtr(ModulePass):
    name = "extended-convert-memref-to-ptr"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertCastOp(),
                    ConvertLoadPattern(),
                    ConvertStorePattern(),
                    ConvertSubviewPattern(),
                    ConvertReinterpretCastOp(),
                ]
            )
        ).rewrite_module(op)


#
# erase MemRefType on all remaining values
# (runs after the patterns above consumed shape info)
#
# Before:  %x : memref<4x8xf32>
# After:   %x : !llvm.ptr
#


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType) -> llvm.LLVMPointerType:
        return llvm.LLVMPointerType()
