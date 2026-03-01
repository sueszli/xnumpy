from dataclasses import dataclass
from functools import reduce

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, StringAttr, UnrealizedConversionCastOp, i64
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from xdsl_exo.dialects import exo


def compute_memref_strides(
    sizes: list[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute] | int]]:
    ops = []
    strides: list[SSAValue[Attribute] | int] = []
    current_stride: SSAValue[Attribute] | int = 1

    for size in reversed(sizes):
        strides.insert(0, current_stride)

        if isinstance(current_stride, int) and isinstance(size, int):
            current_stride = current_stride * size
            continue

        if isinstance(current_stride, int):
            current_stride_op = arith.ConstantOp(IntegerAttr(current_stride, i64))
            ops.append(current_stride_op)
            current_stride_val = current_stride_op.result
        else:
            current_stride_val = current_stride

        if isinstance(size, int):
            size_op = arith.ConstantOp(IntegerAttr(size, i64))
            ops.append(size_op)
            size_val = size_op.result
        else:
            size_val = size

        mul_op = arith.MuliOp(operand1=current_stride_val, operand2=size_val)
        ops.append(mul_op)
        current_stride = mul_op.result

    return ops, strides


def compute_memref_offsets(
    indices: list[SSAValue[Attribute]],
    strides: list[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute] | int]]:
    ops = []
    offsets: list[SSAValue[Attribute]] = []

    for idx, stride in zip(indices, strides):
        if isinstance(stride, int):
            stride_op = arith.ConstantOp(IntegerAttr(stride, i64))
            ops.append(stride_op)
            stride_val = stride_op.result
        else:
            stride_val = stride

        mul_op = arith.MuliOp(operand1=idx, operand2=stride_val)
        ops.append(mul_op)
        offsets.append(mul_op.result)

    if not offsets:
        return ops, [arith.ConstantOp(IntegerAttr(0, i64)).result]

    accumulator = offsets[0]

    for offset in offsets[1:]:
        add_op = arith.AddiOp(operand1=accumulator, operand2=offset)
        ops.append(add_op)
        accumulator = add_op.result

    return ops, [accumulator]


class ConvertReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # compute strides and offsets
        stride_ops, strides = compute_memref_strides(
            get_dynamic_index_list(
                op.static_sizes.get_values(),
                op.sizes,
                memref.DYNAMIC_INDEX,
            )
        )
        offest_ops, offsets = compute_memref_offsets(op.indices, strides)

        # split static and dynamic offsets
        static_offsets, dynamic_offsets = split_dynamic_index_list(offsets, llvm.GEP_USE_SSA_VAL)

        rewriter.replace_matched_op(
            (
                *stride_ops,
                *offest_ops,
                cast_op := UnrealizedConversionCastOp.get([op.input], [llvm.LLVMPointerType()]),
                # get pointer and load
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    static_offsets,
                    op.result.type,
                    ssa_indices=dynamic_offsets,
                ),
                llvm.LoadOp(gep_op, op.result.type),
            )
        )


class ConvertAssignOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        # compute strides and offsets
        stride_ops, strides = compute_memref_strides(
            get_dynamic_index_list(
                op.static_sizes.get_values(),
                op.sizes,
                memref.DYNAMIC_INDEX,
            )
        )
        offset_ops, offsets = compute_memref_offsets(op.indices, strides)

        # split static and dynamic offsets
        static_offsets, dynamic_offsets = split_dynamic_index_list(offsets, llvm.GEP_USE_SSA_VAL)

        rewriter.replace_matched_op(
            (
                *stride_ops,
                *offset_ops,
                cast_op := UnrealizedConversionCastOp.get([op.input], [llvm.LLVMPointerType()]),
                # get pointer and store
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    static_offsets,
                    op.value.type,
                    ssa_indices=dynamic_offsets,
                ),
                llvm.StoreOp(op.value, gep_op),
            )
        )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        # compute strides and offsets
        stride_ops, strides = compute_memref_strides(
            get_dynamic_index_list(
                op.static_input_sizes.get_values(),
                op.input_sizes,
                memref.DYNAMIC_INDEX,
            )
        )
        offset_ops, offsets = compute_memref_offsets(op.indices, strides)

        # split static and dynamic offsets
        static_offsets, dynamic_offsets = split_dynamic_index_list(offsets, llvm.GEP_USE_SSA_VAL)

        rewriter.replace_matched_op(
            (
                *stride_ops,
                *offset_ops,
                cast_op := UnrealizedConversionCastOp.get([op.input], [llvm.LLVMPointerType()]),
                gep_op := llvm.GEPOp(
                    cast_op.results[0],
                    static_offsets,
                    op.result.type.element_type,
                    ssa_indices=dynamic_offsets,
                ),
                UnrealizedConversionCastOp.get([gep_op.result], [op.result.type]),
            )
        )


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AllocOp, rewriter: PatternRewriter):
        if op.mem.data != "DRAM":
            return

        # require static sized memref
        assert isinstance(op.result.type, MemRefType)
        assert all(size != -1 for size in op.result.type.get_shape())

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(IntegerAttr(reduce(lambda x, y: x * y, op.result.type.get_shape()), i64)),
                alloc_op := llvm.CallOp(
                    "malloc",
                    const_op.result,
                    return_type=llvm.LLVMPointerType(),
                ),
                UnrealizedConversionCastOp.get(alloc_op.returned, op.result.type),
            )
        )


class ConvertFreeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.DeallocOp, rewriter: PatternRewriter):
        if not isinstance(op.memref.type, MemRefType) or not isinstance(op.memref.type.memory_space, StringAttr):
            return
        if op.memref.type.memory_space.data != "DRAM":
            return

        rewriter.replace_matched_op(
            (
                cast_op := UnrealizedConversionCastOp.get([op.memref], [llvm.LLVMPointerType()]),
                llvm.CallOp("free", cast_op.results[0]),
            )
        )


@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True

    @attr_type_rewrite_pattern
    def convert_type(self, type: MemRefType):
        return llvm.LLVMPointerType()


class ConvertMemRefToLLVM(ModulePass):
    name = "convert-memref-to-llvm"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        builder = Builder(InsertPoint.at_end(m.body.block))
        builder.insert(
            llvm.FuncOp(
                "malloc",
                llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()),
                llvm.LinkageAttr("external"),
            )
        )
        builder.insert(
            llvm.FuncOp(
                "free",
                llvm.LLVMFunctionType([llvm.LLVMPointerType()]),
                llvm.LinkageAttr("external"),
            )
        )

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertAllocOp(),
                    ConvertFreeOp(),
                ]
            ),
        ).rewrite_module(m)

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertReadOp(),
                    ConvertAssignOp(),
                    ConvertWindowOp(),
                    RewriteMemRefTypes(),
                ]
            ),
        ).rewrite_module(m)
