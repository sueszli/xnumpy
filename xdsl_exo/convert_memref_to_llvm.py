from dataclasses import dataclass
from functools import reduce

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, StringAttr, UnrealizedConversionCastOp, i64
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


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


class EraseVecDeallocOp(RewritePattern):
    """Erases memref.DeallocOp for VEC_AVX2 memory space (stack-allocated, no free needed)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.DeallocOp, rewriter: PatternRewriter):
        if not isinstance(op.memref.type, MemRefType) or not isinstance(op.memref.type.memory_space, StringAttr):
            return
        if op.memref.type.memory_space.data != "VEC_AVX2":
            return

        rewriter.erase_op(op)


class ConvertAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.AllocOp, rewriter: PatternRewriter):
        memref_type = op.memref.type
        if not isinstance(memref_type.memory_space, StringAttr):
            return
        if memref_type.memory_space.data != "DRAM":
            return

        assert all(size != -1 for size in memref_type.get_shape())

        rewriter.replace_matched_op(
            (
                const_op := arith.ConstantOp(IntegerAttr(reduce(lambda x, y: x * y, memref_type.get_shape()), i64)),
                alloc_op := llvm.CallOp(
                    "malloc",
                    const_op.result,
                    return_type=llvm.LLVMPointerType(),
                ),
                UnrealizedConversionCastOp.get(alloc_op.returned, memref_type),
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


class ConvertAllocFreeToLLVM(ModulePass):
    """Converts memref.AllocOp to malloc and memref.DeallocOp to free."""

    name = "convert-alloc-free-to-llvm"

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
                    EraseVecDeallocOp(),
                    ConvertAllocOp(),
                    ConvertFreeOp(),
                ]
            ),
        ).rewrite_module(m)


class LowerMemRefTypesPass(ModulePass):
    """Converts remaining MemRefType to LLVMPointerType."""

    name = "lower-memref-types"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteMemRefTypes(),
                ]
            ),
        ).rewrite_module(m)
