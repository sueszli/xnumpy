from dataclasses import dataclass
from functools import reduce

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, llvm, memref
from xdsl.dialects.builtin import IntegerAttr, MemRefType, ModuleOp, StringAttr, UnrealizedConversionCastOp, i64
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


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
                alloc_op := llvm.CallOp("malloc", const_op.result, return_type=llvm.LLVMPointerType()),
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
        builder.insert(llvm.FuncOp("malloc", llvm.LLVMFunctionType([i64], llvm.LLVMPointerType()), llvm.LinkageAttr("external")))
        builder.insert(llvm.FuncOp("free", llvm.LLVMFunctionType([llvm.LLVMPointerType()]), llvm.LinkageAttr("external")))

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
