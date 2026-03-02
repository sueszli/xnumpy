from dataclasses import dataclass
from functools import reduce

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, memref, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, Float32Type, IndexType, IntegerAttr, MemRefType, ModuleOp, StringAttr, UnrealizedConversionCastOp, VectorType, f32, f64, i32, i64
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, TypeConversionPattern, attr_type_rewrite_pattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from xdsl_exo import patches as llvm_extra

VT_F32x8 = VectorType(f32, [8])
VT_F64x4 = VectorType(f64, [4])


#
# mask generation for prefix (partial width) variants
#


def _mask_f32x8(m):
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [8]), list(range(8))))
    broadcast = vector.BroadcastOp(operands=[m], result_types=[VectorType(i64, [8])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, broadcast, mask], mask.res


def _mask_f64x4_ext(m):
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [4]), list(range(4))))
    ext = arith.ExtSIOp(m, i64)
    broadcast = vector.BroadcastOp(operands=[ext.result], result_types=[VectorType(i64, [4])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, ext, broadcast, mask], mask.res


def _mask_f64x4(m):
    indices = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [4]), list(range(4))))
    broadcast = vector.BroadcastOp(operands=[m], result_types=[VectorType(i32, [4])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return [indices, broadcast, mask], mask.res


#
# core operation builders
#
# each takes (op, vt, pfx) and returns (ops_list, result_value, dst_ptr).
# for pfx variants, argument indices shift by 1 (arg[0] is mask threshold m).
#


def _build_identity(op, vt, pfx):
    off = 1 if pfx else 0
    load = llvm.LoadOp(op.arguments[off + 1], vt)
    return [load], load.dereferenced_value, op.arguments[off]


def _build_abs(op, vt, pfx):
    off = 1 if pfx else 0
    dst, src = op.arguments[off], op.arguments[off + 1]
    load = llvm.LoadOp(src, vt)
    fabs = llvm_extra.FAbsOp(load.dereferenced_value, vt)
    if pfx:
        return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result, dst
    return [load, fabs], fabs.result, dst


def _build_neg(op, vt, pfx):
    off = 1 if pfx else 0
    load = llvm.LoadOp(op.arguments[off + 1], vt)
    neg = llvm_extra.FNegOp(load.dereferenced_value)
    return [load, neg], neg.res, op.arguments[off]


def _build_binary(binop_cls):
    def build(op, vt, pfx):
        off = 1 if pfx else 0
        l1 = llvm.LoadOp(op.arguments[off + 1], vt)
        l2 = llvm.LoadOp(op.arguments[off + 2], vt)
        result = binop_cls(l1.dereferenced_value, l2.dereferenced_value)
        return [l1, l2, result], result.res, op.arguments[off]

    return build


_build_add = _build_binary(llvm.FAddOp)
_build_mul = _build_binary(llvm.FMulOp)


def _build_add_red(op, vt, pfx):
    off = 1 if pfx else 0
    dst = op.arguments[off]
    l_dst = llvm.LoadOp(dst, vt)
    l_src = llvm.LoadOp(op.arguments[off + 1], vt)
    add = llvm.FAddOp(l_dst.dereferenced_value, l_src.dereferenced_value)
    return [l_dst, l_src, add], add.res, dst


def _build_fma(op, vt, pfx):
    off = 1 if pfx else 0
    l1 = llvm.LoadOp(op.arguments[off + 1], vt)
    l2 = llvm.LoadOp(op.arguments[off + 2], vt)
    l3 = llvm.LoadOp(op.arguments[off + 3], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, l3.dereferenced_value], result_types=[vt])
    return [l1, l2, l3, fma], fma.res, op.arguments[off]


def _build_fma_red(op, vt, pfx):
    off = 1 if pfx else 0
    dst = op.arguments[off]
    ld = llvm.LoadOp(dst, vt)
    l1 = llvm.LoadOp(op.arguments[off + 1], vt)
    l2 = llvm.LoadOp(op.arguments[off + 2], vt)
    fma = vector.FMAOp(operands=[l1.dereferenced_value, l2.dereferenced_value, ld.dereferenced_value], result_types=[vt])
    return [ld, l1, l2, fma], fma.res, dst


def _build_broadcast(op, vt, pfx):
    off = 1 if pfx else 0
    bcast = vector.BroadcastOp(operands=[op.arguments[off + 1]], result_types=[vt])
    return [bcast], bcast.vector, op.arguments[off]


def _build_zero(op, vt, pfx):
    off = 1 if pfx else 0
    zero = arith.ConstantOp(DenseIntOrFPElementsAttr.create_dense_float(vt, [0.0] * vt.get_shape()[0]))
    return [zero], zero.result, op.arguments[off]


#
# intrinsic dispatch table
#
# maps callee name -> (builder_fn, vector_type, mask_fn_or_none).
# f64x4 pfx: abs/add_red/copy/load use _mask_f64x4_ext, all others use _mask_f64x4.
#

_VEC_INTRINSICS: dict = {}
for _name, _builder, _f64_mask in [
    ("abs", _build_abs, _mask_f64x4_ext),
    ("add_red", _build_add_red, _mask_f64x4_ext),
    ("copy", _build_identity, _mask_f64x4_ext),
    ("load", _build_identity, _mask_f64x4_ext),
    ("store", _build_identity, _mask_f64x4),
    ("add", _build_add, _mask_f64x4),
    ("mul", _build_mul, _mask_f64x4),
    ("neg", _build_neg, _mask_f64x4),
    ("brdcst_scl", _build_broadcast, _mask_f64x4),
    ("fmadd2", _build_fma, _mask_f64x4),
    ("fmadd1", _build_fma, _mask_f64x4),
    ("fmadd_red", _build_fma_red, _mask_f64x4),
    ("zero", _build_zero, _mask_f64x4),
]:
    _VEC_INTRINSICS[f"vec_{_name}_f32x8"] = (_builder, VT_F32x8, None)
    _VEC_INTRINSICS[f"vec_{_name}_f32x8_pfx"] = (_builder, VT_F32x8, _mask_f32x8)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4"] = (_builder, VT_F64x4, None)
    _VEC_INTRINSICS[f"vec_{_name}_f64x4_pfx"] = (_builder, VT_F64x4, _f64_mask)


#
# blas rewrite patterns
#


class ConvertSelect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "select":
            return
        rewriter.replace_matched_op(
            (
                cmp_op := arith.CmpfOp(op.arguments[0], op.arguments[1], "olt"),
                arith.SelectOp(cmp_op.results[0], op.arguments[2], op.arguments[3]),
            )
        )


class ConvertVecIntrinsic(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        callee = op.callee.root_reference.data

        if callee == "vec_reduce_add_scl_f32x8":
            return self._reduce(op, rewriter, VT_F32x8)
        if callee == "vec_reduce_add_scl_f64x4":
            return self._reduce(op, rewriter, VT_F64x4)

        entry = _VEC_INTRINSICS.get(callee)
        if entry is None:
            return

        builder, vt, mask_fn = entry
        pfx = mask_fn is not None
        core_ops, result, dst = builder(op, vt, pfx)

        if pfx:
            mask_ops, mask = mask_fn(op.arguments[0])
            rewriter.replace_matched_op((*mask_ops, *core_ops, llvm_extra.MaskedStoreOp(result, dst, mask)))
        else:
            rewriter.replace_matched_op((*core_ops, llvm.StoreOp(result, dst)))

    @staticmethod
    def _reduce(op, rewriter, vt):
        assert isinstance(op.arguments[0].owner, llvm.LoadOp)
        acc_load = op.arguments[0].owner
        load = llvm.LoadOp(op.arguments[1], vt)
        reduce = vector.ReductionOp(load.dereferenced_value, vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]), acc=op.arguments[0])
        rewriter.replace_matched_op((load, reduce, llvm.StoreOp(reduce.dest, acc_load.ptr)))


class ConvertMM256StoreuPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_storeu_ps":
            return
        rewriter.replace_matched_op(
            (
                load_op := llvm.LoadOp(op.arguments[1], VectorType(Float32Type(), [8])),
                llvm.StoreOp(load_op.result, op.arguments[0]),
            )
        )


class ConvertMM256FmaddPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_fmadd_ps":
            return
        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load0_op := llvm.LoadOp(op.arguments[0], VectorType(Float32Type(), [8])),
                load1_op := llvm.LoadOp(op.arguments[1], VectorType(Float32Type(), [8])),
                load2_op := llvm.LoadOp(op.arguments[2], VectorType(Float32Type(), [8])),
                fma_op := vector.FMAOp(operands=[load1_op.dereferenced_value, load2_op.dereferenced_value, load0_op.dereferenced_value], result_types=[VectorType(Float32Type(), [8])]),
                llvm.StoreOp(fma_op.res, op.arguments[0], [zero_op.result]),
            )
        )


class ConvertMM256BroadcastSsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_broadcast_ss":
            return
        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                scalar_load_op := memref.LoadOp.get(op.arguments[1], [zero_op.result]),
                broadcast_op := vector.BroadcastOp(operands=[scalar_load_op.results[0]], result_types=[VectorType(Float32Type(), [8])]),
                llvm.StoreOp(broadcast_op.results[0], op.arguments[0], [zero_op.result]),
            )
        )


class ConvertMM256LoaduPsOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "mm256_loadu_ps":
            return
        rewriter.replace_matched_op(
            (
                zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                load_op := llvm.LoadOp(op.arguments[1], VectorType(Float32Type(), [8])),
                llvm.StoreOp(load_op.result, op.arguments[1], [zero_op.result]),
            )
        )


#
# memref to llvm rewrite patterns
#


class EraseVecDeallocOp(RewritePattern):
    # erases memref.deallocop for vec_avx2 memory space (stack-allocated, no free needed).

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


#
# passes
#


class ConvertExternPass(ModulePass):
    name = "convert-extern"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(GreedyRewritePatternApplier([ConvertSelect()])).rewrite_module(m)


class ConvertIntrinsicsPass(ModulePass):
    name = "convert-intrinsics"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSelect(),
                    ConvertVecIntrinsic(),
                    ConvertMM256StoreuPsOp(),
                    ConvertMM256FmaddPsOp(),
                    ConvertMM256BroadcastSsOp(),
                    ConvertMM256LoaduPsOp(),
                ]
            )
        ).rewrite_module(m)


class ConvertAllocFreeToLLVM(ModulePass):
    # converts memref.allocop to malloc and memref.deallocop to free.

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
    # converts remaining memreftype to llvmpointertype.

    name = "lower-memref-types"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteMemRefTypes(),
                ]
            ),
        ).rewrite_module(m)
