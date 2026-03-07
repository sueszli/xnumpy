from collections.abc import Callable
from typing import ClassVar, TypeAlias

from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, VectorType, f32, f64, i64
from xdsl.dialects.llvm import FAbsOp, FNegOp, MaskedStoreOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

# `vec_*` intrinsic lowering: `llvm.CallOp` -> LLVM/vector dialect ops
#
# Naming:
# -------
#     vec_<op>_<type>          - plain version. all lanes written
#     vec_<op>_<type>_pfx      - prefix version. only lanes 0..n-1 written (loop tails)
#
#     <type> = f32x8 | f64x4
#     <op>  = add, mul, neg, abs, fmadd1, fmadd2, fmadd_red, zero, ...
#
# Plain variant:
# --------------
#     llvm.call @vec_add_f32x8(%dst, %a, %b)
#     =>
#     %v0  = llvm.load %a : vector<8xf32>
#     %v1  = llvm.load %b : vector<8xf32>
#     %r   = llvm.fadd %v0, %v1
#            llvm.store %r, %dst
#
# Prefix (_pfx) variant:
# ----------------------
# First arg is a lane-count `n`. A boolean mask selects which lanes get written.
#
#     llvm.call @vec_add_f32x8_pfx(%n, %dst, %a, %b)      e.g. n=3
#     =>
#     %idx  = arith.constant   [0, 1, 2, 3, 4, 5, 6, 7]
#     %bc   = vector.broadcast [3, 3, 3, 3, 3, 3, 3, 3]   (n splatted to all lanes)
#     %mask = llvm.icmp "slt"  [T, T, T, F, F, F, F, F]   (idx < bc)
#     %v0   = llvm.load %a : vector<8xf32>
#     %v1   = llvm.load %b : vector<8xf32>
#     %r    = llvm.fadd %v0, %v1
#             llvm.masked_store %r, %dst, %mask


MaskResult: TypeAlias = tuple[list[Operation], SSAValue]
BuildResult: TypeAlias = tuple[list[Operation], SSAValue]
Builder: TypeAlias = Callable[..., BuildResult]
MaskFn: TypeAlias = Callable[[SSAValue], MaskResult]
Handler: TypeAlias = Callable[[list[SSAValue]], tuple[Operation, ...]]


def _make_mask(lane_count: SSAValue, n_lanes: int, *, extend_lane_count: bool = False) -> MaskResult:
    # mask[i] = (i < lane_count), e.g. lane_count=3 -> [T, T, T, F, F, ...]
    ops = []
    indices = llvm.ConstantOp(DenseIntOrFPElementsAttr.from_list(VectorType(i64, [n_lanes]), list(range(n_lanes))), VectorType(i64, [n_lanes]))
    ops.append(indices)
    if extend_lane_count:
        ext = llvm.SExtOp(lane_count, i64)  # i32 -> i64 to match VectorType(i64, ...)
        ops.append(ext)
        lane_count = ext.res
    broadcast = vector.BroadcastOp(operands=[lane_count], result_types=[VectorType(i64, [n_lanes])])
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return ops + [broadcast, mask], mask.res


def _mask_f32x8(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 8)  # AVX2: 256-bit / 32-bit = 8 lanes


def _mask_f64x4(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 4)  # AVX2: 256-bit / 64-bit = 4 lanes


def _mask_f64x4_ext(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 4, extend_lane_count=True)  # lane_count is i32; upcast to i64


def _build_copy(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src[:]
    load = llvm.LoadOp(src, vec_type)
    return [load], load.dereferenced_value


def _build_abs(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = abs(src[:])
    load = llvm.LoadOp(src, vec_type)
    fabs = FAbsOp(load.dereferenced_value, vec_type)
    return [load, fabs], fabs.result


def _build_abs_pfx(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # step 1 (here):   dst[:] = src[:]        -- write src to all lanes
    # step 2 (caller): dst[:n] = abs(src[:n]) -- MaskedStoreOp overwrites active lanes
    # net:             dst[:n] = abs(src[:n]), dst[n:] = src[n:]
    load = llvm.LoadOp(src, vec_type)
    fabs = FAbsOp(load.dereferenced_value, vec_type)
    return [load, fabs, llvm.StoreOp(load.dereferenced_value, dst)], fabs.result


def _build_neg(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = -src[:]
    load = llvm.LoadOp(src, vec_type)
    neg = FNegOp(load.dereferenced_value)
    return [load, neg], neg.res


def _build_add(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] + src_b[:]
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    result = llvm.FAddOp(load_a.dereferenced_value, load_b.dereferenced_value)
    return [load_a, load_b, result], result.res


def _build_mul(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] * src_b[:]
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    result = llvm.FMulOp(load_a.dereferenced_value, load_b.dereferenced_value)
    return [load_a, load_b, result], result.res


def _build_add_red(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = dst[:] + src[:]
    load_dst = llvm.LoadOp(dst, vec_type)
    load_src = llvm.LoadOp(src, vec_type)
    add = llvm.FAddOp(load_dst.dereferenced_value, load_src.dereferenced_value)
    return [load_dst, load_src, add], add.res


def _build_fma(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, src_c: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] * src_b[:] + src_c[:]
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    load_c = llvm.LoadOp(src_c, vec_type)
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_c.dereferenced_value], result_types=[vec_type])
    return [load_a, load_b, load_c, fma], fma.res


def _build_fma_red(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    load_acc = llvm.LoadOp(dst, vec_type)
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value], result_types=[vec_type])
    return [load_acc, load_a, load_b, fma], fma.res


def _build_broadcast(dst: SSAValue, scalar: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [scalar] * n_lanes
    broadcast = vector.BroadcastOp(operands=[scalar], result_types=[vec_type])
    return [broadcast], broadcast.vector


def _build_zero(dst: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [0.0] * n_lanes
    zero = llvm.ConstantOp(DenseIntOrFPElementsAttr.from_list(vec_type, [0.0] * vec_type.get_shape()[0]), vec_type)
    return [zero], zero.result


def _plain_handler(builder: Builder, vec_type: VectorType) -> Handler:
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        dst, *srcs = args
        ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*ops, llvm.StoreOp(result, dst))

    return handle


def _pfx_handler(builder: Builder, vec_type: VectorType, mask_fn: MaskFn) -> Handler:
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        lane_count, dst, *srcs = args
        mask_ops, mask = mask_fn(lane_count)
        core_ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*mask_ops, *core_ops, MaskedStoreOp(result, dst, mask))

    return handle


def _reduce_handler(vec_type: VectorType) -> Handler:
    # acc_scalar += sum(src_vector); acc_val must come from llvm.LoadOp to recover the store pointer.
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        acc_val, src_ptr = args[0], args[1]
        assert isinstance(acc_val.owner, llvm.LoadOp)
        src_load = llvm.LoadOp(src_ptr, vec_type)
        reduce = vector.ReductionOp(src_load.dereferenced_value, vector.CombiningKindAttr([vector.CombiningKindFlag.ADD]), acc=acc_val)
        return (src_load, reduce, llvm.StoreOp(reduce.dest, acc_val.owner.ptr))

    return handle


def _build_mm256_storeu_ps(dst: SSAValue, src: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = src[:]
    load = llvm.LoadOp(src, VectorType(f32, [8]))
    return (load, llvm.StoreOp(load.dereferenced_value, dst))


def _build_mm256_fmadd_ps(dst: SSAValue, src_a: SSAValue, src_b: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    load_acc = llvm.LoadOp(dst, VectorType(f32, [8]))
    load_a = llvm.LoadOp(src_a, VectorType(f32, [8]))
    load_b = llvm.LoadOp(src_b, VectorType(f32, [8]))
    fma = vector.FMAOp(operands=[load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value], result_types=[VectorType(f32, [8])])
    return (load_acc, load_a, load_b, fma, llvm.StoreOp(fma.res, dst))


def _build_mm256_broadcast_ss(dst: SSAValue, scalar_ptr: SSAValue) -> tuple[Operation, ...]:
    # dst[:] = [*scalar_ptr] * 8  (scalar_ptr is already !llvm.ptr at this stage of the pipeline)
    load = llvm.LoadOp(scalar_ptr, f32)
    broadcast = vector.BroadcastOp(operands=[load.dereferenced_value], result_types=[VectorType(f32, [8])])
    return (load, broadcast, llvm.StoreOp(broadcast.vector, dst))


def _make_intrinsics() -> dict[str, Handler]:
    entries: dict[str, Handler] = {}
    for name, builder, pfx_builder, f64_mask in [
        ("abs", _build_abs, _build_abs_pfx, _mask_f64x4_ext),
        ("add_red", _build_add_red, None, _mask_f64x4_ext),
        ("copy", _build_copy, None, _mask_f64x4_ext),
        ("load", _build_copy, None, _mask_f64x4_ext),
        ("store", _build_copy, None, _mask_f64x4),
        ("add", _build_add, None, _mask_f64x4),
        ("mul", _build_mul, None, _mask_f64x4),
        ("neg", _build_neg, None, _mask_f64x4),
        ("brdcst_scl", _build_broadcast, None, _mask_f64x4),
        ("fmadd2", _build_fma, None, _mask_f64x4),
        ("fmadd1", _build_fma, None, _mask_f64x4),
        ("fmadd_red", _build_fma_red, None, _mask_f64x4),
        ("zero", _build_zero, None, _mask_f64x4),
    ]:
        actual_pfx_builder = pfx_builder if pfx_builder is not None else builder
        for suffix, vec_type, mask_fn in [
            ("f32x8", VectorType(f32, [8]), _mask_f32x8),
            ("f64x4", VectorType(f64, [4]), f64_mask),
        ]:
            entries[f"vec_{name}_{suffix}"] = _plain_handler(builder, vec_type)
            entries[f"vec_{name}_{suffix}_pfx"] = _pfx_handler(actual_pfx_builder, vec_type, mask_fn)

    entries["vec_reduce_add_scl_f32x8"] = _reduce_handler(VectorType(f32, [8]))
    entries["vec_reduce_add_scl_f64x4"] = _reduce_handler(VectorType(f64, [4]))

    entries["mm256_storeu_ps"] = lambda args: _build_mm256_storeu_ps(*args)
    entries["mm256_loadu_ps"] = lambda args: _build_mm256_storeu_ps(*args)  # same lowering in Exo's calling convention
    entries["mm256_fmadd_ps"] = lambda args: _build_mm256_fmadd_ps(*args)
    entries["mm256_broadcast_ss"] = lambda args: _build_mm256_broadcast_ss(*args)

    return entries


class ConvertVecIntrinsic(RewritePattern):
    _INTRINSICS: ClassVar[dict[str, Handler]] = _make_intrinsics()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.CallOp, rewriter: PatternRewriter) -> None:
        if op.callee is None:
            return
        handler = self._INTRINSICS.get(op.callee.root_reference.data)
        if handler is None:
            return
        rewriter.replace_matched_op(handler(list(op.args)))
