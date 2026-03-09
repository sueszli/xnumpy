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
#     <type> = f32x4 | f64x2 (NEON 128-bit)
#     <op>  = add, mul, neg, abs, fmadd1, fmadd2, fmadd_red, zero, ...
#
# Plain variant:
# --------------
#     llvm.call @vec_add_f32x4(%dst, %a, %b)
#     =>
#     %v0  = llvm.load %a : vector<4xf32>
#     %v1  = llvm.load %b : vector<4xf32>
#     %r   = llvm.fadd %v0, %v1
#            llvm.store %r, %dst
#
# Prefix (_pfx) variant:
# ----------------------
# First arg is a lane-count `n`. A boolean mask selects which lanes get written.
#
#     llvm.call @vec_add_f32x4_pfx(%n, %dst, %a, %b)      e.g. n=3
#     =>
#     %idx  = arith.constant   [0, 1, 2, 3]
#     %bc   = vector.broadcast [3, 3, 3, 3]   (n splatted to all lanes)
#     %mask = llvm.icmp "slt"  [T, T, T, F]   (idx < bc)
#     %v0   = llvm.load %a : vector<4xf32>
#     %v1   = llvm.load %b : vector<4xf32>
#     %r    = llvm.fadd %v0, %v1
#             llvm.masked_store %r, %dst, %mask


MaskResult: TypeAlias = tuple[list[Operation], SSAValue]
BuildResult: TypeAlias = tuple[list[Operation], SSAValue]
BuilderFn: TypeAlias = Callable[..., BuildResult]
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
    broadcast = vector.BroadcastOp(lane_count, VectorType(i64, [n_lanes]))
    mask = llvm.ICmpOp(indices.result, broadcast.vector, IntegerAttr(llvm.ICmpPredicateFlag.SLT.to_int(), i64))
    return ops + [broadcast, mask], mask.res


def _mask_f32x4(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 4)  # NEON: 128-bit / 32-bit = 4 lanes


def _mask_f64x2(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 2)  # NEON: 128-bit / 64-bit = 2 lanes


def _mask_f64x2_ext(lane_count: SSAValue) -> MaskResult:
    return _make_mask(lane_count, 2, extend_lane_count=True)  # lane_count is i32; upcast to i64


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


def _build_binop(op_fn: Callable[..., Operation] | None, *ptrs: SSAValue, vec_type: VectorType) -> BuildResult:
    loads = [llvm.LoadOp(p, vec_type) for p in ptrs]
    vals = [ld.dereferenced_value for ld in loads]
    if op_fn is None:
        return list(loads), vals[0]
    result_op = op_fn(*vals)
    return [*loads, result_op], result_op.res


def _build_copy(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src[:]
    return _build_binop(None, src, vec_type=vec_type)


def _build_neg(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = -src[:]
    return _build_binop(FNegOp, src, vec_type=vec_type)


def _build_add(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] + src_b[:]
    return _build_binop(llvm.FAddOp, src_a, src_b, vec_type=vec_type)


def _build_mul(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] * src_b[:]
    return _build_binop(llvm.FMulOp, src_a, src_b, vec_type=vec_type)


def _build_add_red(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = dst[:] + src[:]
    return _build_binop(llvm.FAddOp, dst, src, vec_type=vec_type)


def _build_fma(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, src_c: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = src_a[:] * src_b[:] + src_c[:]
    return _build_binop(vector.FMAOp, src_a, src_b, src_c, vec_type=vec_type)


def _build_fma_red(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    return _build_binop(vector.FMAOp, src_a, src_b, dst, vec_type=vec_type)


def _build_broadcast(dst: SSAValue, scalar: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [scalar] * n_lanes
    broadcast = vector.BroadcastOp(scalar, vec_type)
    return [broadcast], broadcast.vector


def _build_zero(dst: SSAValue, *, vec_type: VectorType) -> BuildResult:
    # dst[:] = [0.0] * n_lanes
    zero = llvm.ConstantOp(DenseIntOrFPElementsAttr.from_list(vec_type, [0.0] * vec_type.get_shape()[0]), vec_type)
    return [zero], zero.result


def _plain_handler(builder: BuilderFn, vec_type: VectorType) -> Handler:
    # build ops then store result to dst (all lanes written)
    def handle(args: list[SSAValue]) -> tuple[Operation, ...]:
        dst, *srcs = args
        ops, result = builder(dst, *srcs, vec_type=vec_type)
        return (*ops, llvm.StoreOp(result, dst))

    return handle


def _pfx_handler(builder: BuilderFn, vec_type: VectorType, mask_fn: MaskFn) -> Handler:
    # build ops then masked-store result to dst (only lanes 0..n-1 written)
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


def _build_neon_storeu(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = src[:]
    load = llvm.LoadOp(src, vec_type)
    return (load, llvm.StoreOp(load.dereferenced_value, dst))


def _build_neon_fmadd(dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = dst[:] + src_a[:] * src_b[:]
    load_acc = llvm.LoadOp(dst, vec_type)
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    fma = vector.FMAOp(load_a.dereferenced_value, load_b.dereferenced_value, load_acc.dereferenced_value)
    return (load_acc, load_a, load_b, fma, llvm.StoreOp(fma.res, dst))


def _build_neon_broadcast(dst: SSAValue, scalar_ptr: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = [*scalar_ptr] * n_lanes  (scalar_ptr is already !llvm.ptr at this stage of the pipeline)
    elem_type = vec_type.element_type
    load = llvm.LoadOp(scalar_ptr, elem_type)
    broadcast = vector.BroadcastOp(load.dereferenced_value, vec_type)
    return (load, broadcast, llvm.StoreOp(broadcast.vector, dst))


def _build_neon_binop(op_cls: type, dst: SSAValue, src_a: SSAValue, src_b: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = op(src_a[:], src_b[:])
    load_a = llvm.LoadOp(src_a, vec_type)
    load_b = llvm.LoadOp(src_b, vec_type)
    result = op_cls(load_a.dereferenced_value, load_b.dereferenced_value)
    return (load_a, load_b, result, llvm.StoreOp(result.res, dst))


def _build_neon_unop(op_cls: type, dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = op(src[:])
    load = llvm.LoadOp(src, vec_type)
    result = op_cls(load.dereferenced_value)
    return (load, result, llvm.StoreOp(result.res, dst))


def _build_neon_zero(dst: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = [0.0] * n_lanes  (scalar zero + broadcast to avoid vector ConstantOp)
    from xdsl.dialects.builtin import FloatAttr

    elem_type = vec_type.element_type
    zero = llvm.ConstantOp(FloatAttr(0.0, elem_type), elem_type)
    broadcast = vector.BroadcastOp(zero.result, vec_type)
    return (zero, broadcast, llvm.StoreOp(broadcast.vector, dst))


def _make_intrinsics() -> dict[str, Handler]:
    entries: dict[str, Handler] = {}

    _OPS: list[tuple[str, BuilderFn, BuilderFn | None, bool]] = [
        ("abs", _build_abs, _build_abs_pfx, True),
        ("add_red", _build_add_red, None, True),
        ("copy", _build_copy, None, True),
        ("load", _build_copy, None, True),
        ("store", _build_copy, None, False),
        ("add", _build_add, None, False),
        ("mul", _build_mul, None, False),
        ("neg", _build_neg, None, False),
        ("brdcst_scl", _build_broadcast, None, False),
        ("fmadd2", _build_fma, None, False),
        ("fmadd1", _build_fma, None, False),
        ("fmadd_red", _build_fma_red, None, False),
        ("zero", _build_zero, None, False),
    ]

    _F32X4 = VectorType(f32, [4])
    _F64X2 = VectorType(f64, [2])

    for name, builder, pfx_builder, uses_ext in _OPS:
        actual_pfx_builder = pfx_builder if pfx_builder is not None else builder
        chosen_f64_mask = _mask_f64x2_ext if uses_ext else _mask_f64x2

        entries[f"vec_{name}_f32x4"] = _plain_handler(builder, _F32X4)
        entries[f"vec_{name}_f32x4_pfx"] = _pfx_handler(actual_pfx_builder, _F32X4, _mask_f32x4)
        entries[f"vec_{name}_f64x2"] = _plain_handler(builder, _F64X2)
        entries[f"vec_{name}_f64x2_pfx"] = _pfx_handler(actual_pfx_builder, _F64X2, chosen_f64_mask)

    entries["vec_reduce_add_scl_f32x4"] = _reduce_handler(_F32X4)
    entries["vec_reduce_add_scl_f64x2"] = _reduce_handler(_F64X2)

    entries["neon_storeu_f32x4"] = lambda args: _build_neon_storeu(*args, vec_type=_F32X4)
    entries["neon_loadu_f32x4"] = lambda args: _build_neon_storeu(*args, vec_type=_F32X4)
    entries["neon_fmadd_f32x4"] = lambda args: _build_neon_fmadd(*args, vec_type=_F32X4)
    entries["neon_broadcast_f32x4"] = lambda args: _build_neon_broadcast(*args, vec_type=_F32X4)
    entries["neon_zero_f32x4"] = lambda args: _build_neon_zero(args[0], vec_type=_F32X4)
    entries["neon_add_f32x4"] = lambda args: _build_neon_binop(llvm.FAddOp, *args, vec_type=_F32X4)
    entries["neon_mul_f32x4"] = lambda args: _build_neon_binop(llvm.FMulOp, *args, vec_type=_F32X4)
    entries["neon_sub_f32x4"] = lambda args: _build_neon_binop(llvm.FSubOp, *args, vec_type=_F32X4)
    entries["neon_neg_f32x4"] = lambda args: _build_neon_unop(FNegOp, *args, vec_type=_F32X4)
    entries["neon_vadd_f32x4"] = lambda args: _build_neon_binop(llvm.FAddOp, *args, vec_type=_F32X4)
    entries["neon_vsub_f32x4"] = lambda args: _build_neon_binop(llvm.FSubOp, *args, vec_type=_F32X4)
    entries["neon_vmul_f32x4"] = lambda args: _build_neon_binop(llvm.FMulOp, *args, vec_type=_F32X4)
    entries["neon_vneg_f32x4"] = lambda args: _build_neon_unop(FNegOp, *args, vec_type=_F32X4)
    entries["neon_storeu_f64x2"] = lambda args: _build_neon_storeu(*args, vec_type=_F64X2)
    entries["neon_loadu_f64x2"] = lambda args: _build_neon_storeu(*args, vec_type=_F64X2)
    entries["neon_fmadd_f64x2"] = lambda args: _build_neon_fmadd(*args, vec_type=_F64X2)
    entries["neon_broadcast_f64x2"] = lambda args: _build_neon_broadcast(*args, vec_type=_F64X2)

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
