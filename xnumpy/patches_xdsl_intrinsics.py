from collections.abc import Callable
from typing import ClassVar, TypeAlias

from xdsl.dialects import llvm, vector
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, VectorType, f32, f64, i64
from xdsl.dialects.llvm import FAbsOp, FNegOp, MaskedStoreOp
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern

from xnumpy.patches_xdsl_llvm import FSqrtOp, VectorFMaxOp

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

F32X4 = VectorType(f32, [4])
F64X2 = VectorType(f64, [2])


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
    # load all ptrs, apply op_fn to their values (or just return first value if op_fn is None)
    loads = [llvm.LoadOp(p, vec_type) for p in ptrs]
    vals = [ld.dereferenced_value for ld in loads]
    if op_fn is None:
        return list(loads), vals[0]
    result_op = op_fn(*vals)
    return [*loads, result_op], result_op.res


def _builder(op_fn: Callable[..., Operation] | None, *arg_indices: int) -> BuilderFn:
    # shorthand for creating a BuilderFn that calls _build_binop.
    # arg_indices select which call args to load: 0 = dst, 1 = first src, 2 = second src, ...
    #
    # e.g. _builder(llvm.FAddOp, 1, 2)  creates  (dst, a, b)  -> load a, load b, fadd
    #      _builder(llvm.FAddOp, 0, 1)  creates  (dst, src)   -> load dst, load src, fadd
    #      _builder(None, 1)            creates  (dst, src)   -> load src (copy)
    def builder(dst: SSAValue, *srcs: SSAValue, vec_type: VectorType) -> BuildResult:
        all_args = (dst, *srcs)
        return _build_binop(op_fn, *(all_args[i] for i in arg_indices), vec_type=vec_type)

    return builder


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


def _build_neon_square(dst: SSAValue, src: SSAValue, *, vec_type: VectorType) -> tuple[Operation, ...]:
    # dst[:] = src[:] * src[:]
    load = llvm.LoadOp(src, vec_type)
    result = llvm.FMulOp(load.dereferenced_value, load.dereferenced_value)
    return (load, result, llvm.StoreOp(result.res, dst))


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
    # callee name -> handler(args: list[SSAValue]) -> tuple[Operation, ...]
    entries: dict[str, Handler] = {}

    # vec_*: (name, builder, pfx_builder, uses_ext)
    ops: list[tuple[str, BuilderFn, BuilderFn | None, bool]] = [
        ("abs", _build_abs, _build_abs_pfx, True),
        ("add_red", _builder(llvm.FAddOp, 0, 1), None, True),  # dst += src
        ("copy", _builder(None, 1), None, True),  # dst = src
        ("load", _builder(None, 1), None, True),  # dst = src
        ("store", _builder(None, 1), None, False),  # dst = src
        ("add", _builder(llvm.FAddOp, 1, 2), None, False),  # dst = a + b
        ("mul", _builder(llvm.FMulOp, 1, 2), None, False),  # dst = a * b
        ("neg", _builder(FNegOp, 1), None, False),  # dst = -src
        ("brdcst_scl", _build_broadcast, None, False),
        ("fmadd2", _builder(vector.FMAOp, 1, 2, 3), None, False),  # dst = a * b + c
        ("fmadd1", _builder(vector.FMAOp, 1, 2, 3), None, False),  # dst = a * b + c
        ("fmadd_red", _builder(vector.FMAOp, 1, 2, 0), None, False),  # dst = a * b + dst
        ("zero", _build_zero, None, False),
    ]
    for name, builder, pfx_builder, uses_ext in ops:
        actual_pfx_builder = pfx_builder if pfx_builder is not None else builder
        chosen_f64_mask = _mask_f64x2_ext if uses_ext else _mask_f64x2
        entries[f"vec_{name}_f32x4"] = _plain_handler(builder, F32X4)
        entries[f"vec_{name}_f32x4_pfx"] = _pfx_handler(actual_pfx_builder, F32X4, _mask_f32x4)
        entries[f"vec_{name}_f64x2"] = _plain_handler(builder, F64X2)
        entries[f"vec_{name}_f64x2_pfx"] = _pfx_handler(actual_pfx_builder, F64X2, chosen_f64_mask)

    # vec_reduce_*
    for suffix, vt in [
        ("f32x4", F32X4),
        ("f64x2", F64X2),
    ]:
        entries[f"vec_reduce_add_scl_{suffix}"] = _reduce_handler(vt)

    # neon binops: dst = op(a, b)
    _NEON_BINOPS: list[tuple[str, type]] = [
        ("add", llvm.FAddOp),
        ("sub", llvm.FSubOp),
        ("mul", llvm.FMulOp),
        ("div", llvm.FDivOp),
        ("vadd", llvm.FAddOp),
        ("vsub", llvm.FSubOp),
        ("vmul", llvm.FMulOp),
    ]
    for op_name, op_cls in _NEON_BINOPS:
        entries[f"neon_{op_name}_f32x4"] = lambda args, o=op_cls: _build_neon_binop(o, *args, vec_type=F32X4)

    # neon acc binops: acc = op(acc, src)
    _NEON_ACC_OPS: list[tuple[str, type]] = [
        ("add_acc", llvm.FAddOp),
        ("fmax_acc", VectorFMaxOp),
        ("mul_acc", llvm.FMulOp),
        ("sub_acc", llvm.FSubOp),
        ("div_acc", llvm.FDivOp),
    ]
    for op_name, op_cls in _NEON_ACC_OPS:
        entries[f"neon_{op_name}_f32x4"] = lambda args, o=op_cls: _build_neon_binop(o, args[0], args[0], args[1], vec_type=F32X4)

    # neon unops: dst = op(src)
    _NEON_UNOPS: list[tuple[str, type]] = [
        ("neg", FNegOp),
        ("vneg", FNegOp),
        ("sqrt", FSqrtOp),
    ]
    for op_name, op_cls in _NEON_UNOPS:
        entries[f"neon_{op_name}_f32x4"] = lambda args, o=op_cls: _build_neon_unop(o, *args, vec_type=F32X4)

    # neon load/store/fmadd/broadcast (both types)
    for suffix, vt in [
        ("f32x4", F32X4),
        ("f64x2", F64X2),
    ]:
        entries[f"neon_storeu_{suffix}"] = lambda args, v=vt: _build_neon_storeu(*args, vec_type=v)
        entries[f"neon_loadu_{suffix}"] = lambda args, v=vt: _build_neon_storeu(*args, vec_type=v)
        entries[f"neon_fmadd_{suffix}"] = lambda args, v=vt: _build_neon_fmadd(*args, vec_type=v)
        entries[f"neon_broadcast_{suffix}"] = lambda args, v=vt: _build_neon_broadcast(*args, vec_type=v)

    # neon misc
    entries["neon_zero_f32x4"] = lambda args: _build_neon_zero(args[0], vec_type=F32X4)
    entries["neon_square_f32x4"] = lambda args: _build_neon_square(*args, vec_type=F32X4)

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
