from typing import Sequence

from xdsl.context import Context
from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import IndexType, IntegerAttr, MemRefType, ModuleOp, NoneAttr, StridedLayoutAttr, i64
from xdsl.dialects.utils import get_dynamic_index_list, split_dynamic_index_list
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from xdsl_exo.dialects import exo

"""
alternative memref based implementation

- ConvertReadOp — exo.ReadOp → memref.LoadOp
- ConvertAssignOp — exo.AssignOp → memref.StoreOp
- ConvertWindowOp — exo.WindowOp → memref.SubviewOp (with stride/offset computation)
"""


class ConvertReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.ReadOp, rewriter: PatternRewriter):
        # convert tensor reads only
        if len(op.indices) < 1:
            return

        ops = [arith.IndexCastOp(idx, IndexType()) for idx in op.indices]
        idx = [op.result for op in ops]

        rewriter.replace_matched_op(
            (
                *ops,
                memref.LoadOp.get(op.input, idx),
            )
        )


class ConvertAssignOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.AssignOp, rewriter: PatternRewriter):
        # convert tensor writes only
        if len(op.indices) < 1:
            return

        assert isinstance(op.input.type, MemRefType)

        ops = [arith.IndexCastOp(idx, IndexType()) for idx in op.indices]
        idx = [op.result for op in ops]

        # if the value is a scalar memref, we need to load
        if isinstance(op.value.type, MemRefType):
            assert op.value.type.get_shape() == (1,), f"expected scalar memref type, got {op.value.type}"

            return rewriter.replace_matched_op(
                (
                    *ops,
                    zero_op := arith.ConstantOp(IntegerAttr(0, IndexType())),
                    load_op := memref.LoadOp.get(op.value, [zero_op.result]),
                    memref.StoreOp.get(load_op.res, op.input, idx),
                )
            )

        rewriter.replace_matched_op((*ops, memref.StoreOp.get(op.value, op.input, idx)))


def compute_memref_strides(
    sizes: list[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute] | int]]:
    ops = []
    strides: list[SSAValue[Attribute] | int] = [1]

    for size in reversed(sizes):
        last_stride = strides[0]

        # if both static, we can compute statically
        if isinstance(last_stride, int) and isinstance(size, int):
            strides.insert(0, last_stride * size)
            continue

        # wrap static values in constant ops
        if isinstance(last_stride, int):
            last_stride = arith.ConstantOp(IntegerAttr(last_stride, i64)).result
            ops.append(last_stride.op)
        if isinstance(size, int):
            size = arith.ConstantOp(IntegerAttr(size, i64)).result
            ops.append(size.op)

        # multiply
        mul_op = arith.MuliOp(operand1=last_stride, operand2=size)
        ops.append(mul_op)
        strides.insert(0, mul_op.result)

    return ops, strides


def compute_memref_offsets(
    indices: list[SSAValue[Attribute]],
    strides: list[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute] | int]]:
    ops = []
    offsets: list[SSAValue[Attribute]] = []

    for idx, stride in zip(indices, strides):
        if isinstance(stride, int):
            stride = arith.ConstantOp(IntegerAttr(stride, i64)).result
            ops.append(stride.op)

        mul_op = arith.MuliOp(operand1=idx, operand2=stride)
        ops.append(mul_op)
        offsets.append(mul_op.result)

    return ops, offsets


def compute_memref_sizes(
    sizes: list[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute] | int]]:
    ops = []
    sizes: list[SSAValue[Attribute] | int] = []

    for size in sizes:
        if isinstance(size, int):
            size = arith.ConstantOp(IntegerAttr(size, i64)).result
            ops.append(size.op)

        # multiply
        mul_op = arith.MuliOp(operand1=size, operand2=size)
        ops.append(mul_op)
        sizes.append(mul_op.result)

    return ops, sizes


def convert_all_to_index(
    input: Sequence[SSAValue[Attribute] | int],
) -> tuple[list[Operation], list[SSAValue[Attribute]]]:
    ops = []
    output_dyn_values = []

    static_values, dyn_values = split_dynamic_index_list(input, memref.DYNAMIC_INDEX)

    for value in dyn_values:
        ops.append(cast_op := arith.IndexCastOp(value, IndexType()))
        output_dyn_values.append(cast_op.result)

    return (
        ops,
        get_dynamic_index_list(
            static_values,
            output_dyn_values,
            memref.DYNAMIC_INDEX,
        ),
    )


class ConvertWindowOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: exo.WindowOp, rewriter: PatternRewriter):
        assert isinstance(op.input.type, MemRefType)
        assert isinstance(op.result.type, MemRefType)

        input_dims = op.input.type.get_num_dims()
        output_dims = op.result.type.get_num_dims()

        stride_ops, strides = compute_memref_strides(
            get_dynamic_index_list(
                op.static_input_sizes.get_values(),
                op.input_sizes,
                memref.DYNAMIC_INDEX,
            )
        )
        offset_ops, offsets = compute_memref_offsets(op.indices, strides)

        # convert all to index
        stride_idx_ops, strides = convert_all_to_index(strides)
        offset_idx_ops, offsets = convert_all_to_index(offsets)
        size_idx_ops, sizes = convert_all_to_index(
            get_dynamic_index_list(
                op.static_output_sizes.get_values(),
                op.output_sizes,
                memref.DYNAMIC_INDEX,
            )
        )

        rewriter.replace_matched_op(
            (
                *stride_ops,
                *offset_ops,
                *stride_idx_ops,
                *offset_idx_ops,
                *size_idx_ops,
                memref.SubviewOp.get(
                    op.input,
                    offsets,
                    sizes,
                    strides,
                    MemRefType(
                        op.result.type.element_type,
                        op.result.type.shape,
                        StridedLayoutAttr(
                            split_dynamic_index_list(strides, -1)[0][input_dims - output_dims + 1 :],
                            NoneAttr(),
                        ),
                        op.result.type.memory_space,
                    ),
                ),
            )
        )


class ConvertTensorRefPass(ModulePass):
    name = "convert-tensor-ref"

    def apply(self, ctx: Context, m: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertReadOp(),
                    ConvertAssignOp(),
                    ConvertWindowOp(),
                ]
            )
        ).rewrite_module(m)
