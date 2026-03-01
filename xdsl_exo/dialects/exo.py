from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects import memref
from xdsl.dialects.builtin import DenseArrayBase, MemRefType, i64
from xdsl.dialects.utils import split_dynamic_index_list
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import AnyAttr, Attribute, AttrSizedOperandSegments, IRDLOperation, ParsePropInAttrDict, VarConstraint, irdl_op_definition, operand_def, prop_def, result_def, var_operand_def
from xdsl.printer import Printer


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "exo.assign"

    value = operand_def()
    input = operand_def()
    indices = var_operand_def(i64)
    sizes = var_operand_def(i64)
    static_sizes = prop_def(DenseArrayBase)

    assembly_format = "$value `,` $input `[` $indices `]` `,` `sizes` `:` `[` $sizes `]` `,` attr-dict `:` type($value) `,` type($input)"

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    def __init__(
        self,
        value: SSAValue | Operation,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sizes: Sequence[SSAValue | int],
    ) -> None:
        static_sizes, dyn_sizes = split_dynamic_index_list(sizes, memref.DYNAMIC_INDEX)
        super().__init__(
            operands=[value, SSAValue.get(input), indices, dyn_sizes],
            result_types=[],
            properties={"static_sizes": DenseArrayBase.from_list(i64, static_sizes)},
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    name = "exo.read"

    input = operand_def()
    indices = var_operand_def(i64)
    sizes = var_operand_def(i64)
    static_sizes = prop_def(DenseArrayBase)
    result = result_def()

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        sizes: Sequence[SSAValue | int],
        result_type: Attribute,
    ) -> None:
        static_sizes, dyn_sizes = split_dynamic_index_list(sizes, memref.DYNAMIC_INDEX)
        super().__init__(
            operands=[
                SSAValue.get(input),
                indices,
                dyn_sizes,
            ],
            result_types=[result_type],
            properties={"static_sizes": DenseArrayBase.from_list(i64, static_sizes)},
        )

    def verify_(self):
        if isinstance(self.input.type, MemRefType) and isinstance(self.result.type, MemRefType):
            assert self.input.type == self.result.type, f"input type {self.input.type} does not match result type {self.result.type}"

        if isinstance(self.input.type, MemRefType) and not isinstance(self.result.type, MemRefType):
            assert self.input.type.element_type == self.result.type, f"input element type {self.input.type.element_type} does not match result type {self.result.type}"
        else:
            assert len(self.indices) == 0, f"expected no indices for non-memref input, but got {self.indices}"

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.input)
        if len(self.indices) > 0:
            printer.print_string("[")
            for i, index in enumerate(self.indices):
                if i > 0:
                    printer.print_string(", ")
                printer.print_ssa_value(index)
            printer.print_string("]")
        printer.print_string(" -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class WindowOp(IRDLOperation):
    T: ClassVar = VarConstraint("T", AnyAttr())

    name = "exo.window"

    input = operand_def(
        MemRefType.constr(element_type=AnyAttr()),
    )
    indices = var_operand_def()
    input_sizes = var_operand_def()
    static_input_sizes = prop_def(DenseArrayBase)
    output_sizes = var_operand_def()
    static_output_sizes = prop_def(DenseArrayBase)

    result = result_def(
        MemRefType.constr(element_type=T),
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True), ParsePropInAttrDict())

    def __init__(
        self,
        input: SSAValue | Operation,
        indices: Sequence[SSAValue | Operation],
        input_sizes: Sequence[SSAValue | Operation | int],
        output_sizes: Sequence[SSAValue[Attribute] | int],
        result_type: MemRefType,
    ) -> None:
        static_input_sizes, dyn_input_sizes = split_dynamic_index_list(input_sizes, memref.DYNAMIC_INDEX)
        static_output_sizes, dyn_output_sizes = split_dynamic_index_list(output_sizes, memref.DYNAMIC_INDEX)

        super().__init__(
            operands=[SSAValue.get(input), indices, dyn_input_sizes, dyn_output_sizes],
            result_types=[result_type],
            properties={
                "static_input_sizes": DenseArrayBase.create_dense_int(i64, static_input_sizes),
                "static_output_sizes": DenseArrayBase.create_dense_int(i64, static_output_sizes),
            },
        )


Exo = Dialect(
    "exo",
    [
        AssignOp,
        ReadOp,
        WindowOp,
    ],
    [],
)
