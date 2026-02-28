# index dialect ops missing from upstream xdsl

from xdsl.dialects.builtin import AnySignlessIntegerOrIndexType, IndexType, IntegerType
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.irdl import Attribute, IRDLOperation, irdl_op_definition, operand_def, result_def


@irdl_op_definition
class CastsOp(IRDLOperation):
    name = "index.casts"

    input = operand_def(AnySignlessIntegerOrIndexType)
    result = result_def(AnySignlessIntegerOrIndexType)

    assembly_format = "$input attr-dict `:` type($input) `to` type($result)"

    def __init__(self, input: SSAValue | Operation, result_type: Attribute) -> None:
        super().__init__(
            operands=[input],
            result_types=[result_type],
        )

    def verify_(self):
        if isinstance(self.input.type, IndexType):
            assert isinstance(self.result.type, IntegerType) and not isinstance(self.result.type, IndexType), "result type must be integer for index type input"

        elif isinstance(self.input.type, IntegerType):
            assert isinstance(self.result.type, IndexType), "result type must be index type for integer type input"


Index = Dialect(
    "index",
    [CastsOp],
    [],
)
