from __future__ import annotations

from exo.core.memory import MemGenError, Memory


class Stack(Memory):
    # stack-allocated memory (uses alloca instead of malloc)
    # use for small temporaries that don't need heap allocation

    @classmethod
    def alloc(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        c_types = {"float": "float", "double": "double", "int8_t": "int8_t", "int32_t": "int32_t"}
        c_type = c_types.get(prim_type, prim_type)
        if not shape:
            return f"{c_type} {new_name};"
        return f'{c_type} {new_name}[{"][".join(shape)}];'

    @classmethod
    def can_read(cls) -> bool:
        return True

    @classmethod
    def free(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        return ""


class NEON(Memory):
    # arm neon 128-bit vector memory (4×f32, 2×f64)

    @classmethod
    def global_(cls) -> str:
        return "#include <arm_neon.h>"

    @classmethod
    def alloc(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        if not shape:
            raise MemGenError(f"{srcinfo}: NEON vectors are not scalar values")

        vec_types: dict[str, tuple[int, str]] = {"float": (4, "float32x4_t"), "double": (2, "float64x2_t")}

        if prim_type not in vec_types:
            raise MemGenError(f"{srcinfo}: NEON vectors must be f32/f64, got {prim_type}")

        reg_width, c_type = vec_types[prim_type]
        if not (shape[-1].isdecimal() and int(shape[-1]) == reg_width):
            raise MemGenError(f"{srcinfo}: NEON vectors of type {prim_type} must be {reg_width}-wide, got {shape}")
        remaining = shape[:-1]
        if remaining:
            result = f'{c_type} {new_name}[{"][".join(map(str, remaining))}];'
        else:
            result = f"{c_type} {new_name};"
        return result

    @classmethod
    def can_read(cls) -> bool:
        return False

    @classmethod
    def free(cls, new_name: str, prim_type: str, shape: tuple[str, ...], srcinfo: object) -> str:
        return ""
