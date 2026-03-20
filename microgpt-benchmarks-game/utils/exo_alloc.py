from __future__ import annotations

import ctypes
import random
from collections.abc import Sequence
from math import prod
from typing import Any


class Ptr:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def ctype(dtype: type[float] | type[int]) -> type[ctypes.c_double] | type[ctypes.c_int32]:
    # python dtype -> ctypes scalar
    if dtype is float:
        return ctypes.c_double
    if dtype is int:
        return ctypes.c_int32
    raise TypeError(dtype)


class Tensor:
    __slots__ = ("shape", "dtype", "_ctype", "_buf", "_offset", "_size")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: type[float] | type[int] = float,
        *,
        buffer: Any = None,
        offset: int = 0,
        fill: float | int | None = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ctype = ctype(dtype)
        self._offset = offset
        self._size = prod(self.shape)
        self._buf = (self._ctype * self._size)() if buffer is None else buffer
        if buffer is None and fill not in (None, 0, 0.0):
            for i in range(self._size):
                self._buf[i] = fill

    @property
    def ctypes(self) -> Ptr:
        return Ptr(ctypes.addressof(self._buf) + self._offset * ctypes.sizeof(self._ctype))

    def flat_index(self, key: int | tuple[int, ...]) -> int:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != len(self.shape):
            raise IndexError(key)
        if len(key) == 1:
            return self._offset + key[0]
        if len(key) == 2:
            return self._offset + key[0] * self.shape[1] + key[1]
        if len(key) == 3:
            return self._offset + (key[0] * self.shape[1] + key[1]) * self.shape[2] + key[2]
        raise ValueError(self.shape)

    def __getitem__(self, key: int | tuple[int, ...]) -> float | int:
        return self._buf[self.flat_index(key)]

    def __setitem__(self, key: int | tuple[int, ...], value: float | int) -> None:
        self._buf[self.flat_index(key)] = value


def array(data: Sequence[float | int] | Sequence[Sequence[float | int]], dtype: type[float] | type[int] = float) -> Tensor:
    # python sequence -> tensor
    if not data:
        raise TypeError("array expects a non-empty Python sequence")
    if isinstance(data[0], Sequence):
        out = empty((len(data), len(data[0])), dtype=dtype)
        for i, row in enumerate(data):
            if len(row) != out.shape[1]:
                raise ValueError("ragged nested sequences are unsupported")
            for j, value in enumerate(row):
                out[i, j] = value
        return out
    out = empty((len(data),), dtype=dtype)
    for i, value in enumerate(data):
        out[i] = value
    return out


def empty(shape: tuple[int, ...], dtype: type[float] | type[int] = float) -> Tensor:
    # allocate uninitialized storage
    return Tensor(shape, dtype=dtype)


def empty_like(x: Tensor) -> Tensor:
    # allocate with x.shape and x.dtype
    return empty(x.shape, dtype=x.dtype)


def full(shape: tuple[int, ...], fill_value: float | int, dtype: type[float] | type[int] = float) -> Tensor:
    # allocate and fill with a scalar
    return Tensor(shape, dtype=dtype, fill=fill_value)


def zeros(shape: tuple[int, ...], dtype: type[float] | type[int] = float) -> Tensor:
    # allocate and fill with 0
    return full(shape, 0.0 if dtype is float else 0, dtype=dtype)


def zeros_like(x: Tensor) -> Tensor:
    # allocate zeros with x.shape and x.dtype
    return zeros(x.shape, dtype=x.dtype)


def reshape(x: Tensor, shape: tuple[int, ...], *, offset: int = 0) -> Tensor:
    # shared-buffer reshape view
    return Tensor(shape, dtype=x.dtype, buffer=x._buf, offset=x._offset + offset)


def normal(shape: tuple[int, ...], loc: float = 0.0, scale: float = 1.0) -> Tensor:
    # iid gaussian fill: n(loc, scale^2)
    out = empty(shape, dtype=float)
    for i in range(out._size):
        out._buf[i] = random.gauss(loc, scale)
    return out


def pack_tensors(tensors: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor], int]:
    # flatten tensors into one shared storage block and return typed views
    total = sum(t._size for t in tensors.values())
    flat = empty((total,), dtype=float)
    flat_ptr = flat.ctypes.data
    elt_bytes = ctypes.sizeof(flat._ctype)
    offset = 0
    views = {}
    for name, tensor in tensors.items():
        ctypes.memmove(flat_ptr + offset * elt_bytes, tensor.ctypes.data, tensor._size * elt_bytes)
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor._size
    return flat, views, elt_bytes


def view_tensors(flat: Tensor, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    # rebuild tensor views over existing flat storage
    offset = 0
    views = {}
    for name, tensor in tensors.items():
        views[name] = reshape(flat, tensor.shape, offset=offset)
        offset += tensor._size
    return views


def tensor_ptrs(tensors: dict[str, Tensor]) -> dict[str, int]:
    # expose raw data pointers for raw-jit entry points
    return {name: tensor.ctypes.data for name, tensor in tensors.items()}
