from __future__ import annotations

import ctypes
from math import prod
from typing import Any

import numpy as np


class Ptr:
    __slots__ = ("data",)

    def __init__(self, data: int):
        self.data = data


def normalize_dtype(dtype: Any) -> type[float] | type[int]:
    base = getattr(dtype, "type", dtype)
    if base in (float, np.float64):
        return float
    if base in (int, np.int64):
        return int
    raise TypeError(dtype)


def ctype(dtype: Any) -> type[ctypes.c_double] | type[ctypes.c_int64]:
    base = normalize_dtype(dtype)
    if base is float:
        return ctypes.c_double
    if base is int:
        return ctypes.c_int64
    raise TypeError(dtype)


class Storage:
    __slots__ = ("dtype", "ctype", "buffer", "numel", "itemsize", "ptr")

    def __init__(self, numel: int, dtype: Any = float, *, fill: float | int | None = None) -> None:
        self.dtype = normalize_dtype(dtype)
        self.ctype = ctype(self.dtype)
        self.numel = numel
        self.itemsize = ctypes.sizeof(self.ctype)
        self.buffer = (self.ctype * numel)()
        self.ptr = ctypes.addressof(self.buffer)
        if fill not in (None, 0, 0.0):
            for i in range(numel):
                self.buffer[i] = fill


class Tensor:
    __slots__ = ("shape", "dtype", "storage", "offset", "numel", "itemsize", "ptr")

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any = float,
        *,
        storage: Storage | None = None,
        offset: int = 0,
        fill: float | int | None = None,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = normalize_dtype(dtype)
        self.numel = prod(self.shape)
        self.storage = storage if storage is not None else Storage(self.numel, self.dtype, fill=fill)
        if self.storage.dtype is not self.dtype:
            raise TypeError((self.storage.dtype, self.dtype))
        if offset < 0 or offset + self.numel > self.storage.numel:
            raise ValueError((offset, self.numel, self.storage.numel))
        self.offset = offset
        self.itemsize = self.storage.itemsize
        self.ptr = self.storage.ptr + offset * self.itemsize

    @property
    def ctypes(self) -> Ptr:
        return Ptr(self.ptr)

    def view(self, shape: tuple[int, ...], *, offset: int = 0) -> Tensor:
        return Tensor(shape, dtype=self.dtype, storage=self.storage, offset=self.offset + offset)

    def flat_index(self, key: int | tuple[int, ...]) -> int:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != len(self.shape):
            raise IndexError(key)
        if len(key) == 1:
            return self.offset + key[0]
        if len(key) == 2:
            return self.offset + key[0] * self.shape[1] + key[1]
        if len(key) == 3:
            return self.offset + (key[0] * self.shape[1] + key[1]) * self.shape[2] + key[2]
        raise ValueError(self.shape)

    def __getitem__(self, key: int | tuple[int, ...]) -> float | int:
        return self.storage.buffer[self.flat_index(key)]

    def __setitem__(self, key: int | tuple[int, ...], value: float | int) -> None:
        self.storage.buffer[self.flat_index(key)] = value


def empty(shape: tuple[int, ...], dtype: Any = float) -> Tensor:
    return Tensor(shape, dtype=dtype)


def full(shape: tuple[int, ...], fill_value: float | int, dtype: Any = float) -> Tensor:
    return Tensor(shape, dtype=dtype, fill=fill_value)


def zeros(shape: tuple[int, ...], dtype: Any = float) -> Tensor:
    return full(shape, 0.0 if normalize_dtype(dtype) is float else 0, dtype=dtype)
