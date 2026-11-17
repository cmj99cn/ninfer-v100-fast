from __future__ import annotations

import collections
import os
import struct

import numpy as np

from tools.artifact.layouts import align_up

from .ops import bf16_from_u16, e2m1_lut, e4m3_lut, ple_decode_row

_NVFP4_SCALE_INVERSE = os.environ.get("NINFER_DIAG_NVFP4_SCALE_INVERSE") == "1"
_NVFP4_DIVISOR_INVERSE = os.environ.get("NINFER_DIAG_NVFP4_DIVISOR_INVERSE") == "1"
_W8_SCALE_INVERSE = os.environ.get("NINFER_DIAG_W8_SCALE_INVERSE") == "1"


def decode_contiguous(raw, fmt, shape):
    if fmt == "bf16":
        words = np.frombuffer(raw, dtype="<u2")
        values = bf16_from_u16(words)
    elif fmt == "fp32":
        values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif fmt == "i32":
        values = np.frombuffer(raw, dtype="<i4")
    else:
        raise ValueError("unsupported contiguous format " + fmt)
    return values.reshape(shape).copy()


def decode_w8(raw, shape):
    rows, columns = shape
    padded = align_up(columns, 128)
    groups = padded // 32
    base_row_bytes = groups * 32
    base_bytes = rows * base_row_bytes
    scale_offset = align_up(base_bytes, 256)
    codes = np.frombuffer(raw[:base_bytes], dtype=np.int8).reshape(rows, base_row_bytes)
    codes = codes[:, :columns].astype(np.float64)
    scales = np.frombuffer(
        raw[scale_offset : scale_offset + rows * groups * 2], dtype="<f2"
    ).reshape(rows, groups)
    scales = np.repeat(scales.astype(np.float64), 32, axis=1)[:, :columns]
    if _W8_SCALE_INVERSE:
        return (codes / np.where(scales == 0, 1.0, scales)).astype(np.float32)
    return (codes * scales).astype(np.float32)


def _scale_index(rows, groups, columns):
    row = np.arange(rows, dtype=np.int64)[:, None]
    group = np.arange(groups, dtype=np.int64)[None, :]
    return (
        (row // 128) * (columns // 64) + group // 4
    ) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + (group % 4)


def decode_batched_nvfp4(artifact, object_id, shape, matrix):
    matrices, rows, columns = shape
    matrix_code_bytes = rows * columns // 2
    code_bytes = matrices * matrix_code_bytes
    scale_offset = align_up(code_bytes, 256)
    matrix_scale_bytes = rows * columns // 16
    divisor_offset = scale_offset + matrices * matrix_scale_bytes
    codes = np.frombuffer(
        artifact.read_object(object_id, matrix * matrix_code_bytes, matrix_code_bytes),
        dtype=np.uint8,
    ).reshape(rows, columns // 2)
    values = np.empty((rows, columns), dtype=np.uint8)
    values[:, 0::2] = codes & 0x0F
    values[:, 1::2] = codes >> 4
    stored_scales = np.frombuffer(
        artifact.read_object(
            object_id, scale_offset + matrix * matrix_scale_bytes, matrix_scale_bytes
        ),
        dtype=np.uint8,
    )
    groups = columns // 16
    scales = stored_scales[_scale_index(rows, groups, columns)]
    codes_f = e2m1_lut()[values]
    scales_f = e4m3_lut()[scales]
    divisor = struct.unpack(
        "<f",
        artifact.read_object(object_id, divisor_offset + matrix * 4, 4),
    )[0]
    scale_rep = np.repeat(scales_f, 16, axis=1)
    if _NVFP4_SCALE_INVERSE:
        decoded = codes_f / np.where(scale_rep == 0, 1.0, scale_rep)
    else:
        decoded = codes_f * scale_rep
    if _NVFP4_DIVISOR_INVERSE:
        decoded = decoded * float(divisor)
    else:
        decoded = decoded / float(divisor)
    return decoded.astype(np.float32)


class WeightStore:
    def __init__(self, artifact):
        self.artifact = artifact
        self._contiguous = {}
        self._w8 = {}
        self._divisors = {}
        self._experts = collections.OrderedDict()
        self._expert_capacity = 24

    def contiguous(self, object_id):
        cached = self._contiguous.get(object_id)
        if cached is None:
            obj = self.artifact.object(object_id)
            cached = decode_contiguous(self.artifact.object_bytes(object_id), obj.format, obj.shape)
            self._contiguous[object_id] = cached
        return cached

    def w8(self, object_id):
        cached = self._w8.get(object_id)
        if cached is None:
            obj = self.artifact.object(object_id)
            cached = decode_w8(self.artifact.object_bytes(object_id), obj.shape)
            self._w8[object_id] = cached
        return cached

    def contiguous_rows(self, object_id, begin, count):
        return self.contiguous(object_id)[begin : begin + count]

    def nvfp4(self, object_id, matrix):
        key = (object_id, int(matrix))
        cached = self._experts.get(key)
        if cached is not None:
            self._experts.move_to_end(key)
            return cached
        obj = self.artifact.object(object_id)
        decoded = decode_batched_nvfp4(self.artifact, object_id, obj.shape, matrix)
        self._experts[key] = decoded
        if len(self._experts) > self._expert_capacity:
            self._experts.popitem(last=False)
        return decoded

    def ple_divisor(self, object_id):
        cached = self._divisors.get(object_id)
        if cached is None:
            obj = self.artifact.object(object_id)
            cached = struct.unpack("<f", self.artifact.read_object(object_id, obj.bytes - 4, 4))[0]
            self._divisors[object_id] = cached
        return cached

    def ple_row(self, shard_object, shard_row):
        row_bytes = 90
        rows_per_page = 4096 // row_bytes
        offset = shard_row // rows_per_page * 4096 + shard_row % rows_per_page * row_bytes
        raw = self.artifact.read_object(shard_object, offset, row_bytes)
        return ple_decode_row(raw, self.ple_divisor(shard_object))
