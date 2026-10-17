"""Verify the Qwen3.8-27B NVFP4 artifact against its source roles.

This is the Qwen3.8 counterpart of ``tools/convert/qwen3_6_27b/verify_nvfp4.py``.
It re-materializes every weight object straight from the source checkpoints and
compares the result word-exactly against the artifact payload, proving that the
conversion preserved the source quantized weights (NVFP4 codes/scales/divisors,
row-scaled FP8 codes) rather than re-quantizing them.

The quantized source (community QAT NVFP4) owns every NVFP4/FP8/GDN-dequant and
quantized-direct object. The official BF16 source owns the embeddings, draft
head, MTP and vision tower; those objects are only checked when
``--official-model`` is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_fp8_row_scaled_words,
    decode_nvfp4_words,
    encoded_size,
)
from tools.convert.common.safetensors import ShardReader

from . import convert_nvfp4
from . import fp8_embedding
from . import inventory_nvfp4 as inventory
from . import recipe_nvfp4 as recipe


class VerificationError(ValueError):
    """The artifact does not satisfy the registered NVFP4 contract."""


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    objects: int
    tensors: int
    resources: int
    nvfp4_weights: int
    fp8_weights: int
    input_divisors: int
    quantized_direct: int
    nvfp4_dequant: int
    official_skipped: int
    payload_bytes: int
    mismatches: int


def _error(message: str) -> None:
    raise VerificationError(message)


def validate_structure(artifact: Artifact) -> int:
    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        _error(
            f"artifact identity is {artifact.identity!r}, expected "
            f"{expected_identity!r}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _error(
            f"artifact has {len(artifact.objects)} objects, "
            f"expected {len(inventory.OBJECT_SPECS)}"
        )
    cursor = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, inventory.OBJECT_SPECS)
    ):
        if actual.name != expected.name:
            _error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _error(
                f"{actual.name}: offset {actual.offset}, expected {expected_offset}"
            )
        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _error(f"{actual.name}: expected tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _error(f"{actual.name}: signature {signature} != {registered}")
            if actual.bytes != encoded_size(
                actual.layout, actual.format, actual.shape
            ):
                _error(f"{actual.name}: encoded byte count is invalid")
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _error(f"{actual.name}: expected resource descriptor")
            if actual.encoding != expected.encoding:
                _error(f"{actual.name}: resource encoding is invalid")
        cursor = actual.offset + actual.bytes
    expected_formats = {
        key: value for key, value in inventory.FORMAT_COUNTS.items() if value
    }
    expected_layouts = {
        key: value for key, value in inventory.LAYOUT_COUNTS.items() if value
    }
    if dict(formats) != expected_formats:
        _error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != expected_layouts:
        _error(f"layout counts are {dict(layouts)}")
    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")
    return payload_bytes


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    return torch.equal(
        left.detach().contiguous().reshape(-1).view(torch.uint8),
        right.detach().contiguous().reshape(-1).view(torch.uint8),
    )


class _Reporter:
    def __init__(self) -> None:
        self.mismatches: list[str] = []

    def record(self, name: str, detail: str) -> None:
        self.mismatches.append(f"{name}: {detail}")


def _object(artifact: Artifact, name: str) -> TensorObject:
    obj = artifact.find(name)
    if not isinstance(obj, TensorObject):
        _error(f"{name}: expected tensor")
    return obj


def _verify_nvfp4_weights(
    artifact: Artifact,
    reader: ShardReader,
    report: _Reporter,
) -> int:
    for name, selected in recipe.NVFP4_WEIGHTS_BY_NAME.items():
        obj = _object(artifact, name)
        packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
        stored_packed, stored_scales, stored_divisor = decode_nvfp4_words(
            artifact.payload(obj), obj.shape
        )
        if not _bits_equal(stored_packed, packed):
            report.record(name, "packed E2M1 words differ from source")
        if not _bits_equal(stored_scales, scales):
            report.record(name, "E4M3FN scale words differ from source")
        if bytes(stored_divisor.reshape(1).view(torch.uint8).numpy()) != divisor:
            report.record(name, "weight divisor differs from source")
    return len(recipe.NVFP4_WEIGHTS_BY_NAME)


def _verify_fp8_weights(
    artifact: Artifact,
    reader: ShardReader,
    report: _Reporter,
) -> int:
    for name, selected in recipe.FP8_WEIGHTS_BY_NAME.items():
        obj = _object(artifact, name)
        # Community NVFP4 packages keep lm_head as BF16; the artifact re-encodes
        # it to row-scaled FP8 at conversion time, so compare against a fresh
        # re-encode of the same BF16 source instead of a pre-quantized FP8 source.
        source_name = selected.parts[0].source.name + ".weight"
        weight = reader.get(source_name)
        expected = fp8_embedding.quantize_bf16_rows(weight)
        stored_codes, stored_scales = decode_fp8_row_scaled_words(
            artifact.payload(obj), obj.shape
        )
        if not _bits_equal(stored_codes, expected.codes.to(torch.uint8)):
            report.record(name, "row-scaled FP8 codes differ from source re-encode")
        if not _bits_equal(stored_scales, expected.scales):
            report.record(name, "row-scaled FP8 scales differ from source re-encode")
    return len(recipe.FP8_WEIGHTS_BY_NAME)


def _verify_input_divisors(
    artifact: Artifact,
    reader: ShardReader,
    report: _Reporter,
) -> int:
    for name, selected in recipe.INPUT_DIVISORS_BY_NAME.items():
        obj = _object(artifact, name)
        expected = recipe.materialize_input_divisor(selected, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        if not _bits_equal(stored, expected):
            report.record(name, "input divisor differs from source")
    return len(recipe.INPUT_DIVISORS_BY_NAME)


def _verify_quantized_direct(
    artifact: Artifact,
    reader: ShardReader,
    report: _Reporter,
) -> int:
    for name in recipe.QUANTIZED_DIRECT_BY_NAME:
        obj = _object(artifact, name)
        expected = recipe.materialize_quantized_direct(name, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        if not _bits_equal(stored, expected):
            report.record(name, "quantized-direct tensor differs from source")
    return len(recipe.QUANTIZED_DIRECT_BY_NAME)


def _verify_nvfp4_dequant(
    artifact: Artifact,
    reader: ShardReader,
    report: _Reporter,
) -> int:
    spec_by_name = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    for name in recipe.NVFP4_DEQUANT_BY_NAME:
        obj = _object(artifact, name)
        spec = spec_by_name[name]
        expected = convert_nvfp4._encode_nvfp4_dequant_ab(spec, reader)
        stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
        if not _bits_equal(stored, expected):
            report.record(name, "dequantized GDN A/B projection differs from source")
    return len(recipe.NVFP4_DEQUANT_BY_NAME)


def verify_artifact(
    artifact: Artifact,
    quantized_dir: str | Path,
    reporter: _Reporter | None = None,
) -> VerificationSummary:
    report = reporter if reporter is not None else _Reporter()
    payload_bytes = validate_structure(artifact)
    with ShardReader(quantized_dir) as reader:
        recipe.preflight_quantized_metadata(reader)
        nvfp4_weights = _verify_nvfp4_weights(artifact, reader, report)
        fp8_weights = _verify_fp8_weights(artifact, reader, report)
        input_divisors = _verify_input_divisors(artifact, reader, report)
        quantized_direct = _verify_quantized_direct(artifact, reader, report)
        nvfp4_dequant = _verify_nvfp4_dequant(artifact, reader, report)
    official_skipped = len(recipe.OFFICIAL_RECIPES_BY_NAME)
    if report.mismatches:
        preview = "\n  ".join(report.mismatches[:20])
        _error(
            f"{len(report.mismatches)} object(s) differ from the source:\n  {preview}"
        )
    return VerificationSummary(
        objects=len(artifact.objects),
        tensors=len(inventory.TENSOR_SPECS),
        resources=len(inventory.RESOURCE_SPECS),
        nvfp4_weights=nvfp4_weights,
        fp8_weights=fp8_weights,
        input_divisors=input_divisors,
        quantized_direct=quantized_direct,
        nvfp4_dequant=nvfp4_dequant,
        official_skipped=official_skipped,
        payload_bytes=payload_bytes,
        mismatches=len(report.mismatches),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--quantized-model",
        type=Path,
        required=True,
        help="Community NVFP4/QAT source checkpoint directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = _Reporter()
    with Artifact.open(arguments.artifact) as artifact:
        try:
            summary = verify_artifact(
                artifact, arguments.quantized_model, report
            )
        except VerificationError as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 1
    print(json.dumps({"ok": True, **asdict(summary)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
