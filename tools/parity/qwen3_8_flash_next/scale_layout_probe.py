"""Scale-block layout probes for the quantized Flash-Next weights.

The artifact stores the W8 projections (``q8_g32_fp16`` / ``row_split_k128_v1``) and the routed
expert bank (``nvfp4`` / ``batched_block_scale_k16_m128x4_v1``) with one scale per code group.
A scale that is paired with the wrong group leaves every tensor plausible and every number
wrong, so this probe checks the pairing directly, without any external checkpoint.

The statistic is the correlation between the logarithm of the stored scale and the logarithm of
the magnitude that the paired code group actually carries.  A correctly paired scale tracks the
group magnitude closely; a mispaired scale does not track it any better than a random
permutation of the same scales, which is the control reported next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.artifact.layouts import align_up
from tools.reference.qwen3_8_flash_next.bindings import V3Artifact
from tools.reference.qwen3_8_flash_next.ops import e2m1_lut, e4m3_lut
from tools.reference.qwen3_8_flash_next.weights import _scale_index


def correlation(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(a * b) / denominator)


def w8_probe(artifact, object_id, rows_to_check=64):
    obj = artifact.object(object_id)
    rows, columns = obj.shape[0], obj.shape[1]
    padded = align_up(columns, 128)
    groups = padded // 32
    base_row_bytes = groups * 32
    base_bytes = rows * base_row_bytes
    scale_offset = align_up(base_bytes, 256)
    raw = artifact.read_object(object_id)
    codes = np.frombuffer(raw[:base_bytes], dtype=np.int8).reshape(rows, base_row_bytes)[:, :columns]
    scales = np.frombuffer(
        raw[scale_offset : scale_offset + rows * groups * 2], dtype="<f2"
    ).reshape(rows, groups)
    row_ids = np.arange(min(rows, rows_to_check))
    code_block = codes[row_ids].reshape(row_ids.size, groups, 32).astype(np.float64)
    scale_block = scales[row_ids].astype(np.float64)
    magnitude = np.max(np.abs(code_block), axis=2)
    positive = magnitude > 0
    log_scale = np.log(scale_block[positive])
    log_magnitude = np.log(magnitude[positive])
    rng = np.random.default_rng(20261117)
    shuffled = rng.permutation(scale_block[positive].size)
    control = correlation(log_scale, log_magnitude[shuffled])
    ratio = scale_block[positive] / np.maximum(magnitude[positive], 1.0)
    return {
        "object": object_id,
        "shape": [rows, columns],
        "groups_per_row": int(groups),
        "scale_magnitude_correlation": correlation(log_scale, log_magnitude),
        "shuffled_control_correlation": control,
        "scale_over_max_code_median": float(np.median(ratio)),
        "scale_over_max_code_iqr": [
            float(np.percentile(ratio, 25)),
            float(np.percentile(ratio, 75)),
        ],
        "rows_checked": int(row_ids.size),
    }


def nvfp4_probe(artifact, object_id, matrix, rows_to_check=64):
    obj = artifact.object(object_id)
    matrices, rows, columns = obj.shape
    matrix_code_bytes = rows * columns // 2
    scale_offset = align_up(matrices * matrix_code_bytes, 256)
    matrix_scale_bytes = rows * columns // 16
    divisor_offset = scale_offset + matrices * matrix_scale_bytes
    raw_codes = artifact.read_object(
        object_id, matrix * matrix_code_bytes, matrix_code_bytes
    )
    codes = np.frombuffer(raw_codes, dtype=np.uint8).reshape(rows, columns // 2)
    values = np.empty((rows, columns), dtype=np.uint8)
    values[:, 0::2] = codes & 0x0F
    values[:, 1::2] = codes >> 4
    stored = np.frombuffer(
        artifact.read_object(
            object_id, scale_offset + matrix * matrix_scale_bytes, matrix_scale_bytes
        ),
        dtype=np.uint8,
    )
    groups = columns // 16
    scales = stored[_scale_index(rows, groups, columns)]
    decoded = e2m1_lut()[values]
    scale_values = e4m3_lut()[scales]
    divisor = float(
        np.frombuffer(
            artifact.read_object(object_id, divisor_offset + matrix * 4, 4), dtype="<f4"
        )[0]
    )
    row_ids = np.arange(min(rows, rows_to_check))
    code_block = decoded[row_ids].reshape(row_ids.size, groups, 16)
    scale_block = scale_values[row_ids].reshape(row_ids.size, groups)
    magnitude = np.sqrt(np.mean(code_block * code_block, axis=2))
    positive = magnitude > 0
    log_scale = np.log(scale_block[positive])
    log_magnitude = np.log(magnitude[positive])
    rng = np.random.default_rng(20261117)
    shuffled = rng.permutation(scale_block[positive].size)
    control = correlation(log_scale, log_magnitude[shuffled])
    ratio = scale_block[positive] / magnitude[positive] / divisor
    return {
        "object": object_id,
        "matrix": int(matrix),
        "shape": [rows, columns],
        "groups_per_row": int(groups),
        "scale_magnitude_correlation": correlation(log_scale, log_magnitude),
        "shuffled_control_correlation": control,
        "scale_over_group_rms_median": float(np.median(ratio)),
        "scale_over_group_rms_iqr": [
            float(np.percentile(ratio, 25)),
            float(np.percentile(ratio, 75)),
        ],
        "rows_checked": int(row_ids.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = {}
    with V3Artifact(args.artifact) as artifact:
        report["w8"] = [
            w8_probe(artifact, artifact.single_object("text/layers/0/gdn/query")),
            w8_probe(artifact, artifact.single_object("text/layers/0/gdn/output")),
            w8_probe(artifact, artifact.single_object("text/layers/3/attention/query")),
            w8_probe(artifact, artifact.single_object("text/layers/3/attention/output")),
        ]
        report["nvfp4"] = [
            nvfp4_probe(
                artifact, artifact.parts("text/layers/0/moe/experts/gate")[0].object, 0
            ),
            nvfp4_probe(
                artifact, artifact.parts("text/layers/0/moe/experts/down")[0].object, 0
            ),
        ]
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
