"""Diagnostic: NVFP4 scale/divisor decode-semantics statistics (read-only probe).

For the same (matrix,row,group) this reports decoded-weight distributions under
four arithmetic interpretations of the stored E4M3 scale word and FP32 divisor,
plus the scale<->code-RMS correlation that motivated the audit.
"""

from __future__ import annotations

import argparse
import struct
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


def corr(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return 0.0 if den == 0.0 else float(np.sum(a * b) / den)


def load_matrix(artifact, object_id, matrix, rows_to_check):
    obj = artifact.object(object_id)
    matrices, rows, columns = obj.shape
    code_bytes_per = rows * columns // 2
    scale_offset = align_up(matrices * code_bytes_per, 256)
    scale_bytes_per = rows * columns // 16
    divisor_offset = scale_offset + matrices * scale_bytes_per

    codes = np.frombuffer(
        artifact.read_object(object_id, matrix * code_bytes_per, code_bytes_per), dtype=np.uint8
    ).reshape(rows, columns // 2)
    values = np.empty((rows, columns), dtype=np.uint8)
    values[:, 0::2] = codes & 0x0F
    values[:, 1::2] = codes >> 4
    stored = np.frombuffer(
        artifact.read_object(object_id, scale_offset + matrix * scale_bytes_per, scale_bytes_per),
        dtype=np.uint8,
    )
    groups = columns // 16
    scale_words = stored[_scale_index(rows, groups, columns)]
    divisor = struct.unpack(
        "<f", artifact.read_object(object_id, divisor_offset + matrix * 4, 4)
    )[0]

    code_f = e2m1_lut()[values][:rows_to_check]
    scale_f = e4m3_lut()[scale_words][:rows_to_check]
    return code_f, scale_f, groups, float(divisor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()

    out = {}
    with V3Artifact(args.artifact) as artifact:
        targets = [
            ("gate/m0", artifact.parts("text/layers/0/moe/experts/gate")[0].object, 0),
            ("up/m0", artifact.parts("text/layers/0/moe/experts/up")[0].object, 0),
            ("down/m0", artifact.parts("text/layers/0/moe/experts/down")[0].object, 0),
        ]
        for name, object_id, matrix in targets:
            code_f, scale_f, groups, divisor = load_matrix(
                artifact, object_id, matrix, args.rows
            )
            nrows = code_f.shape[0]
            code_groups = code_f.reshape(nrows, groups, 16)
            scale_groups = scale_f.reshape(nrows, groups)
            rms_code = np.sqrt(np.mean(code_groups * code_groups, axis=2))
            positive = rms_code > 0
            scale_log = np.log(scale_groups[positive])
            rms_log = np.log(rms_code[positive])
            rng = np.random.default_rng(20261117)
            control = corr(scale_log, rng.permutation(rms_log))

            scale_rep = np.repeat(scale_f, 16, axis=1)
            variants = {
                "current_code_mul_scale_div_divisor": code_f * scale_rep / divisor,
                "scale_divide_code_div_scale_div_divisor": code_f / np.where(
                    scale_rep == 0, 1.0, scale_rep
                ) / divisor,
                "divisor_multiply_code_mul_scale_mul_divisor": code_f * scale_rep * divisor,
                "both_flipped_code_div_scale_mul_divisor": code_f
                / np.where(scale_rep == 0, 1.0, scale_rep)
                * divisor,
            }
            stats = {}
            for label, weight in variants.items():
                wg = weight.reshape(nrows, groups, 16)
                group_var = np.var(wg, axis=2)
                stats[label] = {
                    "mean": float(np.mean(weight)),
                    "std": float(np.std(weight)),
                    "absmax": float(np.max(np.abs(weight))),
                    "group_var_mean": float(np.mean(group_var)),
                    "group_var_std": float(np.std(group_var)),
                    "finite": bool(np.all(np.isfinite(weight))),
                }
            out[name] = {
                "object": object_id,
                "matrix": matrix,
                "shape": list(artifact.object(object_id).shape),
                "divisor": divisor,
                "rows_checked": nrows,
                "scale_word_min": int(scale_f.min()),
                "scale_word_max": int(scale_f.max()),
                "scale_value_median": float(np.median(scale_f)),
                "rms_code_median": float(np.median(rms_code)),
                "corr_logscale_log_rmscode": corr(scale_log, rms_log),
                "shuffled_control": control,
                "variants": stats,
            }

    print(__import__("json").dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
