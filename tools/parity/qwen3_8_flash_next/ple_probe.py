"""Find the PLE gate/formula variant that reproduces strata's post-PLE residual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.parity.qwen3_8_flash_next.layer_compare import read_records
from tools.reference.qwen3_8_flash_next.config import CFG, FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import (
    bf16,
    dense,
    ple_conv_add_norm,
    ple_embedding,
    sigmoid,
    silu,
)
from tools.reference.qwen3_8_flash_next.state import PleState


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("oracle_layers", type=Path)
    p.add_argument("oracle_ple", type=Path)
    p.add_argument("tokens", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tokens = [int(v) for v in args.tokens.read_text().split()][:4]
    pre = read_records(args.oracle_layers)
    post = read_records(args.oracle_ple)
    deltas = np.stack([post[(1, t)] - pre[(0, t)] for t in range(4)]).reshape(4, 4, CFG.hidden)

    with RefModel(args.artifact, config=FlashNextConfig()) as model:
        weights = model._ple_weights()
        rows = model.ple_rows(tokens)
        row_tables = {}
        embedded = np.empty((4, CFG.ple_heads, CFG.ple_head_width), dtype=np.float32)
        for t in range(4):
            for h in range(CFG.ple_heads):
                row = int(rows[t, h])
                shard, within = divmod(row, CFG.ple_shard_rows)
                key = (shard, within)
                if key not in row_tables:
                    row_tables[key] = model.store.ple_row(weights["table"][shard], within)
                embedded[t, h] = row_tables[key]
        embedding = np.asarray(bf16(embedded.reshape(4, CFG.hidden)), dtype=np.float64)
        key = np.asarray(dense(embedding, weights["key"]), dtype=np.float64).reshape(4, 4, CFG.hidden)
        value = np.asarray(dense(embedding, weights["value"]), dtype=np.float64)
        history = PleState().history
        eps = CFG.rms_epsilon
        kinds = {}
        for gain in ("1plusw", "w"):
            coeff = (1.0 + np.asarray(weights["norm_key"])) if gain == "1plusw" else np.asarray(weights["norm_key"])
            key_normed = key / np.sqrt(np.mean(key * key, axis=-1, keepdims=True) + eps) * coeff[None, :, :]
            query_source = np.stack([pre[(0, t)].reshape(4, CFG.hidden) for t in range(4)])
            qcoeff = (1.0 + np.asarray(weights["norm_query"])) if gain == "1plusw" else np.asarray(weights["norm_query"])
            query_normed = query_source / np.sqrt(np.mean(query_source * query_source, axis=-1, keepdims=True) + eps) * qcoeff[None, :, :]
            dot = np.sum(key_normed * query_normed, axis=-1) / np.sqrt(CFG.hidden)
            signed_sqrt = np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), CFG.ple_gate_floor))
            gates = {
                f"{gain}_signedsqrt": sigmoid(signed_sqrt),
                f"{gain}_rawdot_sigmoid": sigmoid(dot),
                f"{gain}_dot": dot,
                f"{gain}_sqrtabs_sigmoid": sigmoid(np.sqrt(np.maximum(np.abs(dot), CFG.ple_gate_floor))),
            }
            for name, gate in gates.items():
                gated = bf16(gate[:, :, None] * value[:, None, :])
                gated_view = np.asarray(gated, dtype=np.float64).reshape(4, 4, CFG.hidden)
                normalized_conv = bf16(
                    gated_view / np.sqrt(np.mean(gated_view * gated_view, axis=-1, keepdims=True) + eps)
                    * (1.0 + np.asarray(weights["norm_conv"]))[None, :, :]
                )
                updated, _ = ple_conv_add_norm(
                    history,
                    np.asarray(normalized_conv, dtype=np.float64).reshape(4, -1),
                    np.zeros((4, 4, CFG.hidden)),
                    np.asarray(gated, dtype=np.float64).reshape(4, -1),
                    weights["conv"], weights["norm_key"], eps,
                )
                conv_part = np.asarray(updated, dtype=np.float64).reshape(4, 4, CFG.hidden)
                pieces = np.stack([gated.reshape(4, -1), conv_part.reshape(4, -1)], axis=-1)
                target = deltas.reshape(4, -1)
                design = pieces.reshape(-1, 2)
                coefficients = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
                predicted = (design @ coefficients).reshape(target.shape)
                kinds[name] = {
                    "gated_norm": float(np.linalg.norm(gated)),
                    "conv_norm": float(np.linalg.norm(conv_part)),
                    "cos_sum_to_oracle": float(
                        np.dot((gated.reshape(4, -1) + conv_part.reshape(4, -1)).ravel(), target.ravel())
                        / (np.linalg.norm(gated.reshape(4, -1) + conv_part.reshape(4, -1)) * np.linalg.norm(target))
                    ),
                    "fit_coefficients_gated_conv": [float(v) for v in coefficients],
                    "fit_r2": float(
                        1.0 - np.sum((target - predicted) ** 2) / np.sum((target - target.mean()) ** 2)
                    ),
                }
    text = json.dumps({"oracle_delta_norm": float(np.linalg.norm(deltas)), "variants": kinds}, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ple_probe.json").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
