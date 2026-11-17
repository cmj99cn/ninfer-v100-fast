"""Diagnostic teacher-forced NLL of the FP64 reference on a token-id file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import bf16_from_u16, hc_final_mix


def all_position_logits(model, normalized):
    object_id = model._object("text/output_head")
    hidden = model.cfg.hidden
    vocab = int(model.artifact.object(object_id).shape[0])
    row_bytes = hidden * 2
    samples = np.asarray(hc_final_mix(normalized, model._final_hc()["down"],
                                      model._final_hc()["up"]), dtype=np.float64)
    tokens = samples.shape[0]
    logits = np.empty((tokens, vocab), dtype=np.float64)
    chunk = 8192
    for start in range(0, vocab, chunk):
        count = min(chunk, vocab - start)
        raw = model.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
        weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, hidden)
        logits[:, start : start + count] = samples @ np.asarray(weights, dtype=np.float64).T
    return logits


def nll_rows(logits, tokens):
    rows = []
    detail = []
    matches = 0
    for position in range(logits.shape[0] - 1):
        values = logits[position]
        maximum = float(np.max(values))
        shifted = np.exp(values - maximum)
        total = float(np.sum(shifted))
        partition = maximum + float(np.log(total))
        target = int(tokens[position + 1])
        probability = shifted / total
        order = np.argsort(-values, kind="stable")
        rank = int(np.flatnonzero(order == target)[0])
        entropy = float(-np.sum(probability * np.log(probability)))
        rows.append(partition - float(values[target]))
        matches += int(order[0]) == target
        detail.append({
            "target": target,
            "nll": rows[-1],
            "rank": rank,
            "target_probability": float(probability[target]),
            "top1": int(order[0]),
            "top1_probability": float(probability[order[0]]),
            "margin": float(values[order[0]] - values[order[1]]),
            "entropy": entropy,
            "logits_std": float(np.std(values)),
        })
    return rows, matches, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vocab-limit", type=int, default=0)
    parser.add_argument("--shifts", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.read_text().split()]
    config = FlashNextConfig(layers=args.layers)
    started = time.perf_counter()
    with RefModel(args.artifact, config=config) as model:
        loaded = time.perf_counter() - started
        forward_started = time.perf_counter()
        captured = model.forward(tokens, capture_tensors=True)
        forward_seconds = time.perf_counter() - forward_started
        normalized = captured["tensors"][(config.layers - 1, "normalized_out")]
        logits_started = time.perf_counter()
        logits = all_position_logits(model, normalized)
        logits_seconds = time.perf_counter() - logits_started
        # Sanity: the last row must reproduce forward's own last-position logits.
        last_delta = float(np.max(np.abs(logits[-1] - captured["logits"])))
    rows, matches, detail = nll_rows(logits, tokens)
    shifted_nll = {}
    if args.shifts:
        for offset in (int(value) for value in args.shifts.split(",")):
            values = []
            for position in range(logits.shape[0]):
                target_position = position + offset
                if target_position < 0 or target_position >= len(tokens):
                    continue
                row = logits[position]
                maximum = float(np.max(row))
                partition = maximum + float(np.log(np.sum(np.exp(row - maximum))))
                values.append(partition - float(row[int(tokens[target_position])]))
            shifted_nll[str(offset)] = float(np.mean(values)) if values else None
    limited = None
    if args.vocab_limit > 0:
        limited_rows = []
        for position in range(logits.shape[0] - 1):
            row = logits[position, : args.vocab_limit]
            maximum = float(np.max(row))
            partition = maximum + float(np.log(np.sum(np.exp(row - maximum))))
            limited_rows.append(partition - float(row[int(tokens[position + 1])]))
        limited = float(np.mean(limited_rows))
    marginal = np.mean(logits, axis=0)
    marginal_order = np.argsort(-marginal, kind="stable")[:24]
    report = {
        "shifted_nll": shifted_nll,
        "vocab_limited_nll": limited,
        "marginal_top": [int(value) for value in marginal_order],
        "marginal_top_values": [float(marginal[int(value)]) for value in marginal_order],
        "artifact": str(args.artifact),
        "layers": config.layers,
        "tokens": tokens,
        "mean_nll": float(np.mean(rows)),
        "top1_matches": matches,
        "mean_entropy": float(np.mean([row["entropy"] for row in detail])),
        "mean_logits_std": float(np.mean([row["logits_std"] for row in detail])),
        "mean_target_rank": float(np.mean([row["rank"] for row in detail])),
        "per_position": rows,
        "detail": detail,
        "load_seconds": loaded,
        "forward_seconds": forward_seconds,
        "logits_seconds": logits_seconds,
        "last_row_max_delta": last_delta,
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
