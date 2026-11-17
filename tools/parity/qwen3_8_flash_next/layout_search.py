"""Coordinate-descent row-layout search over the Flash-Next fused parents.

The search swaps one fused-parent row mapping at a time, measures the teacher-forced NLL of a
fixed real-text token sequence on the FP64 reference, and reports the top-5 next tokens of a
fixed natural prompt as the direct evidence.

The reference side owns the candidate mapping (``tools/reference/qwen3_8_flash_next/layouts.py``);
this driver only runs and scores.  No product code is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.layouts import CANDIDATES, CURRENT, candidate
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import hc_final_mix


def decode_tokens(artifact, ids):
    """Human readable token strings, taken from the artifact's own tokenizer resource."""
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(artifact.resource("tokenizer.json").decode("utf-8"))
        return [tokenizer.id_to_token(int(token)) for token in ids]
    except Exception as error:  # pragma: no cover - diagnostics only
        return [f"<{int(token)}>" for token in ids] + [f"decode failed: {error}"]


def nll_report(logits, tokens):
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
        detail.append({"nll": rows[-1], "rank": rank, "entropy": entropy})
    return {
        "mean_nll": float(np.mean(rows)),
        "top1_matches": matches,
        "mean_entropy": float(np.mean([row["entropy"] for row in detail])),
        "mean_target_rank": float(np.mean([row["rank"] for row in detail])),
        "per_position": rows,
    }


def top_k(logits_row, k=5):
    maximum = float(np.max(logits_row))
    shifted = np.exp(logits_row - maximum)
    total = float(np.sum(shifted))
    order = np.argsort(-logits_row, kind="stable")[:k]
    return [(int(index), float(shifted[index] / total)) for index in order]


def evaluate(artifact, config, layout, tokens, prompt_ids, paris_id):
    started = time.perf_counter()
    with RefModel(artifact, config=config, layout=layout) as model:
        loaded = time.perf_counter()
        captured = model.forward(tokens, capture_tensors=True)
        forward_seconds = time.perf_counter() - loaded
        normalized = captured["tensors"][(config.layers - 1, "normalized_out")]
        samples = np.asarray(
            hc_final_mix(normalized, model._final_hc()["down"], model._final_hc()["up"]),
            dtype=np.float64,
        )
        logits_started = time.perf_counter()
        logits = model.position_logits(samples)
        logits_seconds = time.perf_counter() - logits_started
        report = nll_report(logits, tokens)
        prompt = model.forward(list(prompt_ids))
        prompt_logits = prompt["logits"]
        top = top_k(prompt_logits)
        strings = decode_tokens(model.artifact, [index for index, _ in top] + [paris_id])
        maximum = float(np.max(prompt_logits))
        shifted = np.exp(prompt_logits - maximum)
        total = float(np.sum(shifted))
        paris_nll = float(maximum + np.log(total) - prompt_logits[paris_id])
        paris_rank = int(np.flatnonzero(np.argsort(-prompt_logits, kind="stable") == paris_id)[0])
        last_delta = float(np.max(np.abs(logits[-1] - captured["logits"])))
    return {
        "mean_nll": report["mean_nll"],
        "top1_matches": report["top1_matches"],
        "mean_entropy": report["mean_entropy"],
        "mean_target_rank": report["mean_target_rank"],
        "per_position": report["per_position"],
        "prompt_top5": [
            {"token": index, "text": strings[position], "probability": probability}
            for position, (index, probability) in enumerate(top)
        ],
        "prompt_paris": {
            "token": paris_id,
            "text": strings[-1],
            "nll": paris_nll,
            "rank": paris_rank,
            "probability": float(shifted[paris_id] / total),
        },
        "forward_seconds": forward_seconds,
        "logits_seconds": logits_seconds,
        "load_seconds": loaded - started,
        "total_seconds": time.perf_counter() - started,
        "last_row_max_delta": last_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt-ids", default="760,6511,314,9338,369")
    parser.add_argument("--paris-id", type=int, default=11751)
    parser.add_argument("--candidates", default="all")
    parser.add_argument("--base", default="current", help="candidate the others are applied on top of")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    tokens = [int(value) for value in args.tokens.read_text().split()]
    prompt_ids = [int(value) for value in args.prompt_ids.replace(",", " ").split()]
    config = FlashNextConfig(layers=args.layers)
    names = list(CANDIDATES) if args.candidates == "all" else [
        value for value in args.candidates.replace(",", " ").split() if value
    ]
    base = candidate(args.base) if args.base != "current" else CURRENT
    results = []
    for name in names:
        layout = candidate(name, base)
        report = evaluate(args.artifact, config, layout, tokens, prompt_ids, args.paris_id)
        report["candidate"] = name
        report["base"] = args.base
        report["settings"] = {
            key: value
            for key, value in layout.__dict__.items()
            if key not in ("name",) and value != getattr(CURRENT, key)
        }
        results.append(report)
        top = ", ".join(
            f"{entry['text']!r}:{entry['probability']:.3g}" for entry in report["prompt_top5"]
        )
        print(
            f"{name:32s} nll={report['mean_nll']:8.4f} top1={report['top1_matches']:2d} "
            f"entropy={report['mean_entropy']:7.4f} rank={report['mean_target_rank']:9.0f} "
            f"paris_nll={report['prompt_paris']['nll']:7.4f} "
            f"top5=[{top}] t={report['total_seconds']:.1f}s",
            flush=True,
        )
        if args.output is not None:
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
