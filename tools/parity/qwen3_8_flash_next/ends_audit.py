"""Compare FP64-reference logits with strata's oracle logits at one prompt position."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tokenizers import Tokenizer

from tools.parity.qwen3_8_flash_next.diag_nll import all_position_logits
from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel

TOKENIZER = ROOT / (
    ".superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/"
    "assets/oracle/tokenizer.json"
)


def load_oracle(path: Path, vocab: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == vocab:
        return raw.astype(np.float64)
    if raw.size >= vocab + 2 and int(raw.view("<i4")[0]) == vocab:
        return raw[2 : 2 + vocab].astype(np.float64)
    raise ValueError(f"unexpected oracle logits file: {raw.size} floats, vocab {vocab}")


def describe(row: np.ndarray, target: int, tokenizer: Tokenizer, top: int) -> dict:
    maximum = float(np.max(row))
    probs = np.exp(row - maximum)
    total = float(np.sum(probs))
    probs /= total
    order = np.argsort(-row, kind="stable")[:top]
    all_order = np.argsort(-row, kind="stable")
    rank = int(np.flatnonzero(all_order == target)[0])
    return {
        "mean": float(np.mean(row)),
        "std": float(np.std(row)),
        "min": float(np.min(row)),
        "max": maximum,
        "entropy": float(-np.sum(probs * np.log(probs))),
        "nll_target": float(maximum + np.log(total) - row[target]),
        "target_rank": rank,
        "target_probability": float(probs[target]),
        "top_ids": [int(i) for i in order],
        "top_pieces": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in order],
        "top_logits": [float(row[i]) for i in order],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("tokens", type=Path)
    p.add_argument("oracle_logits", type=Path)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tokens = [int(v) for v in args.tokens.read_text().split()]
    cfg = FlashNextConfig(layers=args.layers)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    with RefModel(args.artifact, config=cfg) as model:
        captured = model.forward(tokens, capture_tensors=True)
        norm = captured["tensors"][(cfg.layers - 1, "normalized_out")]
        logits = all_position_logits(model, norm)

    vocab = logits.shape[1]
    oracle = load_oracle(args.oracle_logits, vocab)
    pos = args.position if args.position >= 0 else len(tokens) - 1
    target = tokens[pos + 1] if pos + 1 < len(tokens) else 11751
    ref_row = logits[pos]
    a, b = ref_row, oracle
    corr = float(np.corrcoef(a, b)[0, 1])
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    report = {
        "artifact": str(args.artifact),
        "layers": args.layers,
        "tokens": tokens,
        "position": pos,
        "target": target,
        "vocab": vocab,
        "reference": describe(ref_row, target, tokenizer, args.top),
        "oracle": describe(oracle, target, tokenizer, args.top),
        "correlation": corr,
        "cosine": cos,
        "mean_abs_delta": float(np.mean(np.abs(a - b))),
        "top5_overlap": len(
            set(np.argsort(-a)[: args.top]) & set(np.argsort(-b)[: args.top])
        ),
        "top5_oracle_in_ref_ranks": [
            int(np.flatnonzero(np.argsort(-a) == i)[0]) for i in np.argsort(-b)[: args.top]
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / f"ref_logits_L{args.layers}.npy", logits)
        (args.out / f"compare_L{args.layers}.json").write_text(text + "\n")
        tensors = captured["tensors"]
        np.savez(
            args.out / f"ref_tensors_L{args.layers}.npz",
            **{
                f"L{layer}_{name}": np.asarray(value, dtype=np.float64)
                for (layer, name), value in tensors.items()
                if name in (
                    "residual_in", "residual_out", "normalized_in", "normalized_out",
                    "attention_block", "attention_injection", "mixer", "stage",
                    "normalized_moe", "moe_block", "moe_injection", "moe", "index_query", "index_keys",
                )
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
