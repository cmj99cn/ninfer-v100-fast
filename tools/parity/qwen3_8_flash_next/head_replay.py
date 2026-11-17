"""Replay our output chain on the oracle's final residual to isolate the head."""

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

from tools.reference.qwen3_8_flash_next.bindings import V3Artifact
from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import bf16_from_u16, hc_final_mix

TOKENIZER = ROOT / (
    ".superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/"
    "assets/oracle/tokenizer.json"
)


def load_residual(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u4")
    t, hc, hidden = (int(v) for v in raw[:3])
    return raw[3:].view("<f4").astype(np.float64).reshape(t, hc, hidden)


def head_logits(model: RefModel, object_name: str, sample: np.ndarray) -> np.ndarray:
    object_id = model._object(object_name)
    vocab = int(model.artifact.object(object_id).shape[0])
    hidden = model.cfg.hidden
    row_bytes = hidden * 2
    out = np.empty((sample.shape[0], vocab), dtype=np.float64)
    for start in range(0, vocab, 8192):
        count = min(8192, vocab - start)
        raw = model.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
        weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, hidden)
        out[:, start : start + count] = sample @ weights.astype(np.float64).T
    return out


def describe(row: np.ndarray, oracle: np.ndarray, target: int, tokenizer: Tokenizer) -> dict:
    order = np.argsort(-row, kind="stable")
    oorder = np.argsort(-oracle, kind="stable")
    mx = float(np.max(row))
    probs = np.exp(row - mx)
    total = float(np.sum(probs))
    return {
        "std": float(np.std(row)),
        "mean": float(np.mean(row)),
        "max": mx,
        "nll_target": float(mx + np.log(total) - row[target]),
        "target_rank": int(np.flatnonzero(order == target)[0]),
        "top5": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in order[:5]],
        "top5_ids": [int(i) for i in order[:5]],
        "oracle_top5": [int(i) for i in oorder[:5]],
        "top5_overlap": len(set(order[:5]) & set(oorder[:5])),
        "correlation_to_oracle": float(np.corrcoef(row, oracle)[0, 1]),
        "cosine_to_oracle": float(np.dot(row, oracle) / (np.linalg.norm(row) * np.linalg.norm(oracle))),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("residual", type=Path)
    p.add_argument("oracle_logits", type=Path)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    residual = load_residual(args.residual)
    oracle = np.fromfile(args.oracle_logits, dtype="<f4").astype(np.float64)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    target = 11751
    cfg = FlashNextConfig(layers=args.layers)
    with RefModel(args.artifact, config=cfg) as model:
        final = model._final_hc()
        sample = hc_final_mix(residual, final["down"], final["up"])
        replay = head_logits(model, "text/output_head", sample)[0]
        tied = head_logits(model, "text/token_embedding", sample)[0]
        scale = float(np.dot(replay, oracle) / np.dot(replay, replay))
        bias = float(np.mean(oracle - scale * replay))
        fitted = scale * replay + bias
        residual_top = np.argsort(-(oracle - fitted))[:10]
        tie = {}
        for token_id in (11751, 271, 198, 220, 248044):
            rows = []
            for name in ("text/output_head", "text/token_embedding"):
                object_id = model._object(name)
                row_bytes = cfg.hidden * 2
                raw = model.artifact.read_object(object_id, int(token_id) * row_bytes, row_bytes)
                rows.append(bf16_from_u16(np.frombuffer(raw, dtype="<u2")).astype(np.float64))
            tie[str(token_id)] = float(np.dot(rows[0], rows[1]) / (np.linalg.norm(rows[0]) * np.linalg.norm(rows[1])))
        design = np.stack([replay, tied, np.ones_like(replay)], axis=1)
        coefficients = np.linalg.lstsq(design, oracle, rcond=None)[0]
        combined = design @ coefficients
        combined_top = np.argsort(-combined)[:10]
        residual_top = np.argsort(-(oracle - combined))[:10]
        report = {
            "residual_shape": list(residual.shape),
            "residual_norm": float(np.linalg.norm(residual[0])),
            "sample_norm": float(np.linalg.norm(sample[0])),
            "output_head": describe(replay, oracle, target, tokenizer),
            "tied_embedding": describe(tied, oracle, target, tokenizer),
            "linear_fit": {
                "scale": scale,
                "bias": bias,
                "r2": float(1.0 - np.sum((oracle - fitted) ** 2) / np.sum((oracle - np.mean(oracle)) ** 2)),
                "fitted_target_rank": int(np.flatnonzero(np.argsort(-fitted) == target)[0]),
                "top_underfit_ids": [int(i) for i in residual_top],
                "top_underfit_pieces": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in residual_top],
            },
            "head_vs_embedding_row_cosine": tie,
            "two_table_fit": {
                "coefficients_head_embedding_bias": [float(v) for v in coefficients],
                "r2": float(1.0 - np.sum((oracle - combined) ** 2) / np.sum((oracle - np.mean(oracle)) ** 2)),
                "combined_top_ids": [int(i) for i in combined_top],
                "combined_top_pieces": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in combined_top],
                "residual_top_pieces": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in residual_top],
            },
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "head_replay.json").write_text(text + "\n")
        np.save(args.out / "oracle_residual.npy", residual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
