"""Diagnostic: swap the artifact PLE table for the official IQ4_NL GGUF table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gguf import GGUFReader
from tokenizers import Tokenizer

from tools.parity.qwen3_8_flash_next.diag_nll import all_position_logits
from tools.parity.qwen3_8_flash_next.layer_compare import read_records
from tools.parity.qwen3_8_flash_next.ple_table_scan import dequant
from tools.reference.qwen3_8_flash_next.config import CFG, FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel

TOKENIZER = ROOT / (
    ".superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/"
    "assets/oracle/tokenizer.json"
)


def install_gguf_ple(model: RefModel, gguf: Path):
    reader = GGUFReader(gguf, "r")
    tensor = reader.tensors[0]
    handle = gguf.open("rb")
    weights = model._ple_weights()
    shard_index = {object_id: index for index, object_id in enumerate(weights["table"])}

    def read_row(shard_object, shard_row):
        global_row = shard_index[shard_object] * CFG.ple_shard_rows + int(shard_row)
        handle.seek(int(tensor.data_offset) + global_row * 90)
        return dequant(np.frombuffer(handle.read(90), dtype=np.uint8))[0]

    model.store.ple_row = read_row
    return handle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("gguf", type=Path)
    p.add_argument("tokens", type=Path)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--oracle-ple", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    tokens = [int(v) for v in args.tokens.read_text().split()]
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    with RefModel(args.artifact, config=FlashNextConfig(layers=args.layers)) as model:
        handle = install_gguf_ple(model, args.gguf)
        captured = model.forward(tokens, capture_tensors=True)
        norm = captured["tensors"][(args.layers - 1, "normalized_out")]
        logits = all_position_logits(model, norm)
        handle.close()
    position = len(tokens) - 1
    row = logits[position]
    order = np.argsort(-row, kind="stable")
    maximum = float(np.max(row))
    probability = np.exp(row - maximum) / np.sum(np.exp(row - maximum))
    report = {
        "layers": args.layers,
        "tokens": tokens,
        "position": position,
        "top5_ids": [int(i) for i in order[:5]],
        "top5_pieces": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in order[:5]],
        "top5_logits": [float(row[i]) for i in order[:5]],
        "target_11751_rank": int(np.flatnonzero(order == 11751)[0]),
        "target_11751_logit": float(row[11751]),
        "target_11751_probability": float(probability[11751]),
        "nll_11751": float(maximum + np.log(np.sum(np.exp(row - maximum))) - row[11751]),
        "logits_std": float(np.std(row)),
        "logits_mean": float(np.mean(row)),
        "entropy": float(-np.sum(probability * np.log(probability))),
    }
    if args.oracle_ple is not None:
        oracle = read_records(args.oracle_ple)
        ref = np.asarray(captured["tensors"][(0, "residual_out")][0], dtype=np.float64).ravel()
        target = oracle[(1, 0)]
        report["post_ple_residual_cos"] = float(np.dot(ref, target) / (np.linalg.norm(ref) * np.linalg.norm(target)))
        report["post_ple_residual_ratio"] = float(np.linalg.norm(ref) / np.linalg.norm(target))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"ple_gguf_override_L{args.layers}.json").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
