"""Diagnosis: top-5 next-token distribution at a chosen position of the FP64 reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.parity.qwen3_8_flash_next.diag_nll import all_position_logits
from tokenizers import Tokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("tokens", type=Path)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--position", type=int, default=-1)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    tokens = [int(v) for v in args.tokens.read_text().split()]
    cfg = FlashNextConfig(layers=args.layers)
    tok = Tokenizer.from_file("/tmp/opencode/oracle/tokenizer.json")
    with RefModel(args.artifact, config=cfg) as model:
        cap = model.forward(tokens, capture_tensors=True)
        norm = cap["tensors"][(cfg.layers - 1, "normalized_out")]
        logits = all_position_logits(model, norm)

    pos = args.position if args.position >= 0 else logits.shape[0] - 1
    row = logits[pos]
    mx = float(np.max(row))
    probs = np.exp(row - mx)
    probs /= float(np.sum(probs))
    order = np.argsort(-row, kind="stable")[: args.top]
    print(f"layers={args.layers} position={pos} target={tokens[pos+1] if pos+1 < len(tokens) else 'N/A'}")
    for i in order:
        piece = tok.decode([int(i)], skip_special_tokens=False)
        print(f"  id={int(i):>6}  prob={float(probs[i]):.6f}  piece={piece!r}")
    target = tokens[pos + 1] if pos + 1 < len(tokens) else None
    if target is not None:
        order_all = np.argsort(-row, kind="stable")
        rank = int(np.flatnonzero(order_all == target)[0])
        print(f"  TARGET id={target} prob={float(probs[target]):.8f} rank={rank}")


if __name__ == "__main__":
    raise SystemExit(main())
