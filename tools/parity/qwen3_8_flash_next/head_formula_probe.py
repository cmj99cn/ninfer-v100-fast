"""Search the final-HC/head formula that maps strata's residual to strata's logits."""

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

from tools.parity.qwen3_8_flash_next.head_replay import head_logits, load_residual
from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import hc_final_mix

TOKENIZER = ROOT / (
    ".superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/"
    "assets/oracle/tokenizer.json"
)


def score(row: np.ndarray, oracle: np.ndarray, tokenizer: Tokenizer) -> dict:
    order = np.argsort(-row, kind="stable")
    return {
        "correlation": float(np.corrcoef(row, oracle)[0, 1]),
        "cosine": float(np.dot(row, oracle) / (np.linalg.norm(row) * np.linalg.norm(oracle))),
        "std": float(np.std(row)),
        "nll_11751": float(np.max(row) + np.log(np.sum(np.exp(row - np.max(row)))) - row[11751]),
        "rank_11751": int(np.flatnonzero(order == 11751)[0]),
        "top5": [tokenizer.decode([int(i)], skip_special_tokens=False) for i in order[:5]],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("residual", type=Path)
    p.add_argument("oracle_logits", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    r = load_residual(args.residual)[0]
    oracle = np.fromfile(args.oracle_logits, dtype="<f4").astype(np.float64)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    cfg = FlashNextConfig()
    with RefModel(args.artifact, config=cfg) as model:
        final = model._final_hc()
        w = np.asarray(final["norm"], dtype=np.float64)
        inv = 1.0 / np.sqrt(np.mean(r * r, axis=-1, keepdims=True) + 1e-6)
        variants = {
            "norm_1plusw": r * inv * (1.0 + w),
            "norm_w": r * inv * w,
            "norm_raw": r.copy(),
            "norm_inv_only": r * inv,
        }
        report = {}
        for name, normalized in variants.items():
            sample = hc_final_mix(normalized[None], final["down"], final["up"])
            logits = head_logits(model, "text/output_head", sample)[0]
            report[name] = score(logits, oracle, tokenizer)
        sample = hc_final_mix(variants["norm_1plusw"][None], final["down"], final["up"])
        report["oracle"] = score(oracle, oracle, tokenizer)
        text = json.dumps(report, indent=2)
        print(text)
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "head_formula_probe.json").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
