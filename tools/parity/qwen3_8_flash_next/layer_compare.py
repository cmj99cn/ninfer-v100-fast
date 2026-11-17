"""Compare FP64-reference layer residuals with strata's per-layer oracle dump."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def read_records(path: Path, width: int = 10240) -> dict:
    record = 16 + width * 4
    buf = path.read_bytes()
    assert len(buf) % record == 0, (len(buf), record)
    out = {}
    for offset in range(0, len(buf), record):
        layer, token, got_width, dtype = np.frombuffer(buf, dtype="<i4", count=4, offset=offset)
        assert got_width == width, (got_width, width)
        out[(int(layer), int(token))] = np.frombuffer(
            buf, dtype="<f4", count=width, offset=offset + 16
        ).astype(np.float64)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("strata_layers", type=Path)
    p.add_argument("ref_tensors", type=Path)
    p.add_argument("--tokens", type=int, default=5)
    p.add_argument("--layers", type=int, default=48)
    args = p.parse_args()
    oracle = read_records(args.strata_layers)
    ref = np.load(args.ref_tensors)
    print("layers cos/norm-ratio (mean over tokens 0..3); strata record layer l vs ref L{l}_residual_out")
    print(f"{'L':>3} {'cos':>8} {'ratio':>8} {'ref_norm':>10} {'oracle_norm':>12}")
    for layer in range(args.layers):
        cosines, ratios, ref_norms, oracle_norms = [], [], [], []
        for token in range(min(4, args.tokens)):
            key = (layer, token)
            if key not in oracle:
                continue
            a = oracle[key]
            name = f"L{layer}_residual_out"
            if name not in ref:
                continue
            b = np.asarray(ref[name][token], dtype=np.float64).ravel()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                continue
            cosines.append(float(np.dot(a, b) / denom))
            ratios.append(float(np.linalg.norm(b) / np.linalg.norm(a)))
            ref_norms.append(float(np.linalg.norm(b)))
            oracle_norms.append(float(np.linalg.norm(a)))
        if cosines:
            print(
                f"{layer:>3} {np.mean(cosines):>8.4f} {np.mean(ratios):>8.3f} "
                f"{np.mean(ref_norms):>10.3f} {np.mean(oracle_norms):>12.3f}"
            )
    a = oracle.get((-1, 0))
    b = ref.get("L0_residual_in")
    if a is not None and b is not None:
        for token in range(4):
            left = oracle[(-1, token)]
            right = np.asarray(b[token], dtype=np.float64).ravel()
            print(
                f"IN token={token} cos={np.dot(left,right)/(np.linalg.norm(left)*np.linalg.norm(right)):.4f} "
                f"ref_norm={np.linalg.norm(right):.3f} oracle_norm={np.linalg.norm(left):.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
