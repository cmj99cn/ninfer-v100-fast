"""Diagnostic: 4-layer NLL under alternative weight scale/divisor decode semantics.

Each candidate runs in an isolated subprocess so the env-var switches in
``tools/reference/qwen3_8_flash_next/weights.py`` take effect.  Product code,
golden and §13 are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATES = {
    "baseline": {},
    "nvfp4_scale_divide": {"NINFER_DIAG_NVFP4_SCALE_INVERSE": "1"},
    "nvfp4_divisor_multiply": {"NINFER_DIAG_NVFP4_DIVISOR_INVERSE": "1"},
    "w8_scale_divide": {"NINFER_DIAG_W8_SCALE_INVERSE": "1"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runner = ROOT / "tools/parity/qwen3_8_flash_next/layout_search.py"
    stats_runner = ROOT / "tools/parity/qwen3_8_flash_next/nvfp4_scale_stats.py"
    env = os.environ.copy()
    for key in set(CANDIDATES["baseline"]) | {k for c in CANDIDATES.values() for k in c}:
        env.pop(key, None)

    weights = {}
    stats_started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(stats_runner), str(args.artifact), "--rows", "64"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    weights = json.loads(result.stdout)
    stats_seconds = time.perf_counter() - stats_started

    table = {}
    for name, overrides in CANDIDATES.items():
        run_env = env.copy()
        run_env.update(overrides)
        output = Path(f"/tmp/nvfp4_audit_{name}.json")
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(runner),
                str(args.artifact),
                str(args.tokens),
                "--layers",
                str(args.layers),
                "--candidates",
                "current",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=run_env,
            check=True,
        )
        wall = time.perf_counter() - started
        report = json.loads(output.read_text())[0]
        table[name] = {
            "overrides": overrides,
            "mean_nll": report["mean_nll"],
            "top1_matches": report["top1_matches"],
            "mean_entropy": report["mean_entropy"],
            "mean_target_rank": report["mean_target_rank"],
            "prompt_top5": report["prompt_top5"],
            "prompt_paris": report["prompt_paris"],
            "load_seconds": report["load_seconds"],
            "forward_seconds": report["forward_seconds"],
            "logits_seconds": report["logits_seconds"],
            "total_seconds": report["total_seconds"],
            "driver_wall_seconds": wall,
            "last_row_max_delta": report["last_row_max_delta"],
        }
        top = ", ".join(
            f"{entry['text']!r}:{entry['probability']:.3g}"
            for entry in report["prompt_top5"]
        )
        print(
            f"{name:24s} nll={report['mean_nll']:9.4f} top1={report['top1_matches']:2d} "
            f"entropy={report['mean_entropy']:7.4f} "
            f"paris_nll={report['prompt_paris']['nll']:8.4f} "
            f"top5=[{top}] t={wall:.1f}s",
            flush=True,
        )

    payload = {
        "artifact": str(args.artifact),
        "tokens": str(args.tokens),
        "layers": args.layers,
        "weight_stats_seconds": stats_seconds,
        "weight_stats": weights,
        "candidates": table,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
