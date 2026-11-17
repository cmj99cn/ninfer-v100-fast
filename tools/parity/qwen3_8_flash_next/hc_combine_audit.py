"""收官诊断: hc_combine_norm 逐候选 4 层 NLL 实验 (只诊断, 不改产品/golden/§13).

每个候选在独立子进程运行, 让 tools/reference 的 import-time 开关生效; 评分器复用
layout_search.evaluate (固定 A32 tokens + 固定 "The capital of France is" 提示).
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

CANDIDATES = [
    ("baseline", {}, "当前转写 (行为保持基线, 期望 12.6216)"),
    ("hc_block_div4", {"NINFER_DIAG_HC_BLOCK_SCALE": "div4"},
     "mixer output 进入残差前再 /4 (h + 2sigmoid(b/4)*block/4)"),
    ("hc_inject_sigmoid_only", {"NINFER_DIAG_HC_INJECT_COEFF": "sigmoid_only"},
     "injection 系数 2*sigmoid(b/4) -> sigmoid(b/4) (去掉因子 2)"),
    ("hc_inject_no_sigmoid", {"NINFER_DIAG_HC_INJECT_COEFF": "no_sigmoid"},
     "injection 系数 2*sigmoid(b/4) -> 2*(b/4) (去掉 sigmoid)"),
    ("hc_norm_gain_w", {"NINFER_DIAG_HC_NORM_GAIN": "w_only"},
     "归一化增益 (1+w) -> w"),
    ("hc_combine_order_normfirst", {"NINFER_DIAG_HC_COMBINE_ORDER": "norm_then_update"},
     "先用旧 residual 算 normalized, 再更新 residual (顺序反转)"),
    ("hc_residual_noround", {"NINFER_DIAG_HC_RESIDUAL_ROUND": "noround"},
     "normalized 用未 BF16 舍入的 updated residual"),
    ("hc_combine_mul", {"NINFER_DIAG_HC_COMBINE_MODE": "mul"},
     "残差组合 h + inj*block -> h * block"),
    ("hc_combine_pre_scale", {"NINFER_DIAG_HC_COMBINE_MODE": "pre_scale"},
     "残差组合 h + inj*block -> h + inj*(block - h)"),
    ("D1_gdn_norm_sigmoid", {"NINFER_DIAG_GDN_NORM_GATE": "sigmoid"},
     "已知 D1: GDN norm gate silu -> sigmoid (单 op +0.28 nat, 作参照)"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--candidates", default="", help="comma-separated subset (default: all)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runner = ROOT / "tools/parity/qwen3_8_flash_next/layout_search.py"
    env = os.environ.copy()
    for key in {k for _, overrides, _ in CANDIDATES for k in overrides}:
        env.pop(key, None)

    names = list(CANDIDATES)
    if args.candidates:
        wanted = set(args.candidates.replace(",", " ").split())
        names = [c for c in names if c[0] in wanted]

    results = []
    stopped = ""
    for name, overrides, description in names:
        output = Path(f"/tmp/hc_audit_{name}.json")
        run_env = env.copy()
        run_env.update(overrides)
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, str(runner), str(args.artifact), str(args.tokens),
             "--layers", str(args.layers), "--candidates", "current", "--output", str(output)],
            cwd=ROOT, env=run_env, check=True, stdout=subprocess.DEVNULL,
        )
        wall = time.perf_counter() - started
        report = json.loads(output.read_text())[0]
        top = ", ".join(
            f"{entry['text']!r}:{entry['probability']:.3g}" for entry in report["prompt_top5"]
        )
        row = {
            "candidate": name, "description": description, "overrides": overrides,
            "mean_nll": report["mean_nll"], "top1_matches": report["top1_matches"],
            "mean_entropy": report["mean_entropy"], "mean_target_rank": report["mean_target_rank"],
            "prompt_top5": report["prompt_top5"],
            "paris_probability": report["prompt_paris"]["probability"],
            "paris_nll": report["prompt_paris"]["nll"], "paris_rank": report["prompt_paris"]["rank"],
            "total_seconds": report["total_seconds"], "driver_wall_seconds": wall,
            "last_row_max_delta": report["last_row_max_delta"],
        }
        results.append(row)
        print(
            f"{name:30s} nll={row['mean_nll']:8.4f} top1={row['top1_matches']:2d} "
            f"H={row['mean_entropy']:7.4f} rank={row['mean_target_rank']:8.0f} "
            f"paris_p={row['paris_probability']:.3g} top5=[{top}] t={wall:.1f}s",
            flush=True,
        )
        if args.output is not None:
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        if row["mean_nll"] < 8 or row["paris_probability"] > 0.05:
            stopped = name
            print(f"STOP cliff triggered by {name}: nll={row['mean_nll']:.4f} paris_p={row['paris_probability']:.4g}", flush=True)
            break
    if stopped:
        print(f"STOPPED_EARLY={stopped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
