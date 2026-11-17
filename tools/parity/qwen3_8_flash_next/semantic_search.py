"""Coordinate-descent semantic-transcription search on the FP64 4-layer reference.

Only tools/reference and tools/parity diagnostics are touched.  Each candidate runs in an
isolated subprocess so the import-time switches in tools/reference take effect; the scorer is
the existing layout_search.evaluate (fixed A32 tokens + fixed "The capital of France is" prompt).
No product code, golden or artifact is modified.
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
    ("baseline", "A", {}, "current transcription (reference baseline)"),
    ("A_rope_adjacent", "A", {"NINFER_DIAG_ROPE_MODE": "adjacent"}, "adjacent even/odd pairs in first 64 dims (vs half-split)"),
    ("A_rope_full_head", "A", {"NINFER_DIAG_ROPE_MODE": "full_head"}, "full 256-dim rotary, 128 pairs"),
    ("A_theta_1e4", "A", {"NINFER_DIAG_ROPE_THETA": "10000"}, "rope_theta 1e7 -> 1e4"),
    ("A_theta_1e6", "A", {"NINFER_DIAG_ROPE_THETA": "1000000"}, "rope_theta 1e7 -> 1e6"),
    ("A_rope_sign_flip", "A", {"NINFER_DIAG_ROPE_SIGN_FLIP": "1"}, "cos/sin sign convention flipped"),
    ("A_pos_offset_1", "A", {"NINFER_DIAG_ROPE_POS_OFFSET": "1"}, "positions start at 1 instead of 0"),
    ("A_rotary_dim_128", "A", {"NINFER_DIAG_ROTARY_DIM": "128"}, "rotary_dim 64 -> head_dim/2 = 128"),
    ("A_theta_5e5", "A", {"NINFER_DIAG_ROPE_THETA": "500000"}, "rope_theta 1e7 -> 5e5"),
    ("A_rope_tail", "A", {"NINFER_DIAG_ROPE_MODE": "tail"}, "rotary on last 64 dims of head"),
    ("A_index_head_axis", "A", {"NINFER_DIAG_ROPE_HEAD_POS": "index_axis"}, "index-query 4 heads use pos//2^h (MRoPE-axis probe)"),
    ("B_hcmix_no_stream_div", "B", {"NINFER_DIAG_HC_MIX_MODE": "no_stream_div"}, "drop 1/nstreams scaling in hc_mix"),
    ("B_hcmix_no_gate_sigmoid", "B", {"NINFER_DIAG_HC_MIX_MODE": "no_gate_sigmoid"}, "drop sigmoid on hc_mix gates"),
    ("B_hccombine_convex", "B", {"NINFER_DIAG_HC_COMBINE_MODE": "convex"}, "residual*(1-a) + block*a"),
    ("B_stream_pair_swap", "B", {"NINFER_DIAG_HC_STREAM_PERM": "pair_swap"}, "swap HC streams 0<->1, 2<->3"),
    ("B_stream_rot1", "B", {"NINFER_DIAG_HC_STREAM_PERM": "rot1"}, "cyclic shift of HC streams by 1"),
    ("B_hcmix_inject_linear", "B", {"NINFER_DIAG_HC_MIX_MODE": "inject_linear"}, "injection linear instead of 2*sigmoid"),
    ("B_stream_reverse", "B", {"NINFER_DIAG_HC_STREAM_PERM": "reverse"}, "reverse HC stream order"),
    ("B_norm_flatten", "B", {"NINFER_DIAG_HC_NORM_MODE": "flatten"}, "HC norm over flattened 10240 dims"),
    ("B_hccombine_no_injection", "B", {"NINFER_DIAG_HC_COMBINE_MODE": "no_injection"}, "residual + block (no per-stream injection)"),
    ("B_eps_1e5", "B", {"NINFER_DIAG_RMS_EPS": "1e-5"}, "rms epsilon 1e-6 -> 1e-5"),
    ("C_gate_swap_ab", "C", {"NINFER_DIAG_GDN_GATE_MODE": "swap_ab"}, "swap a/b control rows in gdn_gating"),
    ("C_update_no_alpha_sk", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "no_alpha_sk"}, "delta = beta*(v - sk), no decay on read"),
    ("C_update_kv_swapped", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "kv_swapped"}, "swap k/v roles in state write"),
    ("C_gate_sigmoid", "C", {"NINFER_DIAG_GDN_GATE_MODE": "gate_sigmoid"}, "decay softplus -> sigmoid(a+bias)"),
    ("C_update_out_uses_k", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "out_uses_k"}, "read state with k instead of q"),
    ("C_conv_reverse_taps", "C", {"NINFER_DIAG_GDN_CONV_MODE": "reverse_taps"}, "reverse causal-conv tap/time order"),
    ("C_update_alpha_outer", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "alpha_outer"}, "decay applied after delta write"),
    ("C_decay_sigmoid", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "decay_sigmoid"}, "alpha = sigmoid(g) instead of exp(g)"),
    ("C_decay_direct", "C", {"NINFER_DIAG_GDN_GATE_MODE": "decay_direct"}, "decay g = -(a + dt_bias)"),
    ("C_update_no_normalize", "C", {"NINFER_DIAG_GDN_UPDATE_MODE": "no_normalize"}, "skip L2 normalization of q/k"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--group", default="A", help="A, B, C or all")
    parser.add_argument("--limit", type=int, default=0, help="max candidates after baseline (0 = all)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--candidates", default="", help="comma-separated candidate names to restrict the run")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runner = ROOT / "tools/parity/qwen3_8_flash_next/layout_search.py"
    env = os.environ.copy()
    for name in {k for _, _, overrides, _ in CANDIDATES for k in overrides}:
        env.pop(name, None)

    names = [c for c in CANDIDATES if c[1] in (args.group.split(",") if args.group != "all" else ("A", "B", "C"))]
    if args.candidates:
        wanted = set(args.candidates.replace(",", " ").split())
        names = [c for c in names if c[0] in wanted]
    if args.skip_baseline:
        names = [c for c in names if c[0] != "baseline"]
    if args.limit > 0:
        non_baseline = [c for c in names if c[0] != "baseline"]
        names = [c for c in names if c[0] == "baseline"] + non_baseline[: args.limit]

    results = []
    stopped = ""
    for name, group, overrides, description in names:
        output = Path(f"/tmp/semantic_{name}.json")
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
            "candidate": name,
            "group": group,
            "description": description,
            "overrides": overrides,
            "mean_nll": report["mean_nll"],
            "top1_matches": report["top1_matches"],
            "mean_entropy": report["mean_entropy"],
            "mean_target_rank": report["mean_target_rank"],
            "prompt_top5": report["prompt_top5"],
            "paris_probability": report["prompt_paris"]["probability"],
            "paris_nll": report["prompt_paris"]["nll"],
            "paris_rank": report["prompt_paris"]["rank"],
            "total_seconds": report["total_seconds"],
            "driver_wall_seconds": wall,
            "last_row_max_delta": report["last_row_max_delta"],
        }
        results.append(row)
        print(
            f"{name:28s} nll={row['mean_nll']:8.4f} top1={row['top1_matches']:2d} "
            f"entropy={row['mean_entropy']:7.4f} rank={row['mean_target_rank']:9.0f} "
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
