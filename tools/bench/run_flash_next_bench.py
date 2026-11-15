#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build" / "tests" / "ninfer_flash_next_engine_bench"
DEFAULT_ARTIFACT = Path("/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer")
DEFAULT_PROMPT = "1024 7 98140 33 4097 512 100 2048"
DEFAULT_CASES = "512,3700,15000,59000"
NCCL = "/home/cmj/nccl-lib/whl/nvidia/nccl/lib"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Measure the full Qwen3.8-Flash-Next model through the public Engine."
    )
    parser.add_argument("--artifact", default=os.environ.get("NINFER_FLASH_NEXT_ARTIFACT",
                                                             str(DEFAULT_ARTIFACT)))
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--prompt-ids", default=DEFAULT_PROMPT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--output", default="")
    return parser


def output_path(requested):
    if requested:
        return Path(requested)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "profiles" / "bench" / f"flash_next_bench_{stamp}.json"


def run(args):
    binary = Path(args.binary)
    if not binary.is_file():
        raise SystemExit(
            f"the Engine benchmark driver is missing: {binary}\n"
            "build it with: ninja -C build ninfer_flash_next_engine_bench"
        )
    if not Path(args.artifact).is_file():
        raise SystemExit(f"the artifact is missing: {args.artifact}")
    command = [
        str(binary), "--bench",
        "--artifact", args.artifact,
        "--prompt-ids", args.prompt_ids,
        "--cases", args.cases,
        "--decode-tokens", str(args.decode_tokens),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
        "--prefill-chunk", str(args.prefill_chunk),
    ]
    environment = dict(os.environ)
    environment["NINFER_FLASH_NEXT_ARTIFACT"] = args.artifact
    environment["LD_LIBRARY_PATH"] = NCCL + ":" + environment.get("LD_LIBRARY_PATH", "")
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        raise SystemExit(
            f"the Engine benchmark exited {completed.returncode}\n{completed.stdout}\n{completed.stderr}"
        )
    report = json.loads(completed.stdout)
    report["tool"] = "tools/bench/run_flash_next_bench.py"
    report["command"] = " ".join(shlex.quote(part) for part in command)
    report["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["notes"] = [
        "single request, greedy, tensor-parallel width 2 on 2 x Tesla V100-SXM2 32 GB",
        "MTP is outside this milestone, so acceptance is N/A",
        "the token-id corpus cycles the 8-token reference prompt to the requested occupancy",
        "decode throughput counts committed decode tokens divided by decode seconds",
    ]
    return report


def print_summary(report):
    print(f"artifact {report['artifact']}")
    print(f"device {report['device']['name']}  tp {report['engine']['tp']}  "
          f"kv-capacity {report['engine']['kv_capacity']}  "
          f"prefill-chunk {report['engine']['prefill_chunk']}")
    print(f"{'occupied':>10} {'prompt':>8} {'decode':>7} {'tok/s':>9} {'ttft ms':>9} "
          f"{'hit rate':>9} {'miss bytes':>12}")
    for case in report["cases"]:
        mean = case["mean"]
        print(f"{case['occupied_context']:>10} {case['prompt_tokens']:>8} "
              f"{case['decode_tokens']:>7} {mean['committed_tokens_per_second']:>9.2f} "
              f"{mean['ttft_seconds'] * 1e3:>9.1f} {mean['expert_hit_rate'] * 100:>8.2f}% "
              f"{mean['expert_miss_bytes']:>12}")


def main():
    args = build_parser().parse_args()
    report = run(args)
    path = output_path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
