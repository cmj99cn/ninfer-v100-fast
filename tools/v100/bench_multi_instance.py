#!/usr/bin/env python3
"""Multi-instance serving load benchmark for the 8xV100 profiles.

Fires a fixed number of concurrent streaming chat-completion requests, each
carrying a long shared prefix (default 50K tokens per instance group), and
records:

  * committed decode tok/s from the /metrics scheduler counter delta,
  * client TTFT and (when a JSONL request log is supplied) the server TTFT,
  * MTP acceptance from request_done speculative counters,
  * queue wait (client TTFT minus server TTFT) plus sampled queue pressure.

The `instances` array and all aggregates are written to a timestamped JSON
report under `profiles/bench/`. The benchmark is deliberately separate from the
serve launcher: start `tools/v100/bench_multi_instance_8gpu.sh` in one terminal
and point `--request-log` at the same file the server writes.

`--dry-run` prints the request plan and target URLs only. It performs no HTTP
requests, creates no directories, and writes no files, so it is safe on a
2-GPU host.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen3.8-27b"
FILLER = (
    "The compiler lowers the fused attention pass into a single flat dispatch that "
    "schedules the prefill slab and the decode slab side by side. Each lane owns its "
    "paged KV blocks and the collective allreduce carries the row-parallel joins. "
)
WORDS_BACKFILL = (
    "attention prefill decode mixed tensor-parallel width tensor row column shard slab "
    "page block KV cache ring buffer allreduce collective kernel launch capture replay "
    "graph"
)


def make_prompt(prefix_tokens: int, group: int) -> str:
    """Deterministic prose prefix; the server reports the real token count."""
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    # English prose here tokenizes at roughly six characters per token; the server
    # reports the real count and the report records it, so this is only a target.
    target_chars = prefix_tokens * 6
    words = FILLER.split()
    while len(" ".join(words)) < target_chars:
        words += WORDS_BACKFILL.split()
    return f"Instance group {group} shared prefix marker {group}.\n" + " ".join(words)


def fetch_metrics(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def timing_stats(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))) - 1)]
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(p50, 3),
        "p90": round(p90, 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def stream_request(args, model: str, prompt: str, group: int, index: int,
                   start_event: threading.Event) -> dict:
    start_event.wait()
    parsed = urlsplit(args.url)
    port = parsed.port or 80
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    result = {
        "index": index,
        "group": group,
        "ok": False,
        "error": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "ttft_ms": 0.0,
        "e2e_ms": 0.0,
        "tpot_ms": 0.0,
        "finish_reason": None,
    }
    dispatch = time.monotonic()
    conn = http.client.HTTPConnection(parsed.hostname, port, timeout=args.timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            detail = response.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {response.status}: {detail[:300]}")

        ttft = 0.0
        completion_tokens = 0
        prompt_tokens = 0
        finish_reason = None
        last_token_time = dispatch
        for raw in response:
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                completion_tokens = int(usage.get("completion_tokens", completion_tokens))
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            emitted = bool(delta.get("content") or delta.get("reasoning") or
                           delta.get("tool_calls"))
            if emitted:
                now = time.monotonic()
                if ttft == 0.0:
                    ttft = now - dispatch
                last_token_time = now
        e2e = time.monotonic() - dispatch
        if ttft == 0.0:
            ttft = e2e
        if completion_tokens == 0 and finish_reason is not None:
            completion_tokens = 1
        decode_seconds = max(0.0, last_token_time - dispatch - ttft)
        result.update({
            "ok": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft * 1000.0,
            "e2e_ms": e2e * 1000.0,
            "tpot_ms": (decode_seconds / max(1, completion_tokens - 1)) * 1000.0,
            "finish_reason": finish_reason,
        })
    except Exception as exc:  # noqa: BLE001 - report per-request failures, keep the run
        result["error"] = str(exc)
        result["e2e_ms"] = (time.monotonic() - dispatch) * 1000.0
    finally:
        conn.close()
    return result


class MetricsSampler(threading.Thread):
    def __init__(self, url: str, interval: float):
        super().__init__(daemon=True)
        self.url = url
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples: list[dict] = []

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                metrics = fetch_metrics(self.url, timeout=max(1.0, self.interval * 4))
            except Exception:  # noqa: BLE001 - sampling is best-effort
                continue
            scheduler = metrics.get("scheduler") or {}
            instances = metrics.get("instances") or []
            self.samples.append({
                "waiting_requests": int(scheduler.get("waiting_requests", 0)),
                "running_requests": int(scheduler.get("running_requests", 0)),
                "instance_pending_total": sum(
                    int(item.get("pending_requests", 0)) for item in instances),
                "instance_active_total": sum(
                    int(item.get("active_requests", 0)) for item in instances),
                "instance_free_kv_pages_min": min(
                    (int(item.get("free_kv_pages", 0)) for item in instances), default=None),
            })

    def stop(self) -> None:
        self.stop_event.set()


def read_jsonl_slice(path: Path, offset: int) -> list[dict]:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    records = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarize_request_log(records: list[dict]) -> dict:
    done = [record for record in records if record.get("event") == "request_done"]
    errors = [record for record in records if record.get("event") == "request_error"]
    if not done:
        return {"available": False, "request_done": 0, "request_error": len(errors)}

    ttfts = []
    accepted = 0
    drafted = 0
    rounds = 0
    fallback = 0
    per_position: list[int] = []
    backend = None
    for record in done:
        timings = record.get("timings_seconds") or {}
        if "ttft" in timings:
            ttfts.append(float(timings["ttft"]) * 1000.0)
        spec = record.get("speculative") or {}
        backend = spec.get("backend", backend)
        accepted += int(spec.get("accepted_tokens", 0))
        drafted += int(spec.get("drafted_tokens", 0))
        rounds += int(spec.get("rounds", 0))
        fallback += int(spec.get("fallback_steps", 0))
        positions = spec.get("accepted_per_position") or []
        if len(per_position) < len(positions):
            per_position.extend([0] * (len(positions) - len(per_position)))
        for position, count in enumerate(positions):
            per_position[position] += int(count)

    acceptance = None
    if drafted > 0:
        acceptance = {
            "backend": backend,
            "rounds": rounds,
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_ratio": round(accepted / drafted, 4),
            "fallback_steps": fallback,
            "accepted_per_position": per_position,
        }
    return {
        "available": True,
        "request_done": len(done),
        "request_error": len(errors),
        "server_ttft_ms": timing_stats(ttfts),
        "acceptance": acceptance,
    }


def queue_pressure(samples: list[dict]) -> dict | None:
    if not samples:
        return None
    waiting = [sample["waiting_requests"] for sample in samples]
    pending = [sample["instance_pending_total"] for sample in samples]
    minimum_free = [sample["instance_free_kv_pages_min"] for sample in samples
                    if sample["instance_free_kv_pages_min"] is not None]
    return {
        "samples": len(samples),
        "max_waiting_requests": max(waiting),
        "mean_waiting_requests": round(statistics.mean(waiting), 3),
        "max_instance_pending_total": max(pending),
        "mean_instance_pending_total": round(statistics.mean(pending), 3),
        "min_instance_free_kv_pages": min(minimum_free) if minimum_free else None,
    }


def plan(args, groups: int) -> list[tuple[int, int]]:
    return [(index, index % groups) for index in range(args.num_requests)]


def dry_run(args, groups: int) -> None:
    print("multi-instance benchmark dry-run")
    print(f"  POST {args.url.rstrip('/')}/v1/chat/completions")
    print(f"  GET  {args.metrics_url}")
    print(f"  groups={groups} requests={args.num_requests} "
          f"concurrency={min(args.max_concurrency, args.num_requests)} "
          f"prefix_tokens={args.prefix_tokens} max_tokens={args.max_tokens} "
          f"model={args.model}")
    if args.request_log:
        print(f"  request log (read-only in a live run): {args.request_log}")
    print("  request plan (no requests sent, no files written):")
    for index, group in plan(args, groups):
        print(f"    #{index:02d} group={group} prefix_tokens={args.prefix_tokens} "
              f"max_tokens={args.max_tokens}")
    print("  start the server with tools/v100/bench_multi_instance_8gpu.sh first, "
          "then rerun without --dry-run")


def build_report(args, groups, start_metrics, end_metrics, results, wall_seconds,
                 request_log, samples, log_offset_before) -> dict:
    instances = end_metrics.get("instances") or []
    start_instances = start_metrics.get("instances") or []
    if len(start_instances) == len(instances):
        instances = [
            dict(item,
                 committed_tokens_delta=(int(item.get("committed_tokens", 0)) -
                                         int(start_instances[index].get("committed_tokens", 0))),
                 committed_tok_s=round(
                     (int(item.get("committed_tokens", 0)) -
                      int(start_instances[index].get("committed_tokens", 0))) /
                     wall_seconds, 3) if wall_seconds > 0 else 0.0)
            for index, item in enumerate(instances)
        ]

    start_committed = int((start_metrics.get("scheduler") or {}).get("committed_decode_tokens", 0))
    end_committed = int((end_metrics.get("scheduler") or {}).get("committed_decode_tokens", 0))
    committed_delta = max(0, end_committed - start_committed)
    start_prefill = int((start_metrics.get("scheduler") or {}).get("computed_prefill_tokens", 0))
    end_prefill = int((end_metrics.get("scheduler") or {}).get("computed_prefill_tokens", 0))
    prefill_delta = max(0, end_prefill - start_prefill)

    ok = [result for result in results if result["ok"]]
    ttfts = [result["ttft_ms"] for result in ok]
    prompt_tokens = [result["prompt_tokens"] for result in ok if result["prompt_tokens"] > 0]
    client_completion_tokens = sum(result["completion_tokens"] for result in ok)
    client_ttft = timing_stats(ttfts)

    log_summary = summarize_request_log(request_log) if request_log else {
        "available": False, "request_done": 0, "request_error": 0,
        "server_ttft_ms": None, "acceptance": None,
    }

    pressure = queue_pressure(samples)
    queue_wait = {
        "mean_estimate_ms": None,
        "mean_littles_law_ms": None,
        "source": "client TTFT mean minus request-log server TTFT mean",
        "note": ("mean_estimate_ms includes network/connection time; "
                 "mean_littles_law_ms is sampled waiting-requests divided by completion rate"),
    }
    if client_ttft and log_summary.get("server_ttft_ms"):
        mean_wait = client_ttft["mean"] - log_summary["server_ttft_ms"]["mean"]
        queue_wait["mean_estimate_ms"] = round(max(0.0, mean_wait), 3)
    if pressure and ok and wall_seconds > 0:
        completion_rate = len(ok) / wall_seconds
        queue_wait["mean_littles_law_ms"] = round(
            pressure["mean_waiting_requests"] / completion_rate * 1000.0, 3)

    acceptance = log_summary.get("acceptance")
    if acceptance is None:
        acceptance_note = ("no --request-log supplied" if not args.request_log else
                           "no MTP request_done records in the new log slice")
    else:
        acceptance_note = None

    return {
        "schema": "ninfer.multi_instance_bench.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "profile": {
            "url": args.url,
            "metrics_url": args.metrics_url,
            "model": args.model,
            "groups": groups,
            "num_requests": args.num_requests,
            "max_concurrency": args.max_concurrency,
            "prefix_tokens": args.prefix_tokens,
            "max_tokens": args.max_tokens,
        },
        "run": {
            "wall_seconds": round(wall_seconds, 6),
            "requests_ok": len(ok),
            "requests_failed": len(results) - len(ok),
            "client_completion_tokens": client_completion_tokens,
            "request_log_offset_before": log_offset_before,
            "request_log_records_read": len(request_log),
        },
        "aggregates": {
            "committed_decode_tokens_delta": committed_delta,
            "committed_tok_s": round(committed_delta / wall_seconds, 3)
            if wall_seconds > 0 else 0.0,
            "computed_prefill_tokens_delta": prefill_delta,
            "client_completion_tok_s": round(client_completion_tokens / wall_seconds, 3)
            if wall_seconds > 0 else 0.0,
            "ttft_ms": client_ttft,
            "server_ttft_ms": log_summary.get("server_ttft_ms"),
            "acceptance": acceptance,
            "acceptance_note": acceptance_note,
            "queue_wait_ms": queue_wait,
            "queue_pressure": pressure,
            "prompt_tokens": timing_stats([float(value) for value in prompt_tokens]),
            "metrics_scheduler_start": start_metrics.get("scheduler"),
            "metrics_scheduler_end": end_metrics.get("scheduler"),
            "prefix_cache_host_end": (end_metrics.get("prefix_cache") or {}).get("host"),
        },
        "instances": instances,
        "per_request": results,
        "notes": [
            "committed_tok_s uses the process-global scheduler counter delta divided by wall time; "
            "run this benchmark against a dedicated server process.",
            "The committed counter excludes the prefill-produced first token per request, so it is "
            "slightly below client_completion_tok_s.",
            "instances are ordered as reported by /metrics; per-instance tok/s uses the same wall.",
            "queue_wait_ms is an estimate; queue_pressure samples show admission-queue depth.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--metrics-url", default="",
                        help="default: <url>/metrics?format=json")
    parser.add_argument("--api-key",
                        default=os.environ.get("NINFER_API_KEY", "ninfer-eval-key"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--groups", type=int, default=0,
                        help="instance groups; 0 = detect from /metrics (4 in dry-run)")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--sample-interval", type=float, default=0.25,
                        help="seconds between /metrics samples during the run")
    parser.add_argument("--request-log", default=os.environ.get("NINFER_REQUEST_LOG", ""),
                        help="server --request-log-jsonl file for acceptance/server TTFT")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "profiles" / "bench"))
    parser.add_argument("--tag", default="",
                        help="report filename suffix; default is the UTC timestamp")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; send no requests and write no files")
    args = parser.parse_args()

    if args.num_requests <= 0 or args.max_concurrency <= 0:
        parser.error("--num-requests and --max-concurrency must be positive")
    if args.prefix_tokens <= 0 or args.max_tokens <= 0:
        parser.error("--prefix-tokens and --max-tokens must be positive")
    if not args.metrics_url:
        args.metrics_url = args.url.rstrip("/") + "/metrics?format=json"

    if args.dry_run:
        dry_run(args, args.groups if args.groups > 0 else 4)
        return 0

    start_metrics = fetch_metrics(args.metrics_url, timeout=30.0)
    detected_groups = len(start_metrics.get("instances") or [])
    groups = args.groups if args.groups > 0 else (detected_groups or 1)
    if groups <= 0:
        raise SystemExit("--groups must be positive")

    request_log_path = Path(args.request_log) if args.request_log else None
    log_offset_before = request_log_path.stat().st_size if (
        request_log_path and request_log_path.exists()) else 0

    prompts = {group: make_prompt(args.prefix_tokens, group) for group in range(groups)}
    start_event = threading.Event()
    sampler = MetricsSampler(args.metrics_url, args.sample_interval)
    sampler.start()

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.max_concurrency, args.num_requests)) as pool:
        futures = [
            pool.submit(stream_request, args, args.model, prompts[group], group, index,
                        start_event)
            for index, group in plan(args, groups)
        ]
        wall_start = time.monotonic()
        start_event.set()
        for future in futures:
            results.append(future.result())
    wall_end = time.monotonic()
    wall_seconds = wall_end - wall_start
    sampler.stop()
    sampler.join(timeout=2.0)
    time.sleep(0.2)

    end_metrics = fetch_metrics(args.metrics_url, timeout=30.0)
    request_log: list[dict] = []
    if request_log_path and request_log_path.exists():
        request_log = read_jsonl_slice(request_log_path, log_offset_before)

    report = build_report(args, groups, start_metrics, end_metrics, results, wall_seconds,
                          request_log, sampler.samples, log_offset_before)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = output_dir / f"multi_instance_{tag}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    aggregates = report["aggregates"]
    print(f"requests ok/failed : {report['run']['requests_ok']}/"
          f"{report['run']['requests_failed']}")
    print(f"wall               : {report['run']['wall_seconds']:.3f} s")
    print(f"committed tok/s    : {aggregates['committed_tok_s']:.3f}")
    print(f"client completion  : {aggregates['client_completion_tok_s']:.3f} tok/s")
    if aggregates["ttft_ms"]:
        print(f"client TTFT ms     : mean {aggregates['ttft_ms']['mean']:.1f} "
              f"p50 {aggregates['ttft_ms']['p50']:.1f} p90 {aggregates['ttft_ms']['p90']:.1f}")
    if aggregates["acceptance"]:
        print(f"MTP acceptance     : {aggregates['acceptance']['accepted_tokens']}/"
              f"{aggregates['acceptance']['drafted_tokens']} "
              f"({aggregates['acceptance']['acceptance_ratio'] * 100:.1f}%)")
    else:
        print(f"MTP acceptance     : unavailable ({aggregates['acceptance_note']})")
    wait = aggregates["queue_wait_ms"]
    estimate = "n/a" if wait["mean_estimate_ms"] is None else f"{wait['mean_estimate_ms']:.1f}"
    littles = "n/a" if wait["mean_littles_law_ms"] is None else f"{wait['mean_littles_law_ms']:.1f}"
    if wait["mean_estimate_ms"] is not None or wait["mean_littles_law_ms"] is not None:
        print(f"queue wait estimate: client/server {estimate} ms, Little's law {littles} ms")
    print(f"report             : {report_path}")
    return 0 if report["run"]["requests_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
