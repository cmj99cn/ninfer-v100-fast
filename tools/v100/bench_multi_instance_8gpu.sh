#!/usr/bin/env bash
# 8xV100 multi-instance serving profiles (8xTP1, 4xTP2, or 2xTP4).
# 8xTP1 keeps eight full model copies and therefore requires 32GB cards.
#
# This script only constructs and runs the serve command. Benchmarking is a
# separate process (tools/v100/bench_multi_instance.py) so the server stays up
# across runs. All profiles use the artifact selected by
# NINFER_V100X2_ARTIFACT/ARTIFACT (local NVFP4 default), INT8 group-64 KV, the
# hybrid prefix cache with one shared pinned host tier, and MTP with at most
# three drafts.
#
# `--dry-run` prints the complete command and exits without creating
# directories, touching the artifact, or allocating any GPU. It is safe on a
# 2-GPU host.
#
# Usage:
#   bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1 --dry-run
#   bash tools/v100/bench_multi_instance_8gpu.sh 4xTP2 --dry-run
#   bash tools/v100/bench_multi_instance_8gpu.sh 2xTP4 --dry-run
#   bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1
#   bash tools/v100/bench_multi_instance_8gpu.sh 4xTP2
#   bash tools/v100/bench_multi_instance_8gpu.sh 2xTP4
#
# Environment overrides:
#   NINFER              serve executable (default: <repo>/build/apps/ninfer-serve)
#   NINFER_V100X2_ARTIFACT | ARTIFACT
#                       model artifact (default: local NVFP4 artifact; override for the 8-card host)
#   HOST / PORT         listen address / port (default: 0.0.0.0 / 8080)
#   API_KEY             bearer token (default: ninfer-eval-key; empty disables auth)
#   MAX_CONTEXT         logical context ceiling (default: 262144)
#   GLOBAL_PENDING      shared admission queue bound (default: 16)
#   HOST_CACHE_MIB      shared pinned host tier (default: 153600 = 150 GiB)
#   DRAFT_TOKENS        MTP draft window (default: 3)
#   REQUEST_LOG         JSONL request log (default: profiles/bench/multi_instance_<mode>.requests.jsonl)
set -euo pipefail

usage() {
  echo "usage: $0 [8xTP1|4xTP2|2xTP4] [--dry-run]" >&2
  exit 2
}

help() {
  echo "usage: $0 [8xTP1|4xTP2|2xTP4] [--dry-run]"
  echo "  --dry-run  print the complete serve command without starting anything"
  echo "  env: NINFER NINFER_V100X2_ARTIFACT HOST PORT API_KEY MAX_CONTEXT"
  echo "       GLOBAL_PENDING HOST_CACHE_MIB DRAFT_TOKENS REQUEST_LOG"
  exit 0
}

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ninfer=${NINFER:-"${repo_dir}/build/apps/ninfer-serve"}
artifact=${NINFER_V100X2_ARTIFACT:-${ARTIFACT:-/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer}}
host=${HOST:-0.0.0.0}
port=${PORT:-8080}
api_key=${API_KEY-ninfer-eval-key}
max_context=${MAX_CONTEXT:-262144}
global_pending=${GLOBAL_PENDING:-16}
host_cache_mib=${HOST_CACHE_MIB:-153600}
draft_tokens=${DRAFT_TOKENS:-3}

mode=
dry_run=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -h|--help) help ;;
    8xTP1|4xTP2|2xTP4)
      [[ -z "$mode" ]] || usage
      mode=$arg
      ;;
    *) usage ;;
  esac
done
mode=${mode:-4xTP2}

case "$mode" in
  8xTP1)
    groups=8
    tp=1
    conc=1
    kv_capacity=262144
    group_devices="0;1;2;3;4;5;6;7"
    ;;
  4xTP2)
    groups=4
    tp=2
    conc=4
    kv_capacity=1048576
    group_devices="0,1;2,3;4,5;6,7"
    ;;
  2xTP4)
    groups=2
    tp=4
    conc=8
    kv_capacity=2097152
    group_devices="0,1,2,3;4,5,6,7"
    ;;
  *) usage ;;
esac

request_log=${REQUEST_LOG:-"${repo_dir}/profiles/bench/multi_instance_${mode}.requests.jsonl"}

cmd=(
  "$ninfer" "$artifact"
  --host "$host" --port "$port"
  --api-key "$api_key"
  --groups "$groups" --group-devices "$group_devices" --tp "$tp"
  --max-context "$max_context"
  --kv-capacity "$kv_capacity"
  --max-concurrency "$conc"
  --global-pending-requests "$global_pending"
  --kv-dtype int8
  --spec mtp --draft-tokens "$draft_tokens"
  --prefix-cache --host-cache-mib "$host_cache_mib"
  --no-thinking --greedy
  --request-log-jsonl "$request_log"
)

if [[ "$dry_run" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -x "$ninfer" ]] || { echo "missing executable: $ninfer (set NINFER)" >&2; exit 1; }
[[ -f "$artifact" ]] || { echo "missing artifact: $artifact (set NINFER_V100X2_ARTIFACT)" >&2; exit 1; }
mkdir -p "$(dirname -- "$request_log")"

exec "${cmd[@]}"
