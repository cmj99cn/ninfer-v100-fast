#!/usr/bin/env bash
# Qwen3.8-27B official V3 artifact on 2xV100: FP8 text + Q6/Q5/Q4/W8 vision.
#
# The official artifact differs from the fork's v2 NVFP4-text artifact in two ways:
#   * text tensors are fp8_e4m3fn_row_bf16 -> NINFER_TEXT_RECIPE_FP8=1
#   * vision tensors are Q6G64 (no separate qkv) -> NINFER_VISION_RECIPE_QVIS=1
# Without both switches the loader stops at the first text/vision contract mismatch.
#
# Usage: bash tools/v100/serve-qwen38-official-v3.sh [start|stop|status]
#
# Environment overrides:
#   PORT (9101), HOST (0.0.0.0), MODEL_ID (qwen38-official-v3), API_KEY (ninfer-eval-key)
#   ARTIFACT, EXE, TP (2), DEVICES (0,1)
#   MAX_CONTEXT (262144), KV_CAPACITY (auto), KV_DTYPE (int8)
#   MAX_CONCURRENCY (2), PREFILL_CHUNK (8192)
#   MEDIA_CACHE_MIB (1024), MEDIA_LIVE_MIB (2048)
#   PREFIX_CACHE (1), SPEC (empty; e.g. mtp), DRAFT_TOKENS (3)
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
exe=${EXE:-"${repo_dir}/build/apps/ninfer-serve"}
artifact=${ARTIFACT:-/home/cmj/models/ninfer-artifacts/qwen3_8_27b_nvfp4.ninfer}
host=${HOST:-0.0.0.0}
port=${PORT:-9101}
model_id=${MODEL_ID:-qwen38-official-v3}
api_key=${API_KEY:-ninfer-eval-key}
tp=${TP:-2}
devices=${DEVICES:-0,1}
max_context=${MAX_CONTEXT:-262144}
kv_capacity=${KV_CAPACITY:-auto}
kv_dtype=${KV_DTYPE:-int8}
max_concurrency=${MAX_CONCURRENCY:-2}
prefill_chunk=${PREFILL_CHUNK:-8192}
media_cache_mib=${MEDIA_CACHE_MIB:-1024}
media_live_mib=${MEDIA_LIVE_MIB:-2048}
prefix_cache=${PREFIX_CACHE:-1}
spec=${SPEC:-}
draft_tokens=${DRAFT_TOKENS:-3}
pidfile=${PIDFILE:-/tmp/ninfer-official-v3.pid}
logfile=${LOGFILE:-/tmp/ninfer-official-v3.log}

export NINFER_TEXT_RECIPE_FP8=1
export NINFER_VISION_RECIPE_QVIS=1
export LD_LIBRARY_PATH="/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/nccl-lib:${LD_LIBRARY_PATH:-}"

health_url="http://127.0.0.1:${port}/health"

health() {
  curl -s -o /dev/null --max-time 2 "${health_url}"
}

run() {
  local args=(
    "${artifact}"
    --host "${host}" --port "${port}"
    --model-id "${model_id}"
    --api-key "${api_key}"
    --tp "${tp}" --devices "${devices}"
    --max-context "${max_context}"
    --kv-capacity "${kv_capacity}"
    --kv-dtype "${kv_dtype}"
    --max-concurrency "${max_concurrency}"
    --prefill-chunk "${prefill_chunk}"
    --vision
    --media-cache-mib "${media_cache_mib}"
    --media-live-mib "${media_live_mib}"
  )
  if [[ "${prefix_cache}" == "1" ]]; then args+=(--prefix-cache); fi
  if [[ -n "${spec}" ]]; then args+=(--spec "${spec}" --draft-tokens "${draft_tokens}"); fi
  exec "${exe}" "${args[@]}" "$@"
}

start() {
  if health; then
    echo "already RUNNING on port ${port}"
    return 1
  fi
  [[ -x "${exe}" ]] || { echo "missing executable: ${exe}" >&2; return 1; }
  [[ -f "${artifact}" ]] || { echo "missing artifact: ${artifact}" >&2; return 1; }

  echo "starting official V3 (log: ${logfile}, port: ${port})..."
  setsid nohup "${BASH_SOURCE[0]}" __run > "${logfile}" 2>&1 < /dev/null &
  echo $! > "${pidfile}"

  for _ in $(seq 1 120); do
    sleep 2
    if health; then
      echo "health OK on port ${port}"
      curl -s --max-time 3 "${health_url}"; echo
      return 0
    fi
    if ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
      echo "server died; tail:"; tail -30 "${logfile}"; return 1
    fi
  done
  echo "not healthy after 240s; tail:"; tail -30 "${logfile}"; return 1
}

stop() {
  if [[ -f "${pidfile}" ]]; then
    local pid
    pid=$(cat "${pidfile}")
    if kill -0 "${pid}" 2>/dev/null; then
      echo "stopping ${pid}..."
      kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pidfile}"
  fi
  echo "stopped"
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status)
    if health; then echo "RUNNING (port ${port})"; curl -s --max-time 3 "${health_url}"; echo
    else echo "NOT running"; exit 1; fi ;;
  __run) shift; run "$@" ;;
  -h|--help) sed -n '1,40p' "${BASH_SOURCE[0]}"; ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 1 ;;
esac
