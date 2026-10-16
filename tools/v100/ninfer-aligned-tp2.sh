#!/usr/bin/env bash
# Aligned TP2 profile for the A/B vs 1cat (mirror of start_ninfer_aligned.sh):
# TP2, max-context 262144, max-concurrency 2, prefill-chunk 8192,
# prefix-cache ON, kv int8, no MTP.
# NINFER_SM70_PREFILL_GEMM: 0=CUTLASS, 1=cuBLAS. Default 1: A/B on Qwen3.8-27B-NVFP4
# TP2 cold prefill (7419/14800/59026) showed cuBLAS +20~23% prefill tps over CUTLASS.
set -euo pipefail
export LD_LIBRARY_PATH="/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/nccl-lib:${LD_LIBRARY_PATH:-}"
export NINFER_VOLTA_SPLITD=1
export NINFER_SM70_PREFILL_GEMM="${NINFER_SM70_PREFILL_GEMM:-1}"
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PORT="${PORT:-9010}"
exec "${repo_dir}/build/apps/ninfer-serve" \
  "${NINFER_ARTIFACT:-/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer}" \
  --host 0.0.0.0 --port "$PORT" \
  --tp 2 --devices 0,1 \
  --api-key ninfer-eval-key \
  --model-id ninfer-nvfp4 \
  --kv-dtype int8 \
  --max-context 262144 \
  --max-concurrency 2 \
  --prefill-chunk 8192 \
  --prefix-cache \
  --no-thinking --greedy
