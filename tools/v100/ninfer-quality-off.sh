#!/usr/bin/env bash
# Quality A/B profile matched to the 1cat-vLLM BFCL run, thinking OFF.
#   * 1cat vLLM defaults to enable_thinking=false, so pass --no-thinking here.
#   * request sampling still applies (run_bfcl_t1.sh sends temperature=1.0).
# Everything else mirrors tools/v100/ninfer-quality-matched.sh.
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
  --kv-dtype "${KV_DTYPE:-int8}" \
  --max-context 262144 \
  --max-concurrency 2 \
  --prefill-chunk 8192 \
  ${PREFIX_FLAG:---prefix-cache} \
  --no-thinking \
  --presence-penalty 0.0 \
  --seed 42
