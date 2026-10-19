#!/usr/bin/env bash
# Quality A/B profile matched to the 1cat-vLLM BFCL run, thinking OFF.
#   * 1cat vLLM defaults to enable_thinking=false, so pass --no-thinking here.
#   * request sampling still applies (run_bfcl_t1.sh sends temperature=1.0).
# Everything else mirrors tools/v100/ninfer-quality-matched.sh.
set -euo pipefail
export LD_LIBRARY_PATH="/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/nccl-lib:${LD_LIBRARY_PATH:-}"
# NINFER_QUALITY_PARITY=1 disables every fork-only numeric route so the harness
# reproduces upstream 842078a1's kernel selection (Split-D attention, cuBLAS prefill
# GEMM, cutlass input projections and the decode attention V2 are all fork-only).
# Each of those changes the per-op result by ~1e-4 relative L2, which the model
# amplifies to 0.2-0.5 by the last layer (bench-results/ninfer-quality-investigation.md).
if [ "${NINFER_QUALITY_PARITY:-0}" = "1" ]; then
  export NINFER_VOLTA_SPLITD=0
  export NINFER_SM70_ATTN_V2=0
  export NINFER_SM70_PREFILL_GEMM=0
  export NINFER_VOLTA_CUTLASS_MIN_T=1000000
  export NINFER_NVFP4_FORCE_QPN=0
fi
export NINFER_VOLTA_SPLITD="${NINFER_VOLTA_SPLITD:-1}"
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
  --max-context "${MAXCTX:-262144}" \
  --max-concurrency 2 \
  --prefill-chunk "${PREFILL_CHUNK:-8192}" \
  ${PREFIX_FLAG:---prefix-cache} \
  --no-thinking \
  --presence-penalty 0.0 \
  --seed 42
