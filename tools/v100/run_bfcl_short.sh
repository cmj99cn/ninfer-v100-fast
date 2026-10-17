#!/usr/bin/env bash
# Short BFCL slice for quick iteration: multi_turn_base only.
# $1=model_id $2=api_url $3=api_key $4=tag [ $5=limit ]
set -euo pipefail
MODEL_ID="$1"; API_URL="$2"; API_KEY="$3"; TAG="$4"; LIMIT="${5:-20}"; TEMPERATURE="${TEMPERATURE:-1.0}"
WD="/home/cmj/eval-runs/bfcl/$TAG"
mkdir -p "$WD"
/home/cmj/eval-venv/bin/evalscope eval \
  --model "$MODEL_ID" \
  --api-url "$API_URL" \
  --api-key "$API_KEY" \
  --eval-type openai_api \
  --datasets bfcl_v3 \
  --dataset-args '{"bfcl_v3":{"subset_list":["multi_turn_base"],"extra_params":{"is_fc_model":true,"underscore_to_dot":true}}}' \
  --eval-batch-size 1 \
  --limit "$LIMIT" \
  --generation-config "{\"max_tokens\":8192,\"timeout\":240,\"stream\":true,\"temperature\":${TEMPERATURE},\"top_p\":1.0}" \
  --work-dir "$WD"
