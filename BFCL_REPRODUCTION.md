# BFCL 质量差距：复现与已采取措施 / BFCL Quality Gap: Reproduction and Measures Taken

本文记录 **截至 2026-10-23** 的完整复现方法、实验时间线、结果和踩过的坑。
This document records the full reproduction method, experiment timeline, results, and
detours as of **2026-10-23**.

问题说明 / Issue summary: [`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md)
状态 / Status: **未解决；主动修复已暂停 / Unresolved; active fixing paused**

> 不要把本文当作“已修复”的说明。它是一个可复现的负结果集合，用来让后来者少走弯路。
> This is not a “fixed” note. It is a reproducible set of negative results, intended to
> save others from repeating the same detours.

---

## 中文

### 1. 环境

- 硬件：2×Tesla V100-SXM2-32GB，NVLink；单机。
- 系统：Linux x86_64，glibc 2.39。
- ninfer：本仓库构建，`build/apps/ninfer-serve`。
- vLLM：1cat-vLLM 1.5.1，独立 venv，`FLASH_ATTN_V100`，`fp8_e5m2` KV。
- 评测：`evalscope`，BFCL-v3。
- 关键设置：thinking OFF；`temperature=1.0`；`top_p=1.0`；`top_k=20`；`max_tokens=8192`；
  `stream=true`；`--prefix-cache`；并发 2；`--max-context 262144`。

**不要用 `--limit 20` 下结论。** 它会把分数系统性抬高，并且掩盖 15pp 级差距。所有
结论都必须基于 200 条全量。

### 2. 制品、binary 与 hash

| 对象 | 路径 | SHA-256 |
|---|---|---|
| 本引擎当前 binary | `/home/cmj/ninfer-v100-fast/build/apps/ninfer-serve` | `10d1d99ad3d03882c43443b5e929c82c2da05c64f1fb29e4a58f2d1eef3be6fe` |
| 当前 upstream binary | `/home/cmj/ninfer-upstream/build/apps/ninfer-serve` | `9c203ccfb9429241c90b7d5ce13cf56ba912acd28a757a87ced5d5f9806d945c` |
| oldprompt 实验 binary | `/home/cmj/ninfer-upstream/build/apps/ninfer-serve-oldprompt` | `f33ca3d9564f3cf3109870a7dedc6952069baf1438c0d7982e3669ee06a1ee24` |
| official V3 `.ninfer` | `/home/cmj/models/ninfer-artifacts/qwen3_8_27b_nvfp4.ninfer` | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| our all-NVFP4 `.ninfer` | `/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer` | `8a130b51cdb5b5886f77ff5174963cbdc6aecc3c7229699d7c9a0828f6e58af6` |
| vLLM HF 权重目录 | `/home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4` | 5 个 safetensors shard；索引 `eee26dbb…` |

**关键区别**：vLLM 直接加载 `Qwen3.8-27B-QUASAR-NVFP4` 的 HF safetensors；
ninfer 加载的是从同一 `unsloth/Qwen3.8-27B-NVFP4` 源（revision
`60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`）转换出的 `.ninfer`。二者同源，但**不是
同一个运行时制品**。`our all-NVFP4` 的转换记录在
`qwen3_8_27b_nvfp4.ninfer.conversion.json`。

### 3. 最小复现：ninfer + evalscope

启动 ninfer（参数与质量 A/B 一致；`API_KEY` 用你自己的，不要提交密钥）：

```bash
export LD_LIBRARY_PATH="/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/nccl-lib:${LD_LIBRARY_PATH:-}"
export NINFER_VOLTA_SPLITD=1
export NINFER_SM70_PREFILL_GEMM=1

/home/cmj/ninfer-v100-fast/build/apps/ninfer-serve \
  /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 9010 \
  --tp 2 --devices 0,1 \
  --api-key "$NINFER_API_KEY" --model-id ninfer-nvfp4 \
  --kv-dtype int8 --max-context 262144 \
  --max-concurrency 2 --prefill-chunk 8192 \
  --prefix-cache --no-thinking \
  --presence-penalty 0.0 --seed 42
```

等 health 通过：

```bash
until curl -s -m 2 http://127.0.0.1:9010/health >/dev/null; do sleep 2; done
```

跑全量 200：

```bash
/home/cmj/eval-venv/bin/evalscope eval \
  --model ninfer-nvfp4 \
  --api-url http://127.0.0.1:9010/v1 \
  --api-key "$NINFER_API_KEY" \
  --eval-type openai_api --datasets bfcl_v3 \
  --dataset-args '{"bfcl_v3":{"subset_list":["multi_turn_base"],"extra_params":{"is_fc_model":true,"underscore_to_dot":true}}}' \
  --eval-batch-size 1 --limit 200 \
  --generation-config '{"max_tokens":8192,"timeout":240,"stream":true,"temperature":1.0,"top_p":1.0,"top_k":20}' \
  --work-dir /home/cmj/eval-runs/bfcl/ninfer_default_s42_n200
```

跑完看 review：

```bash
python3 - <<'PY'
import glob, json
p = sorted(glob.glob('/home/cmj/eval-runs/bfcl/ninfer_default_s42_n200/*/reviews/*/*.jsonl'))[-1]
rows = [json.loads(l) for l in open(p) if l.strip()]
ok = sum(int(r['sample_score']['score']['value'].get('acc', 0)) for r in rows)
print(f'{ok}/{len(rows)} ({100*ok/len(rows):.1f}%)')
PY
```

### 4. 最小复现：vLLM 对照

vLLM 必须用同一份源权重，不要用 `.ninfer`。启动时显式指定 `top_k`，否则会被模型的
`generation_config.json` 覆盖为 `top_k=20`：

```bash
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0
export VLLM_SM70_FP8_COORDINATED_TUNING=0
export PYTHONHASHSEED=0

/home/cmj/1cat-vllm-1.5.1/venv/bin/vllm serve \
  /home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --served-model-name qwen38-27b-nvfp4 \
  --trust-remote-code --dtype half \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 9000
```

评测命令同上，只改 `--model / --api-url / --api-key`，并在 generation config 里分别跑：

- `top_k=20`（vLLM 默认实际值）
- `top_k=0`（关闭截断，全词表）
- `top_k=20, top_p=0.95`（贴合模型默认）

实测前 50 条：

| vLLM 请求 | 通过 |
|---|---:|
| 默认（`top_k=20, top_p=0.95`） | 26/50 |
| 显式 `top_k=20, top_p=1.0` | 25/50 |
| 显式 `top_k=0, top_p=1.0` | 27/50 |

### 5. 实验时间线（含失败与弯路）

#### 2026-10-16：初始 80 条

- 4 个子集各 20 条，thinking 开：ninfer **18.75%** vs 1cat **38.75%**。
- 当时误以为这是稳定结论；后来发现小样本只能定性，不能做 6pp 归因。

#### 2026-10-19 前后：全量 200 与参数 A/B

- ninfer 默认：**20.5%（41/200）**。
- vLLM 同口径：**38.5%（77/200）**。
- 尝试 `--prefill-chunk 512/128`、bf16 KV：23–25%，与默认无显著差异。
- 结论：chunk 和 KV dtype 不是主因；`limit-20` 的“chunk 128=50%”是噪声。

#### prompt bug 与 `ce64d89a`

- 现象：ninfer 发给模型的多轮历史比 vLLM 多 150 tokens（5210 vs 5060）。
- 原因：OpenAI/Anthropic schema 用 `nlohmann::json`，默认按字母排序；并给每个 tool
  注入 `"strict": false`。
- 修复后：两侧 prompt **逐 token id 相同**（5060 tokens / 22746 B，尾部一致）。
- 教训：prompt 里有任何 JSON 序列化，都要做 token-level 对比，不要只比字符串长度。

#### 2026-10-22：旧 prompt 回退 A/B

目的：验证“旧 upstream 25% 是不是旧 prompt 序列化带来的”。

做法：

1. 从当前 upstream 树临时回退 5 个文件到 HEAD：
   `openai_schema.cpp`、`openai_schema.h`、`http_server.cpp`、
   `anthropic_schema.cpp`、`responses_schema.cpp`。
2. 保留 vision/QVIS 等其余未提交修改，保证 official V3 能加载。
3. 构建 `build/apps/ninfer-serve-oldprompt`，记录 hash。
4. 用同一个 official V3 artifact、seed 42/43/44、并发 2、前 50 条。
5. 恢复源码和当前 binary，校验 hash。

结果：

| seed | oldprompt | 当前 upstream |
|---|---:|---:|
| 42 | 17/50 | 19/50 |
| 43 | 14/50 | 18/50 |
| 44 | 13/50 | 21/50 |
| 合计 | **44/150（29.3%）** | **58/150（38.7%）** |

结论：旧 prompt 更差，prompt 回退不能恢复旧 25%。

**这里踩了一个大坑**：`pkill -x ninfer-serve` 对 `ninfer-serve-oldprompt` 无效，
因为 Linux `comm` 只保留 15 字符（`ninfer-serve-ol`）。结果旧 server 没退出，新
server 没绑上端口，seed43 第一次运行无效。后来改成：

```bash
pkill -9 -f 'build/apps/ninfer-serve'
```

并等待 `pgrep -f` 为空。**任何时候用 `pkill -x` 杀带后缀的 binary 都要小心。**

#### 2026-10-21/22：固定 seed 不可复现

- 同一 binary、同一 seed、同一请求序列，两次 run 在请求 **326/328** 分叉；
  同一请求的 generation 为 75 vs 49。
- 并发 1 也分叉，说明不是 batching 调度。
- 结论：引擎存在 run-to-run 数值/采样路径差异；**单 seed 的 6pp 差异不可信**。
- 后续所有关键结论都用 3 seeds 或全量配对。

#### 2026-10-22：parity 与 3 seeds

- 打开 `NINFER_QUALITY_PARITY=1`（关闭 fork-only Split-D、cuBLAS prefill GEMM、
  CUTLASS 输入投影、decode attention V2、强制 QPN）。
- 3 seeds 全 200：**14.0% / 10.5% / 11.0%，均值 11.8%（71/600）**。
- 旧单 seed 19% 未复现。
- 当前 upstream 前 50 条 3 seeds：**38.7%（58/150）**；fork 前 50：**38.0%（57/150）**。
- 结论：upstream vs fork 不是稳定差异；旧 25% 不能作为目标。

#### 2026-10-22/23：upstream 全量 `std::bad_alloc`

- 当前 upstream 在 200 run 的样本 **118**、100 run 的样本 **53** 附近
  `std::bad_alloc`。
- 因此 “upstream + our all-NVFP4” 的完整 200 格记为 **N/E（not evaluable）**，
  不要把它当“待跑”或“0 分”。
- 若要继续，需先修分配问题，或改成分段跑并明确标注不完整。

#### 2026-10-23 03:42–04:06：vLLM top_k A/B

- 发现 vLLM 启动日志：模型 `generation_config.json` 覆盖默认值为
  `temperature=1.0, top_k=20, top_p=0.95`。
- 显式 top_k=20：25/50；显式 top_k=0：27/50；默认：26/50。
- 结论：**top_k 不是 15pp 差距的原因**；vLLM 本来也不是全词表采样。

#### 2026-10-23 04:10：暂停

- prompt、top_k、chunk、KV、parity、parser 都排除后，没有低成本假设。
- 剩余首要嫌疑是 `.ninfer` 转换保真度；其次是 NVFP4/attention 数值和 prefix-cache
  状态。需要 logits 级证据，不能再盲扫参数。
- 决策：暂停主动修复，转功能；保留本文和失败样本。

### 6. 结果文件怎么找

每次评测在 `/home/cmj/eval-runs/bfcl/<TAG>/` 下：

- `reports/`：汇总 JSON。
- `predictions/`：逐样本输出（`messages` 里有完整多轮轨迹和 `perf_metrics`）。
- `reviews/`：评分，`sample_score.score.value.acc` 是 0/1。

建议保存：

- `reports/*.json`
- `reviews/*/*.jsonl`
- `predictions/*/*.jsonl`
- server log（含 prompt 长度、finish_reason、错误）
- binary/artifact hash

### 7. 不要重复的坑

1. **`--limit 20` 不能用于结论**。会高估且掩盖差距。
2. **`pkill -x ninfer-serve` 杀不掉带后缀的 binary**。用 `pkill -9 -f` 并等待。
3. **不要同时跑 vLLM 和 ninfer**。2×V100 显存不够，且会污染延迟/温度。
4. **不要用单 seed 做 6pp 归因**。固定 seed 仍会在 req326 分叉；至少 3 seeds。
5. **不要把 old 25% 当目标**。旧 binary 已覆盖，prompt A/B 已否定 prompt 原因。
6. **不要假设 `.ninfer` 与 HF safetensors 等价**。同源不等于同制品；official V3 与
   our all-NVFP4 相差 11pp 就是证据。
7. **不要只跑 parity 五开关**。前 50 条已基本持平；继续扫是浪费 GPU 窗口。
8. **不要在 200 条未跑完时报告百分比**。明确写 N/E 或“部分完成”。
9. **不要提交 API key**。这里的启动脚本用 `$API_KEY` 占位；生产环境用环境变量。
10. **不要在无人值守时启动超过窗口的任务**。全量 200 约 2h；留足清理时间。

### 8. 如果还要继续，最小下一步

不要先改 kernel。先做这三个证据采集：

1. 对 26 个 vLLM-only 样本，抓首轮 prefill 最后位置 top-20 logits，找第一分叉 token。
2. 同一请求开/关 `--prefix-cache`，比较第一分叉 token。
3. 在同一 engine 里分别加载 official V3、our all-NVFP4，比较同一层输出；若可行，
   再做 HF safetensors 直接加载路径。

只有证据指向某一层/某一 dtype 后，才值得改 kernel 或转换器。

---

## English

### 1. Environment

- Hardware: 2×Tesla V100-SXM2-32GB, NVLink, single node.
- OS: Linux x86_64, glibc 2.39.
- ninfer: this repository, `build/apps/ninfer-serve`.
- vLLM: 1cat-vLLM 1.5.1 in its own venv, `FLASH_ATTN_V100`, `fp8_e5m2` KV.
- Evaluation: `evalscope`, BFCL-v3.
- Key settings: thinking OFF; `temperature=1.0`; `top_p=1.0`; `top_k=20`;
  `max_tokens=8192`; `stream=true`; `--prefix-cache`; concurrency 2;
  `--max-context 262144`.

**Do not conclude from `--limit 20`.** It systematically overestimates and hides a
15pp-scale gap. All conclusions must use the full 200 samples.

### 2. Artifacts, binaries, hashes

| Object | Path | SHA-256 |
|---|---|---|
| Current ninfer binary | `/home/cmj/ninfer-v100-fast/build/apps/ninfer-serve` | `10d1d99ad3d03882c43443b5e929c82c2da05c64f1fb29e4a58f2d1eef3be6fe` |
| Current upstream binary | `/home/cmj/ninfer-upstream/build/apps/ninfer-serve` | `9c203ccfb9429241c90b7d5ce13cf56ba912acd28a757a87ced5d5f9806d945c` |
| oldprompt binary | `/home/cmj/ninfer-upstream/build/apps/ninfer-serve-oldprompt` | `f33ca3d9564f3cf3109870a7dedc6952069baf1438c0d7982e3669ee06a1ee24` |
| official V3 `.ninfer` | `/home/cmj/models/ninfer-artifacts/qwen3_8_27b_nvfp4.ninfer` | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| our all-NVFP4 `.ninfer` | `/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer` | `8a130b51cdb5b5886f77ff5174963cbdc6aecc3c7229699d7c9a0828f6e58af6` |
| vLLM HF weights | `/home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4` | 5 safetensors shards; index `eee26dbb…` |

**Key difference**: vLLM loads the HF safetensors in `Qwen3.8-27B-QUASAR-NVFP4`
directly; ninfer loads a `.ninfer` converted from the same
`unsloth/Qwen3.8-27B-NVFP4` source (revision
`60e813d4dbbdc5d64cf3f5a8caf2897bedf03679`). Same source, **different runtime
artifact**. The conversion record is in `qwen3_8_27b_nvfp4.ninfer.conversion.json`.

### 3. Minimal reproduction: ninfer + evalscope

Start ninfer (use your own API key; never commit it):

```bash
export LD_LIBRARY_PATH="/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/nccl-lib:${LD_LIBRARY_PATH:-}"
export NINFER_VOLTA_SPLITD=1
export NINFER_SM70_PREFILL_GEMM=1

/home/cmj/ninfer-v100-fast/build/apps/ninfer-serve \
  /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 9010 \
  --tp 2 --devices 0,1 \
  --api-key "$NINFER_API_KEY" --model-id ninfer-nvfp4 \
  --kv-dtype int8 --max-context 262144 \
  --max-concurrency 2 --prefill-chunk 8192 \
  --prefix-cache --no-thinking \
  --presence-penalty 0.0 --seed 42
```

Wait for health:

```bash
until curl -s -m 2 http://127.0.0.1:9010/health >/dev/null; do sleep 2; done
```

Run the full 200:

```bash
/home/cmj/eval-venv/bin/evalscope eval \
  --model ninfer-nvfp4 \
  --api-url http://127.0.0.1:9010/v1 \
  --api-key "$NINFER_API_KEY" \
  --eval-type openai_api --datasets bfcl_v3 \
  --dataset-args '{"bfcl_v3":{"subset_list":["multi_turn_base"],"extra_params":{"is_fc_model":true,"underscore_to_dot":true}}}' \
  --eval-batch-size 1 --limit 200 \
  --generation-config '{"max_tokens":8192,"timeout":240,"stream":true,"temperature":1.0,"top_p":1.0,"top_k":20}' \
  --work-dir /home/cmj/eval-runs/bfcl/ninfer_default_s42_n200
```

Read the review:

```bash
python3 - <<'PY'
import glob, json
p = sorted(glob.glob('/home/cmj/eval-runs/bfcl/ninfer_default_s42_n200/*/reviews/*/*.jsonl'))[-1]
rows = [json.loads(l) for l in open(p) if l.strip()]
ok = sum(int(r['sample_score']['score']['value'].get('acc', 0)) for r in rows)
print(f'{ok}/{len(rows)} ({100*ok/len(rows):.1f}%)')
PY
```

### 4. Minimal reproduction: vLLM baseline

vLLM must load the source HF weights, not `.ninfer`. Specify `top_k` explicitly;
otherwise the model `generation_config.json` overrides it with `top_k=20`:

```bash
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0
export VLLM_SM70_FP8_COORDINATED_TUNING=0
export PYTHONHASHSEED=0

/home/cmj/1cat-vllm-1.5.1/venv/bin/vllm serve \
  /home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4 \
  --served-model-name qwen38-27b-nvfp4 \
  --trust-remote-code --dtype half \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 9000
```

Use the same eval command with different model/URL/key, and run three generation configs:

- `top_k=20` (the model's effective default)
- `top_k=0` (disable truncation, full vocabulary)
- `top_k=20, top_p=0.95` (match the model defaults)

First 50 results:

| vLLM request | Pass |
|---|---:|
| default (`top_k=20, top_p=0.95`) | 26/50 |
| explicit `top_k=20, top_p=1.0` | 25/50 |
| explicit `top_k=0, top_p=1.0` | 27/50 |

### 5. Timeline, including failures and detours

#### 2026-10-16: initial 80 samples

- Four subsets × 20, thinking ON: ninfer **18.75%** vs 1cat **38.75%**.
- We initially treated this as stable; later learned small samples can only be
  qualitative, not 6pp attribution.

#### Around 2026-10-19: full 200 and parameter A/Bs

- ninfer default: **20.5% (41/200)**.
- vLLM same settings: **38.5% (77/200)**.
- `--prefill-chunk 512/128`, bf16 KV: 23–25%, no significant difference.
- Conclusion: chunk and KV dtype are not the cause; the “chunk 128 = 50%” result was
  limit-20 noise.

#### Prompt bug and `ce64d89a`

- Symptom: ninfer's multi-turn history was 150 tokens longer than vLLM's
  (5210 vs 5060).
- Cause: OpenAI/Anthropic schema used `nlohmann::json`, which sorts keys
  alphabetically, and injected `"strict": false` into every tool.
- After the fix: both sides produce **identical token IDs** (5060 tokens / 22746 B,
  identical tail).
- Lesson: any JSON serialization in the prompt needs token-level comparison, not just
  string length.

#### 2026-10-22: old-prompt rollback A/B

Goal: test whether the old upstream 25% came from old prompt serialization.

Procedure:

1. Revert five files in the current upstream tree to HEAD:
   `openai_schema.cpp`, `openai_schema.h`, `http_server.cpp`,
   `anthropic_schema.cpp`, `responses_schema.cpp`.
2. Keep the other local changes (vision/QVIS) so official V3 still loads.
3. Build `build/apps/ninfer-serve-oldprompt` and record its hash.
4. Use the same official V3 artifact, seeds 42/43/44, concurrency 2, first 50.
5. Restore the source and current binary; verify hashes.

Results:

| seed | oldprompt | current upstream |
|---|---:|---:|
| 42 | 17/50 | 19/50 |
| 43 | 14/50 | 18/50 |
| 44 | 13/50 | 21/50 |
| total | **44/150 (29.3%)** | **58/150 (38.7%)** |

Conclusion: the old prompt is worse; rolling it back cannot recover the old 25%.

**Major pitfall here**: `pkill -x ninfer-serve` does not match
`ninfer-serve-oldprompt`, because Linux `comm` is truncated to 15 characters
(`ninfer-serve-ol`). The old server stayed alive, the new one failed to bind, and the
first seed43 run was invalid. The fix:

```bash
pkill -9 -f 'build/apps/ninfer-serve'
```

and wait until `pgrep -f` is empty. Always be careful with `pkill -x` on suffixed
binaries.

#### 2026-10-21/22: fixed seed is not reproducible

- Same binary, same seed, same request sequence: two runs diverged at request
  **326/328**; generation lengths were 75 vs 49.
- Concurrency 1 also diverged, so it is not batching scheduling.
- Conclusion: run-to-run numerical/sampling-path variance exists; **single-seed 6pp
  differences are not trustworthy**.
- All key conclusions after this used 3 seeds or full paired comparisons.

#### 2026-10-22: parity and 3 seeds

- Enable `NINFER_QUALITY_PARITY=1` (disable fork-only Split-D, cuBLAS prefill GEMM,
  CUTLASS input projections, decode attention V2, forced QPN).
- Full 200 × 3 seeds: **14.0% / 10.5% / 11.0%, mean 11.8% (71/600)**.
- The old single-seed 19% did not reproduce.
- Current upstream first 50 × 3 seeds: **38.7% (58/150)**; fork first 50:
  **38.0% (57/150)**.
- Conclusion: upstream vs fork is not a stable difference; the old 25% is not a target.

#### 2026-10-22/23: upstream full-200 `std::bad_alloc`

- The current upstream died with `std::bad_alloc` around sample **118** in a 200 run and
  sample **53** in a 100 run.
- Therefore “upstream + our all-NVFP4” full 200 is **N/E (not evaluable)**, not “pending”
  and not “0”.
- To continue, fix the allocation issue or run in explicitly marked segments.

#### 2026-10-23 03:42–04:06: vLLM top_k A/B

- vLLM startup log revealed that the model `generation_config.json` overrides defaults
  to `temperature=1.0, top_k=20, top_p=0.95`.
- Explicit top_k=20: 25/50; explicit top_k=0: 27/50; default: 26/50.
- Conclusion: **top_k is not the cause of the 15pp gap**; vLLM was never doing
  full-vocabulary sampling by default.

#### 2026-10-23 04:10: paused

- After ruling out prompt, top_k, chunk, KV, parity, and parser, no low-cost hypothesis
  remained.
- Top remaining suspect: `.ninfer` conversion fidelity. Next: NVFP4/attention numerics
  and prefix-cache state. We need logits-level evidence, not more parameter sweeps.
- Decision: pause active fixing, shift to features, keep this document and the failing
  samples.

### 6. Where results live

Each eval writes to `/home/cmj/eval-runs/bfcl/<TAG>/`:

- `reports/`: summary JSON.
- `predictions/`: per-sample output (`messages` contains the full multi-turn trajectory
  and `perf_metrics`).
- `reviews/`: scores; `sample_score.score.value.acc` is 0/1.

Keep:

- `reports/*.json`
- `reviews/*/*.jsonl`
- `predictions/*/*.jsonl`
- server log (prompt length, finish_reason, errors)
- binary/artifact hashes

### 7. Detours not to repeat

1. **Never conclude from `--limit 20`.** It overestimates and hides the gap.
2. **`pkill -x ninfer-serve` does not kill suffixed binaries.** Use `pkill -9 -f` and wait.
3. **Do not run vLLM and ninfer at the same time.** 2×V100 memory is insufficient and it
   pollutes latency/temperature.
4. **Do not attribute 6pp from one seed.** Fixed seeds still diverge at req326; use at
   least 3 seeds.
5. **Do not target the old 25%.** Its binary is gone and prompt A/B ruled out prompt
   rollback.
6. **Do not assume `.ninfer` and HF safetensors are equivalent.** Same source is not the
   same artifact; official V3 vs our all-NVFP4 differs by 11pp.
7. **Do not run only the five parity switches.** The first 50 are already flat; more
   sweeps waste GPU windows.
8. **Do not report percentages from an incomplete 200.** Write N/E or “partial”.
9. **Do not commit API keys.** The commands use `$API_KEY` placeholders.
10. **Do not exceed the unattended GPU window.** A full 200 takes about 2 hours; leave
    cleanup time.

### 8. If you continue: smallest next steps

Do not patch kernels first. Collect these three pieces of evidence:

1. For the 26 vLLM-only samples, dump top-20 logits/probs at the last prefill position
   and find the first divergent token.
2. Run the same request with `--prefix-cache` on/off and compare the first divergent
   token.
3. Load official V3 and our all-NVFP4 in the same engine and compare same-layer outputs;
   if possible, add a direct HF-safetensors path.

Only after the evidence points to a layer or dtype should you change a kernel or the
converter.
