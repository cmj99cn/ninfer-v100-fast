# BFCL 多轮工具调用质量差距 / BFCL Multi-Turn Tool-Calling Quality Gap

状态 / Status: **未解决；暂停盲目主动修复，公开请求社区协作 / Unresolved; blind active fixing paused, community help requested**

最后更新 / Last updated: 2026-10-27

复现与实验过程 / Reproduction and experiment log: [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)

项目说明 / Project overview: [`README.md`](README.md) · 版本说明 / Release notes: [`CHANGELOG.md`](CHANGELOG.md)

---

## 中文

### 结论先说

在**同一份源 NVFP4 量化权重**、同一 BFCL-v3 子集、同一采样参数、同一 2×V100 硬件上：

- 1cat-vLLM：**38.5%（77/200）**
- 本引擎（ninfer）最新全量：**27.0%（54/200）**
- 差距：**11.5pp**

27.0% 是目前 ninfer 全量 200 的最高记录，但只比此前最佳 25.0% 高 2pp。逐样本与
c512 相比为 **+14 / -10，净增 4 条、24 条翻转**；单 seed 不能支撑 6pp 级归因。
因此这是“没有回归、略有改善”，不是“质量差距已解决”。

**这个问题已经投入大量时间，仍未解决。** 我们做过的 A/B 包括：全量 200 条多 seed、
prompt 回退与 tokenizer 对齐、top_k 全词表、upstream/fork parity、KV int8/bf16、
prefill chunk 128/512/8192、tool parser 严格性、official V3 与自转换 all-NVFP4
制品对比，以及 SM70 数值/测试修复。结论是这些都不是剩余差距的完整解释。

2026-10-23 起**暂停盲目的主动修复**，不再承诺短期解决；保留可复现步骤、负结果和
失败样本，避免后来者重复同样的弯路。现在**公开请求社区协作定位质量问题**。

### 最新全量结果（2026-10-26/27）

设置：BFCL-v3 `multi_turn_base` 全量 200 条，thinking OFF，
`temperature=1.0`、`top_p=1.0`、`top_k=20`、`seed=42`，TP2，NVFP4 制品，
`--prefix-cache` 开启。

| 指标 | 值 |
|---|---:|
| 通过 | **54/200 = 27.0%** |
| 1cat-vLLM 对照 | 77/200 = 38.5% |
| 实际请求数 | 2404 |
| 总耗时 | 5959 s（约 1 小时 39 分） |
| 平均延迟 | 2.471 s |
| 平均 TTFT | 804.9 ms |
| 平均 TPOT | 13.8 ms |
| 平均吞吐 | 38.91 tok/s |
| 平均输入 / 输出 | 5262 / 96 tokens |

逐样本：与 vLLM 两边都过 43 条，只有 vLLM 过 34 条，只有 ninfer 过 11 条。
与 c512 对比：两边都过 40 条，本次新过 14 条、新丢 10 条。

结果目录：`/home/cmj/eval-runs/bfcl/ninfer_sm70fix_s42_n200/20261026_094605`

### 当前数字

| 配置 | 分数 |
|---|---:|
| 1cat-vLLM（同源 NVFP4，fp8_e5m2 KV，FLASH_ATTN_V100） | **38.5%（77/200）** |
| ninfer 最新全量（SM70 修复后，seed 42） | **27.0%（54/200）** |
| ninfer 此前最佳（chunk 512） | 25.0%（50/200） |
| ninfer 默认（our all-NVFP4，int8 KV，chunk 8192） | 20.5%（41/200） |
| ninfer chunk 128 | 23.0%（46/200） |
| ninfer bf16 KV | 23.0%（46/200） |
| ninfer fork parity + our all-NVFP4 | 23.0%（46/200） |
| ninfer fork parity + official V3（3 seeds） | **11.8%（71/600）** |
| 当前 upstream + official V3（前 100，seed 42/43） | 20.0 / 19.0% |
| oldprompt + official V3（前 50，3 seeds） | 29.3%（44/150） |

同一引擎换制品就能移动 10pp 以上：official V3 **11.8%** vs our all-NVFP4 **23.0%**。
这说明**制品转换本身是主要变量之一**，不能把它当成“等价权重”。

### 我们为提升质量做过的工作

1. **Prompt 模板/tokenizer 对齐**：修复 `ce64d89a` 中 OpenAI/Anthropic schema 的键序
   和注入的 `"strict": false`。修复后 vLLM 与 ninfer 的 prompt **逐 token id 相同**
   （5060 tokens / 22746 B，尾部一致）。
2. **旧 prompt 回退**：把 5 个 prompt 文件回退到 HEAD 重建 binary，3 seeds 前 50 条
   只有 **29.3%**，低于当前 upstream 的 **38.7%**。旧 prompt 更差。
3. **采样对齐**：vLLM 模型的 `generation_config.json` 本身就是
   `top_k=20, top_p=0.95, temperature=1.0`；显式 top_k=20 为 **25/50**，显式
   top_k=0（全词表）为 **27/50**，默认 26/50。采样不是差距来源。
4. **prefill chunk / KV 量化**：chunk 128/512、bf16 KV 在全量 200 条下均无显著差异
   （p=0.12 / 0.38 / 0.40）。此前 limit-20 的“chunk 128=50%”是小样本噪声。
5. **upstream vs fork parity**：前 50 条 3 seeds，upstream **38.7%**，fork **38.0%**，
   没有可测优势；旧 upstream 25% 单点不可复现，旧 binary 已被覆盖。
6. **tool parser 严格性**：复核后不是主因；malformed tool call 离线恢复 0/9，不能解释
   主要 vLLM-only 样本。
7. **制品对比**：同一引擎加载 official V3 与 our all-NVFP4，分数相差 11pp 以上；
   自转换 `.ninfer` 涉及 embedding encoder、NVFP4 blockscale 布局、FP8 矩阵、
   BF16/FP16/FP32 norm/lm_head，以及 1268 个对象的 dtype/layout 重排。
8. **SM70 数值/测试修复**：修复 cuBLAS BF16 输出经 FP16 暂存的溢出（改为 FP32
   暂存），补齐 A4/A8 能力门控、Q4/Q5 workspace、零容量 arena 等；全量 ctest
   **135/135 通过**，15 个硬件/产物门控跳过。
9. **prefix cache / 多实例**：完成共享 host tier、跨实例 manifest、2×TP1 真实测试；
   快速 cold/warm 与 MTP 质量 A/B 未观察到回归。

### 已排除的方向

1. **Prompt 模板/tokenizer 不一致**：修复后 prompt 逐 token id 相同，不是模板差异。
2. **旧 prompt 回退**：旧 prompt 只有 29.3%，更差，不存在“回退 prompt 恢复旧 25%”的修复。
3. **top_k / 全词表采样**：vLLM 默认就是 top_k=20；top_k=0 与默认差异不能解释差距。
4. **prefill chunk / KV 量化**：chunk 128/512、bf16 KV 在全量 200 下均无显著差异。
5. **upstream vs fork**：当前 upstream 与 fork 前 50 条 3 seeds 基本持平。
6. **tool parser 严格性**：不是主要失败形态。
7. **固定 seed 可复现性**：并发 2 和并发 1 的重复 run 都在请求 326/328 分叉
   （generation 75 vs 49），说明存在 run-to-run 数值/采样路径差异；单 seed 结果
   不能用于 6pp 级归因。

### 仍未排除

按优先级：

1. **自转换 `.ninfer` 制品的转换保真度（首要嫌疑）**
   vLLM 直接加载 `Qwen3.8-27B-QUASAR-NVFP4` 的 HF safetensors；ninfer 先把它转成
   18 GiB `.ninfer`。转换涉及：
   - embedding encoder `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`
   - NVFP4 blockscale 布局、FP8 矩阵、BF16/FP16/FP32 norm/lm_head
   - 1268 个对象的 dtype/layout 重排
   这些都可能改变数值。official V3 与 our all-NVFP4 相差 11pp 已经证明“制品不是等价物”。

2. **NVFP4 GEMM / attention 数值路径**：即使制品相同，SM70 NVFP4 kernel、attention
   reduction 顺序、prefix-cache 下的状态也可能放大误差。但需要 logits 证据再改 kernel。

3. **Prefix cache 状态差异**：固定 seed 下仍分叉；需要对比开/关 prefix cache 的首个
   分叉 token。

4. **vLLM-only 样本的具体失败形态**：34 条只有 vLLM 通过的样本中，ninfer 是格式错、
   参数错、还是多轮状态错，尚未逐条归类。

### 公开请求社区协作

如果你能复现，请通过 GitHub Issue/Discussion 提交，并尽量附上以下证据：

1. **Prompt 对齐**：同一 BFCL 样本在 vLLM 与 ninfer 的完整 prompt token IDs。
2. **首分叉定位**：首轮 prefill 最后位置的 top-20 logits/probs，以及第一个分叉 token。
3. **Prefix cache A/B**：同一请求开/关 `--prefix-cache` 的第一分叉点。
4. **制品/层对比**：official V3、our all-NVFP4、以及可行的 HF safetensors 直接加载
   路径，在同一 engine 内的同层输出。
5. **失败分类**：逐样本标记格式错 / 函数名错 / 参数错 / 多轮状态错。
6. **复现信息**：硬件、CUDA/驱动、binary 与 artifact SHA-256、完整命令、结果目录。

请不要：

- 只报“BFCL 分数低”，不附样本、prompt token、第一分叉 token 或 logits；
- 用 `--limit 20` 下结论；它会系统性高估并掩盖差距，必须全量 200 条；
- 追旧 upstream 25% 单点；旧 binary 已覆盖，prompt A/B 已否定 prompt 原因；
- 只做 parity 五开关或 kernel sweep；前 50 条已基本持平；
- 假设 our all-NVFP4 与 vLLM 加载的 HF safetensors 等价。

本项目不承诺修复周期或 SLA；仓库所有者不具备模型/推理引擎技术背景，无法回答技术问题。
请把可复现证据写进 Issue/Discussion 或仓库文档。

### 复现入口

- 最小复现、vLLM 对照、完整实验时间线、踩过的坑：
  [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)
- 当前二进制、制品 hash、评测命令和结果目录：见复现文档第 2–4 节。
- 全量评测脚本与读分方法：见 [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md) 第 3–4 节。

---

## English

### Bottom line

On the **same source NVFP4 quantized weights**, the same BFCL-v3 subset, the same
sampling parameters, and the same 2×V100 hardware:

- 1cat-vLLM: **38.5% (77/200)**
- this engine (ninfer), latest full run: **27.0% (54/200)**
- gap: **11.5pp**

27.0% is the highest full-200 ninfer result so far, but only 2pp above the previous best
25.0%. Against c512 the per-sample result is **+14 / -10, net +4 with 24 flips**; a single
seed cannot support 6pp attribution. The result means “no regression, modest improvement”,
not “quality gap solved”.

**This issue has consumed a large amount of time and is still unresolved.** We ran full
200-sample multi-seed A/Bs, prompt rollback and tokenizer alignment, full-vocabulary top_k,
upstream/fork parity, KV int8/bf16, prefill chunk 128/512/8192, parser strictness,
official V3 vs our all-NVFP4 artifacts, and SM70 numerical/test fixes. None of them fully
explains the remaining gap.

Blind active fixing has been **paused since 2026-10-23**. We are not promising a short-term fix.
The reproducible steps, negative results, and failing samples are kept here so others do
not repeat the same detours. **We are now publicly asking the community to help localize
the quality gap.**

### Latest full result (2026-10-26/27)

Settings: BFCL-v3 `multi_turn_base`, full 200 samples, thinking OFF,
`temperature=1.0`, `top_p=1.0`, `top_k=20`, `seed=42`, TP2, NVFP4 artifact,
`--prefix-cache` enabled.

| Metric | Value |
|---|---:|
| Pass | **54/200 = 27.0%** |
| 1cat-vLLM baseline | 77/200 = 38.5% |
| Actual requests | 2404 |
| Wall time | 5959 s (~1 h 39 min) |
| Average latency | 2.471 s |
| Average TTFT | 804.9 ms |
| Average TPOT | 13.8 ms |
| Average throughput | 38.91 tok/s |
| Average input / output | 5262 / 96 tokens |

Per-sample: both pass 43, vLLM-only 34, ninfer-only 11. Against c512: both pass 40,
new passes 14, new losses 10.

Results: `/home/cmj/eval-runs/bfcl/ninfer_sm70fix_s42_n200/20261026_094605`

### Current numbers

| Configuration | Score |
|---|---:|
| 1cat-vLLM (same source NVFP4, fp8_e5m2 KV, FLASH_ATTN_V100) | **38.5% (77/200)** |
| ninfer latest full run (after SM70 fixes, seed 42) | **27.0% (54/200)** |
| ninfer previous best (chunk 512) | 25.0% (50/200) |
| ninfer default (our all-NVFP4, int8 KV, chunk 8192) | 20.5% (41/200) |
| ninfer chunk 128 | 23.0% (46/200) |
| ninfer bf16 KV | 23.0% (46/200) |
| ninfer fork parity + our all-NVFP4 | 23.0% (46/200) |
| ninfer fork parity + official V3 (3 seeds) | **11.8% (71/600)** |
| current upstream + official V3 (first 100, seeds 42/43) | 20.0 / 19.0% |
| oldprompt + official V3 (first 50, 3 seeds) | 29.3% (44/150) |

Changing only the artifact moves the same engine by more than 10pp: official V3
**11.8%** vs our all-NVFP4 **23.0%**. The converted artifact is therefore a major
variable, not an equivalent weight file.

### Work done to improve quality

1. **Prompt template/tokenizer alignment**: fixed the OpenAI/Anthropic schema key order
   and the injected `"strict": false` in `ce64d89a`. After the fix, vLLM and ninfer
   produce **identical token IDs** (5060 tokens / 22746 B, identical tail).
2. **Old prompt rollback**: reverting five prompt files to HEAD and rebuilding gave
   **29.3%** on the first 50 across 3 seeds, below the current upstream's **38.7%**.
3. **Sampling alignment**: vLLM's model `generation_config.json` already sets
   `top_k=20, top_p=0.95, temperature=1.0`. Explicit top_k=20 scored **25/50**;
   explicit top_k=0 scored **27/50**; default scored 26/50.
4. **prefill chunk / KV quantization**: chunk 128/512 and bf16 KV show no significant
   difference on the full 200 (p=0.12 / 0.38 / 0.40). The earlier “chunk 128 = 50%”
   was limit-20 noise.
5. **upstream vs fork parity**: first 50, 3 seeds: upstream **38.7%**, fork **38.0%**;
   no measurable advantage. The old upstream 25% single run is not reproducible, and its
   binary has been overwritten.
6. **tool parser strictness**: not the main cause; offline recovery of malformed calls
   was 0/9 and cannot explain the main vLLM-only samples.
7. **Artifact comparison**: official V3 vs our all-NVFP4 differs by more than 11pp in the
   same engine. The self-converted `.ninfer` changes the embedding encoder, NVFP4
   blockscale layouts, FP8 matrices, BF16/FP16/FP32 norms and lm_head, and the
   dtype/layout of 1268 objects.
8. **SM70 numerical/test fixes**: fixed cuBLAS BF16 output staged through FP16 (now FP32),
   added A4/A8 capability gating, Q4/Q5 workspace, zero-capacity arena fixes, etc.; full
   ctest **135/135 passed**, 15 hardware/artifact-gated skips.
9. **prefix cache / multi-instance**: shared host tier, cross-instance manifests, and
   2×TP1 real tests completed; quick cold/warm and MTP quality A/B showed no regression.

### Ruled out

1. **Prompt template/tokenizer mismatch**: after the fix, prompt token IDs are identical.
2. **Old prompt rollback**: old prompt scores only 29.3%, worse; there is no “roll back
   the prompt to recover the old 25%” fix.
3. **top_k / full-vocabulary sampling**: vLLM already uses top_k=20; top_k=0 vs default
   does not explain the gap.
4. **prefill chunk / KV quantization**: chunk 128/512 and bf16 KV show no significant
   difference on the full 200.
5. **upstream vs fork**: current upstream and fork are essentially flat on the first 50
   across 3 seeds.
6. **tool parser strictness**: not the main failure shape.
7. **Fixed-seed determinism**: both concurrency-2 and concurrency-1 duplicate runs
   diverged at request 326/328 (generation 75 vs 49). There is run-to-run
   numerical/sampling-path variance; single-seed results cannot support 6pp attribution.

### Still open

By priority:

1. **Conversion fidelity of the self-converted `.ninfer` artifact (top suspect)**
   vLLM loads the HF safetensors directly from `Qwen3.8-27B-QUASAR-NVFP4`; ninfer first
   converts them into an 18 GiB `.ninfer`. The conversion changes:
   - embedding encoder `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1`
   - NVFP4 blockscale layouts, FP8 matrices, BF16/FP16/FP32 norms and lm_head
   - dtype/layout of 1268 objects
   Any of these can change numerics. The 11pp difference between official V3 and our
   all-NVFP4 already proves the artifacts are not equivalent.
2. **NVFP4 GEMM / attention numerics**: even with the same artifact, SM70 NVFP4 kernels,
   attention reduction order, and prefix-cache state can amplify error. Do not patch
   kernels without logits evidence.
3. **Prefix-cache state**: fixed-seed runs still diverge; compare the first divergent
   token with prefix cache on/off.
4. **Failure shape of the 34 vLLM-only samples**: not yet classified into format error,
   wrong function, wrong arguments, or wrong multi-turn state.

### Public request for community help

If you can reproduce this, please open a GitHub Issue/Discussion with:

1. **Prompt alignment**: full prompt token IDs for the same BFCL sample from vLLM and ninfer.
2. **First divergence**: top-20 logits/probs at the last prefill position and the first
   divergent token.
3. **Prefix-cache A/B**: the first divergence with `--prefix-cache` on vs off.
4. **Artifact/layer comparison**: same-layer outputs for official V3, our all-NVFP4, and
   (if feasible) a direct HF-safetensors path in the same engine.
5. **Failure taxonomy**: per-sample format / function name / arguments / multi-turn state.
6. **Reproduction details**: hardware, CUDA/driver, binary and artifact SHA-256, exact
   commands, results directory.

Please do not:

- report only a low score without samples, prompt tokens, the first divergent token, or logits;
- use `--limit 20` for conclusions; it systematically overestimates and hides the gap;
  use the full 200;
- chase the old upstream 25% single point; the binary is gone and the prompt A/B already
  ruled out prompt rollback;
- run only parity switches or blind kernel sweeps; the first 50 are already flat;
- assume our all-NVFP4 and the HF safetensors loaded by vLLM are equivalent.

No fix timeline or SLA is promised. The repository owner has no model/inference-engine
technical background and cannot answer technical questions. Put reproducible evidence in
the Issue/Discussion or repository documents.

### Reproduction entry points

- Minimal reproduction, vLLM baseline, full timeline, pitfalls:
  [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)
- Current binary/artifact hashes, evaluation commands, and result directories: see
  sections 2–4 of the reproduction document.
- Full evaluation script and score reading: see sections 3–4 of
  [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md).
