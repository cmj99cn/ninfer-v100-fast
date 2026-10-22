# BFCL 多轮工具调用质量差距 / BFCL Multi-Turn Tool-Calling Quality Gap

状态 / Status: **未解决 / Unresolved**
最后更新 / Last updated: 2026-10-23
复现与实验过程 / Reproduction and experiment log: [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)

---

## 中文

### 结论先说

在**同一份源 NVFP4 量化权重**、同一 BFCL-v3 子集、同一采样参数、同一 2×V100 硬件上：

- 1cat-vLLM：**38.5%（77/200）**
- 本引擎（ninfer）：**20.5–23.0%（41–46/200）**
- 前 100 条逐样本：**41 条全部失败，26 条只有 vLLM 通过，4 条只有 upstream 通过**。

**这个问题已经投入大量时间，仍未解决。** 我们做过的 A/B 包括：全量 200 条多 seed、
prompt 回退、top_k 全词表、upstream/fork parity、KV int8/bf16、prefill chunk
128/512/8192、tool parser 严格性、固定 seed 复现。结论是这些都不是主因。

2026-10-23 起**暂停主动修复**，不再承诺短期解决；保留可复现步骤、负结果和失败样本，
避免后来者重复同样的弯路。

### 当前数字

| 配置 | 分数 |
|---|---:|
| 1cat-vLLM（同源 NVFP4，fp8_e5m2 KV，FLASH_ATTN_V100） | **38.5%（77/200）** |
| ninfer 默认（our all-NVFP4，int8 KV，chunk 8192） | **20.5%（41/200）** |
| ninfer chunk 512 | 25.0%（50/200） |
| ninfer chunk 128 | 23.0%（46/200） |
| ninfer bf16 KV | 23.0%（46/200） |
| ninfer fork parity + our all-NVFP4 | 23.0%（46/200） |
| ninfer fork parity + official V3（3 seeds） | **11.8%（71/600）** |
| 当前 upstream + official V3（前 100，seed42/43） | 20.0 / 19.0% |
| oldprompt + official V3（前 50，3 seeds） | 29.3%（44/150） |

同一引擎换制品就能移动 10pp 以上：official V3 **11.8%** vs our all-NVFP4 **23.0%**。
这说明**制品转换本身是主要变量之一**，不能把它当成“等价权重”。

### 已排除的方向

1. **Prompt 模板/tokenizer 不一致**：已修 `ce64d89a`。修复后 vLLM 与 ninfer 的
   prompt **逐 token id 相同**（5060 tokens / 22746 B，尾部一致）。不是模板差异。
2. **旧 prompt 回退**：把 5 个 prompt 文件回退到 HEAD 重建 binary，3 seeds 前 50 条
   只有 **29.3%**，低于当前 upstream 的 **38.7%**。旧 prompt 更差。
3. **top_k / 全词表采样**：vLLM 模型 `generation_config.json` 本身就是
   `top_k=20, top_p=0.95, temperature=1.0`；显式 top_k=20 为 **25/50**，显式
   top_k=0（全词表）为 **27/50**，默认 26/50。不是采样差异。
4. **prefill chunk / KV 量化**：chunk 128/512、bf16 KV 在全量 200 条下均无显著差异
   （p=0.12 / 0.38 / 0.40）。之前 limit-20 的“chunk 128=50%”是小样本噪声。
5. **upstream vs fork**：前 50 条 3 seeds，upstream **38.7%**，fork **38.0%**，没有
   可测优势；旧 upstream 25% 单点不可复现，旧 binary 已被覆盖。
6. **tool parser 严格性**：复核后不是主因；malformed tool call 离线恢复 0/9，不能解释
   26 个 vLLM-only 样本。
7. **固定 seed 可复现性**：并发 2 和并发 1 的重复 run 都在请求 326/328 分叉
   （generation 75 vs 49），说明存在 run-to-run 数值/采样路径差异，单 seed 结果
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

4. **vLLM-only 样本的具体失败形态**：26 条只有 vLLM 通过的样本中，ninfer 是格式错、
   参数错、还是多轮状态错，尚未逐条归类。

### 需要什么证据才能继续

- 同一 BFCL 首样本，vLLM 与 ninfer 的完整 prompt token IDs（已确认当前一致，需保留为回归）。
- 26 个 vLLM-only 样本的首轮 prefill 最后位置 top-20 logits/probs，定位第一个分叉 token。
- 同一引擎分别加载 official V3、our all-NVFP4、以及从 HF safetensors 直接 mmap 的
  路径（若可行），比较同一层输出。
- 开/关 prefix cache 的同一请求 logits 对比。
- 逐样本失败分类：格式错 / 函数名错 / 参数错 / 多轮状态错。

### 不要重复的弯路

- 不要用 `--limit 20` 下结论；它会系统性高估且掩盖差距。必须全量 200 条。
- 不要追旧 upstream 25% 单点；旧 binary 已覆盖，prompt A/B 已否定 prompt 原因。
- 不要只做 parity 五开关/kernel sweep；前 50 条已基本持平。
- 不要只报“BFCL 分数低”；必须附样本、prompt token、第一分叉 token 或 logits 证据。
- 不要假设 our all-NVFP4 与 vLLM 加载的 HF safetensors 等价；它们是同源、不同运行时制品。

---

## English

### Bottom line

On the **same source NVFP4 quantized weights**, the same BFCL-v3 subset, the same
sampling parameters, and the same 2×V100 hardware:

- 1cat-vLLM: **38.5% (77/200)**
- this engine (ninfer): **20.5–23.0% (41–46/200)**
- Per-sample on the first 100: **41 fail everywhere, 26 pass only with vLLM, 4 pass only
  with upstream**.

**This issue has consumed a large amount of time and is still unresolved.** We ran full
200-sample multi-seed A/Bs, prompt rollback, full-vocabulary top_k, upstream/fork parity,
KV int8/bf16, prefill chunk 128/512/8192, parser strictness, and fixed-seed determinism
checks. None of them explains the gap.

As of 2026-10-23, active fixing is **paused**. We are not promising a short-term fix.
The reproducible steps, negative results, and failing samples are kept here so others
do not repeat the same detours.

### Current numbers

| Configuration | Score |
|---|---:|
| 1cat-vLLM (same source NVFP4, fp8_e5m2 KV, FLASH_ATTN_V100) | **38.5% (77/200)** |
| ninfer default (our all-NVFP4, int8 KV, chunk 8192) | **20.5% (41/200)** |
| ninfer chunk 512 | 25.0% (50/200) |
| ninfer chunk 128 | 23.0% (46/200) |
| ninfer bf16 KV | 23.0% (46/200) |
| ninfer fork parity + our all-NVFP4 | 23.0% (46/200) |
| ninfer fork parity + official V3 (3 seeds) | **11.8% (71/600)** |
| current upstream + official V3 (first 100, seeds 42/43) | 20.0 / 19.0% |
| oldprompt + official V3 (first 50, 3 seeds) | 29.3% (44/150) |

Changing only the artifact moves the same engine by more than 10pp: official V3
**11.8%** vs our all-NVFP4 **23.0%**. The converted artifact is therefore a major
variable, not an equivalent weight file.

### Ruled out

1. **Prompt template/tokenizer mismatch**: fixed in `ce64d89a`. After the fix, vLLM and
   ninfer produce **identical token IDs** (5060 tokens / 22746 B, identical tail).
2. **Old prompt rollback**: reverting five prompt files to HEAD and rebuilding gave
   **29.3%** on the first 50 across 3 seeds, below the current upstream's **38.7%**.
3. **top_k / full-vocabulary sampling**: vLLM's model `generation_config.json` already
   sets `top_k=20, top_p=0.95, temperature=1.0`. Explicit top_k=20 scored **25/50**;
   explicit top_k=0 scored **27/50**; default scored 26/50.
4. **prefill chunk / KV quantization**: chunk 128/512 and bf16 KV show no significant
   difference on the full 200 (p=0.12 / 0.38 / 0.40). The earlier “chunk 128 = 50%”
   was limit-20 noise.
5. **upstream vs fork**: first 50, 3 seeds: upstream **38.7%**, fork **38.0%**; no
   measurable advantage. The old single-run upstream 25% is not reproducible, and its
   binary has been overwritten.
6. **tool parser strictness**: not the main cause; offline recovery of malformed calls
   was 0/9 and cannot explain 26 vLLM-only samples.
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
4. **Failure shape of the 26 vLLM-only samples**: not yet classified into format error,
   wrong function, wrong arguments, or wrong multi-turn state.

### Evidence needed to continue

- Full prompt token IDs for the same BFCL sample from vLLM and ninfer (currently known
  to match; keep as a regression check).
- Top-20 logits/probs at the last prefill position for the 26 vLLM-only samples, to find
  the first divergent token.
- Same-layer outputs for official V3, our all-NVFP4, and (if possible) a direct
  HF-safetensors path in the same engine.
- Same-request logits with prefix cache on and off.
- Per-sample failure taxonomy: format / function name / arguments / multi-turn state.

### Detours not to repeat

- Do not conclude from `--limit 20`; it systematically overestimates and hides the gap.
  Use the full 200.
- Do not chase the old upstream 25% single run; the binary is gone and the prompt A/B
  already ruled out prompt rollback.
- Do not run only the five parity switches or blind kernel sweeps; the first 50 are
  already flat.
- Do not report only “BFCL is low”; attach samples, prompt tokens, the first divergent
  token, or logits.
- Do not assume our all-NVFP4 and the HF safetensors loaded by vLLM are equivalent.
  They come from the same source but are different runtime artifacts.
