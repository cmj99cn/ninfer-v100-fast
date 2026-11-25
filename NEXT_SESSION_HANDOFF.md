# NInfer Flash-Next 交接文档

> 最后更新：2026-11-25
> 分支：`feat/qwen3-8-flash-next`（基于 main v0.2.5，27B NVFP4 发布）
> HEAD：`b0b79c4b`（`perf(flash-next): wire MTP fold commit and optimize verify with batched small-T kernels`），本轮性能工作已提交；本文档改动未提交、未推送（由控制者统一提交）
> 工作区：本轮只改文档，无代码/测试改动。
> 产品状态：Flash-Next target 开发中，非发布产品；主模型保持 TP2，未改并行架构。MTP 已快过 base 1.56×，但 120 tok/s 目标仍未达成。

---

## 0. 一句话结论

Flash-Next 在 2×V100 TP2 / f16 上已经端到端跑通（greedy 11751）。本轮 MTP 接线与优化（commit
`b0b79c4b`：GDN replay fold commit、verify 多列权重复用 kernel、draft TP2 角色拆分、Q4
proposal/head 默认、draft 置信度 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` 提前终止、sink 流同步）后，
最终配置实测 **110.3–110.5 tok/s**、acceptance **0.860–0.861**，base 70.54 → **1.56×**；
起点 30.33 tok/s（fold commit 接线前）到本轮 110，约 **3.6×**。**120 tok/s 目标仍未达成**
（1.56×，非 2×），但「MTP 落后 base / 负收益」的旧结论已失效。120 / 99% 属随机采样口径，
greedy 下不可达（见 4.1）；用户已裁决接受 Q4 为默认（draft 是提案、正确性由 verify 保证），
greedy 等价逐 token 一致。结论与下一步见第 6 节。

---

## 1. 分支 / 环境 / artifact

| 项 | 值 |
|---|---|
| 分支 | `feat/qwen3-8-flash-next`（基于 main v0.2.5，27B NVFP4 发布） |
| HEAD | `b0b79c4b`（本轮性能工作已提交，未推送，由控制者统一提交） |
| 硬件 | 2× Tesla V100-SXM2-32GB，TP2 |
| 工具链 | CUDA 12.8，`sm_70`，f16 activation（`NINFER_TEXT_DTYPE=f16`） |
| build | `/home/cmj/strata/.venv/bin/ninja -C build` |
| ctest | `/home/cmj/strata/.venv/bin/ctest` |
| Flash-Next artifact | `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` |
| MTP hidden fixture | `.superpowers/sdd/2026-11-16-flash-next-m3-mtp/assets/mtp_hidden_8tok.f16` |
| 专家画像 | `tools/bench/expert-profile-flash-next.txt`（转换自 `/home/cmj/strata/data/expert-profile.bin`，同一模型 48 层 × 512 专家，按热度排序的 24576 个 (layer, expert) 对） |
| SDD 账本 | `.superpowers/sdd/2026-11-19-performance-review/progress.md`（本轮全部测量、负结果与决策） |

定位说明：本分支是 Flash-Next target 开发，不是发布产品。发布 identity 仍是 `qwen3.8-27b/nvfp4`；Flash-Next 主模型保持 TP2，未改并行架构，层分 / TP1 / PP2 等方案只做过测量并回退。

---

## 2. 当前状态

### 2.1 性能（128 decode tokens，2 reps，cache 15000 MiB + 专家画像，0 miss，greedy）

最终配置：Q4 proposal/head 默认 + draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + 专家画像 0 miss。

| 配置 | committed tok/s | acceptance | 状态 |
|---|---:|---:|---|
| base，无 spec | **70.54** | — | 参照基线 |
| 起点（fold commit 接线前） | 30.33 | — | 本轮起点；MTP 尚未接线 |
| MTP 最终配置（Q4 + draft=3 + min_p=0.8） | **110.3–110.5** | 0.860–0.861 | 3 次实测 110.39 / 110.53 / 110.28；**1.56× base** |
| MTP draft=1，顺序 T=1 verify | 51.78 | 0.881 | 历史最佳（旧配置），当时比 base 慢 27%，已失效 |
| MTP draft=3，顺序 T=1 verify（min_p=0） | 42.9 | 0.772 | 历史记录；加 min_p=0.8 与 Q4 后提升 |
| MTP batch verify（`decode_span`） | 27.0 | 0.56 | 数值不等价，已回退 |

- 无专家画像时 MTP 只有 13.6–14 tok/s（hit rate 0.85，约 15k misses / 128 tokens）；画像解决冷缓存后 hit rate 1.0、0 miss，是早期关键一步。
- 120 tok/s 目标**仍未达成**（1.56×，非 2×）；但「当前没有配置能让 MTP 快过 base」的旧结论已被推翻。
- 起点 30.33 → 本轮 110，约 **3.6×**；51.78 只是旧配置的最佳值，现已被最终配置取代。

### 2.2 验证

- `ctest -R "flash_next|mtp|moe_nvfp4"`：**38/38 通过**（历史全量记录）。本轮最终门禁：构建 OK，
  `mtp_verify` / `fold_commit` / `mtp_state` / `gdn_replay_storage` 退出 0；`draft` / `chain` /
  `confidence` / `greedy_equivalence` 因缺隐藏 fixture `NINFER_MTP_HIDDEN` 退出 77（SKIP，内核 oracle 仍运行）；`git diff --check` 干净。
- NLL：`ninfer_flash_next_text_nll --tokens .../assets/upstream/capital5.ids` → `greedy_first 11751`、`greedy_next 11751`、`mean_nll 3.7311021849`；与记录的 BF16 基线一致（Δ=0.0，gate Δ≤0.01）。
- `mtp_verify` parity、`mtp_state` bit-exact、MoE T=4 oracle：全绿。
- Q4 默认（用户裁决）验证：draft 是提案、正确性由 verify 保证；chain oracle 由 `{220,27967,813}` 改为 `{220,27967,53}`（Q4 语义），greedy 等价（MTP on/off）逐 token 一致；`NINFER_DIAG_FP16_DRAFT_HEAD=1` 保留作对照并恢复 `{220,27967,813}`。

---

## 3. 已完成工作

1. Flash-Next package 注册、artifact binder、TP2 full model 端到端可用，Text/MTP 走同一 Engine 路线；greedy 输出与 BF16 基线一致。
2. f16 activation 路径 + 一组已归因的 decode 优化（two-byte GEMV、f16 head/PLE、HC fused decode、shared expert + router、W8 K=3072、GDN qkvz fusion、expert route history/prefetch），逐组 reverse-ablation 均为 load-bearing，详见 SDD 账本。
3. MTP 三个关键改动：draft bank 量化常驻（5.03 GB → 2.52 GB，每步 host sync 24 → 1，启动可用显存 13.59 → 17.36 GB）；`StateTransaction` 改为按 rank 的常驻设备缓冲 + D2D snapshot/restore（9.45 → 22.02 tok/s）；全接受时跳过重复 replay（`e0b1a77c`）。
4. cached routed MoE 扩展到 T≤4，并新增独立 oracle 的 T=2/T=4 测试；`MoE T=4 oracle` 通过。
5. 专家画像接线：`--expert-cache-profile` 进入 EngineOptions，`ExpertCache::seed()` 按 `expert_begin` 把 global expert 过滤到 TP2 本 rank 的 local shard。这是早期最大单项收益（14 → 51.78 tok/s），现已是最终配置的组成部分。
6. 定位并修复 GDN qkvz 的 T=1 / T>1 归约顺序差异：把 `w8_gdn_qkvz_decode` 扩展到 T=1..4 且保持同一累加顺序，`span_diag` 的 winner mismatch 从 1/5 降到 0/5；该 kernel 作为正确组件保留。
7. 本轮 MTP 接线与优化（commit `b0b79c4b`，起点 30.33 tok/s）：GDN replay fold commit（Tasks 1–10，bit-exact，现为生产路径）；verify 多列权重复用 kernel（hc_mix / linear / W8 / route / conv / gdn_proj / GDN stage+extract）；draft TP2 角色拆分；Q4 proposal/head 默认（artifact 自带同源权重）；draft 置信度 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` 提前终止；sink 流同步。最终 110.3–110.5 tok/s、1.56× base。

---

## 4. 关键结论

1. **专家画像解决了 cache 冷启动**：相同模型的热专家常驻（无淘汰）把 14 tok/s 提到 51.78 tok/s、0 miss；这是早期一步，本轮在它之上达到 110.3–110.5 tok/s。TP2 本身不是瓶颈。
2. **「MTP 负收益的根因是 verify 经济」已被本轮推翻**：旧记录认为每轮 2 次顺序 T=1 verify 只 commit 约 1.44 token、必然输给 base；GDN replay fold commit + verify 多列权重复用 kernel 大幅降低每轮 verify 成本后，最终 **1.56× 快过 base**。该结论仅作历史保留。
3. **batch verify 要赢必须 bit-exact**（仍成立）：greedy 语义下接受判定是严格 argmax 比较；通用 `forward(T)` 的 T>1 与 T=1 数值不等价（GDN conv/recurrent + PLE history）。已定位并修复 GDN qkvz 归约顺序，但 PLE 等仍未完全等价，`decode_span` 实验 acceptance 仍停在 0.56、已回退。
4. **概率 rejection sampling 不解决问题**：greedy 语义下它在数学上退化为严格比较（vLLM `temp==0` 分支可证），不能替代 bit-exact。
5. **对标事实**：1cat-vllm 用 TP2 + vLLM spec 框架、Strata 用层分 PP + 定制 `verify_kernels.cu`，两者 MTP 均 >100 tok/s；本分支现已达到 ~110 tok/s，与 Strata 同档，硬件/TP2 不是理由，差距在 verify 实现——现已补齐。

### 4.1 上游 Neroued/ninfer 对照（决定性发现，2026-11-23 查证）

证据：上游 <https://github.com/Neroued/ninfer>，README 与 `docs/performance/qwen3.8-flash-next.md`（同一模型 Qwen3.8-Flash-Next）。

1. **head 不是 acceptance 差异来源**：上游 optimized proposal head（`proposal/head q4_g64_fp16`）与主 head 同源 `lm_head.weight`；本地 conversion report 证明 `weight/001310` 与 `weight/000526` 均标注 `source base-hf-bf16:lm_head.weight`。我们的 `MtpDraft` 用未量化主 head，数学等价/更好，因此量化 proposal head 不能解释 acceptance 差距。
2. **99% acceptance 只在随机采样饱和口径**：上游 99.1–99.6% acceptance 仅在 decode saturation 场景（随机采样 temp 0.6 / top-p 0.95 / top-k 20 + 130-token structured JSON）。上游单请求 greedy 口径为 Structured 91%、Code 84%、Translation 72%、AIME 50%、Story 35%。
3. **我们的 0.88 与上游同口径一致**：bench 是 greedy（temp 0）+ 严格 argmax 前缀比较，此前 0.881 acceptance 对应上游 greedy Structured 91%；Q4 默认后最终配置 acceptance 为 **0.860**（FP16 对照 0.870），仍在同一 greedy 区间。随机采样 acceptance = 1 − TV(p,q) ≈ 1，greedy acceptance = P(argmax 相等)，后者天然更低；两者不可直接比较。
4. **120 tok/s 目标口径错位**：该数字来自 27B greedy 104.7；上游 Flash-Next 单请求 greedy 为 66.8 single Spark / 107.2 dual Spark（structured），>100 属随机采样口径。greedy 契约下 99% 不可达；**120 tok/s 仍未达成**（1.56×，非 2×）。「MTP 不超 base（51.78 < 70.54）是预期行为」的旧判断已被本轮实测推翻（70.54 → 110.4，1.56×），保留该历史数字仅为记录。
5. **对下一步的影响**：以 bit-exact small-T verify kernel 冲 120 的必要性被推翻；用户已裁决接受 Q4 为默认（见第 6 节）。本节不撤销第 4 节中仍成立的技术结论（bit-exact、rejection sampling、对标事实），也不改变已记录的负结果。

---

## 5. 负结果与回退（全部已回退，明细见 SDD 账本）

- **CUDA Graph 对 MTP 无效**：51.85 vs 51.78 tok/s（base 同样 62.5 vs 63.6 eager），整轮图 A/B −4.4%。**根因已定位**（见 `analysis-graph-negative-rootcause.md`）：sm_70 上 `cudaGraphLaunch` 的 O(N) 逐节点 CPU 提交成本（3,054 节点 / 2 ranks → ~0.6 ms/round）超过它省下的、已被设备执行掩盖的 host 间隙；Strata 同为 V100 却能用 graph，靠的是 kernel 内 t 维权复用（eager 侧属性），该收益我们的多列 kernel 已在 eager 侧领取。负收益结论保留，原因已查明——不是 kernel 更慢，也不是测量口径不公。
- **GDN replay fold 链路**：旧记录为「完整 36 层 record + fold 实现编译通过，但 11.36 tok/s < 14 tok/s，已回退（仅保留 36 层 geometry 注册）」——**该记录已过时**：fold commit 现已是生产路径（Tasks 1–10），bit-exact 验证通过，是本轮 1.56× 的关键组成。仅作历史保留。
- **batch verify 数值不等价**：生产 verify 切到 `decode_span` 后 acceptance 0.88 → 0.56、27.0 tok/s；qkvz 修复后 argmax 5/5 对齐但状态摘要仍不同，不能上线。
- **层分 / 去 allreduce**：`NINFER_SKIP_ALLREDUCE` 只带来 +3.5%，不足以支撑架构重写；PP2 会把每个 token 的两半层串行化，反而翻倍延迟。TP1/PP2/层分均不是路径。
- **其他已回退项**：staged routed GEMV（isolated 254 GB/s 但端到端 62.4–62.7，寄存器 50→87 导致 occupancy 下降）；partial-accept checkpoint 去 replay（acceptance 0.44 → 0.30–0.32）；FP8 MTP 路径（acceptance 0.0）；per-step checkpoint + route history（18.88 tok/s、acceptance 0.323）；深度 staging / probe-split（噪声内无收益）。
- **长生成 thrash**：无画像时 128 token 进入 miss 累积区（hit 0.849），画像已把 0 miss 保持到 128 token；不要用短窗口 64-token 的数字代替长生成结论。

---

## 6. 结论与下一步（Q4 默认已获用户裁决）

用户裁决：接受 Q4 proposal/head 为默认——draft 是提案、正确性由 verify 保证；chain oracle 由
`{220,27967,813}` 改为 `{220,27967,53}`（Q4 语义），`NINFER_DIAG_FP16_DRAFT_HEAD=1` 保留作对照并
恢复 FP16 oracle；greedy 等价（MTP on/off）逐 token 一致。最终配置（Q4 + draft=3 +
`NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + 专家画像 0 miss，TP2 / greedy / 128 token / 2 reps）三次实测
**110.3–110.5 tok/s**、acceptance **0.860–0.861**，base 70.54 → **1.56×**，起点 30.33 → 约 **3.6×**。

1. **120 tok/s 仍未达成**（1.56×，非 2×，gate 141）；剩余差距来自 HBM 地板 + 单 head 顺序 draft。
   draft expert 7-kernel 合并、跨轮 gating 等为增量杠杆，要到 2× 需结构上更便宜的 draft head 或
   多 head/并行 draft artifact。「MTP 落后 base」的旧结论已失效，不再作为 Gate。
2. **口径澄清保留**：120 / 99% 来自随机采样饱和口径，greedy 下不可达；本轮 1.56× 是 greedy
   committed tok/s，与上游 greedy 口径可比，不与随机采样 saturation 数字混用。
3. 负结果历史（第 5 节）保留：CUDA Graph 负收益已定位根因；GDN replay fold 已翻案为生产路径；
   batch verify `decode_span` 数值不等价仍成立，不再以它冲 120；层分 / 并行架构未改（主模型 TP2）。
4. 若后续要随机采样性能，需引入 stochastic sampling（如 temp 0.6 / top-p 0.95 / top-k 20）与
   对应的 speculative accept（1 − TV(p,q) 或等价概率接受），并按该口径重定 tok/s 与 acceptance
   目标（参考上游饱和场景 99.1–99.6%）；greedy 的严格 argmax 前缀比较不再是该口径的验收指标。

---

## 7. 常用命令

先设置环境（每个新 shell 执行一次）：

```bash
cd /home/cmj/ninfer-v100-fast/.worktrees/flash-next
export NINFER_FLASH_NEXT_ARTIFACT=/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer
export NINFER_MTP_HIDDEN=/home/cmj/ninfer-v100-fast/.worktrees/flash-next/.superpowers/sdd/2026-11-16-flash-next-m3-mtp/assets/mtp_hidden_8tok.f16
export LD_LIBRARY_PATH=/home/cmj/nccl-lib/whl/nvidia/nccl/lib:/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/cuda128/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export FLASH_NEXT_PROFILE=tools/bench/expert-profile-flash-next.txt
export NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8
```

构建（按需重建目标；本次交接不运行）：

```bash
/home/cmj/strata/.venv/bin/ninja -C build -j8 \
  ninfer_flash_next_engine_bench ninfer_flash_next_text_nll \
  ninfer_flash_next_mtp_verify_test ninfer_flash_next_mtp_state_test \
  ninfer_moe_nvfp4_test
```

bench — base 无 spec（70.54 tok/s 参照；起点为 30.33 tok/s）：

```bash
./build/tests/ninfer_flash_next_engine_bench --bench \
  --artifact "$NINFER_FLASH_NEXT_ARTIFACT" \
  --cases 512 --decode-tokens 128 --warmup 64 --reps 2 \
  --prefill-chunk 1024 --expert-cache-mib 15000 \
  --expert-cache-profile "$FLASH_NEXT_PROFILE"
```

bench — 最终 MTP 配置（Q4 默认 + draft=3 + min_p=0.8，110.3–110.5 tok/s，0.860–0.861）：

```bash
./build/tests/ninfer_flash_next_engine_bench --bench \
  --artifact "$NINFER_FLASH_NEXT_ARTIFACT" \
  --cases 512 --decode-tokens 128 --warmup 64 --reps 2 \
  --prefill-chunk 1024 --expert-cache-mib 15000 \
  --expert-cache-profile "$FLASH_NEXT_PROFILE" \
  --spec mtp --draft-tokens 3
```

`NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` 由上面的环境导出提供；去掉该变量即旧口径（历史 draft=1
51.78 / 0.881、draft=3 42.9 / 0.772 均为 min_p=0 记录）。Q4 对照加
`NINFER_DIAG_FP16_DRAFT_HEAD=1`（恢复到 FP16 oracle `{220,27967,813}`）。CUDA Graph 对照加
`--graph`（已证明负收益，根因见第 5 节，仅用于复现负结果）。

ctest（38/38）：

```bash
/home/cmj/strata/.venv/bin/ctest --test-dir build \
  -R "flash_next|mtp|moe_nvfp4" -j1 --output-on-failure
```

NLL / 数值 gate：

```bash
./build/tests/ninfer_flash_next_text_nll \
  --artifact "$NINFER_FLASH_NEXT_ARTIFACT" \
  --tokens .superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/assets/upstream/capital5.ids
# 期望：greedy_first 11751, greedy_next 11751, mean_nll 3.7311021849
```

MTP span 差异诊断（T>1 vs 顺序 T=1，含 GDN conv/recurrent、PLE history 的 winner 与状态摘要）：

```bash
NINFER_MTP_SPAN_DIAG=1 ./build/tests/ninfer_flash_next_mtp_verify_test
```

单项 bit-exact / oracle：

```bash
./build/tests/ninfer_flash_next_mtp_state_test
./build/tests/ninfer_moe_nvfp4_test
```

所有测量与决策的账本在 `.superpowers/sdd/2026-11-19-performance-review/progress.md`；写入新结论时追加到该文件，不要新建平行的最终报告。

---

## 8. 可直接粘贴给新会话的提示词

```text
你接手 /home/cmj/ninfer-v100-fast/.worktrees/flash-next，先读 NEXT_SESSION_HANDOFF.md、
AGENTS.md 和 .superpowers/sdd/2026-11-19-performance-review/progress.md。
不要读取旧对话；不要重复已经回退的实验（CUDA Graph——负收益根因已定位、batch verify
decode_span、层分/去 allreduce、checkpoint 去 replay、staged routed GEMV、FP8 MTP）。
GDN replay fold 已翻案为生产路径，不再是回退项。

当前事实：
- 分支 feat/qwen3-8-flash-next，HEAD b0b79c4b（本轮性能工作已提交，未推送，由控制者
  统一提交）；本文档改动只改文档，不提交不推送。
- Flash-Next 是开发中 target，非发布产品；主模型 TP2，并行架构未改。
- 环境：2×V100-SXM2-32GB，TP2，sm_70，CUDA 12.8，f16 activation。
- 128-token / 2 reps / cache 15000 + 专家画像 / 0 miss / greedy，最终配置 =
  Q4 proposal/head 默认 + draft=3 + NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8：
  base 70.54 tok/s；MTP 110.3-110.5 tok/s、acceptance 0.860-0.861（3 次实测
  110.39/110.53/110.28），1.56× base；起点 30.33 tok/s（fold commit 接线前）
  到本轮 110，约 3.6×。无画像仅 13.6-14 tok/s。
- 历史配置（已被取代，保留记录）：draft=1 51.78 / 0.881（当时比 base 慢 27%）、
  draft=3 42.9 / 0.772、batch verify 27.0 / 0.56（数值不等价，已回退）。
- 120 tok/s 仍未达成（1.56×，非 2×）；「MTP 落后 base / 负收益」的旧结论已失效。
- 用户裁决：接受 Q4 proposal/head 为默认——draft 是提案、正确性由 verify 保证；
  chain oracle 由 {220,27967,813} 改为 {220,27967,53}（Q4 语义），
  NINFER_DIAG_FP16_DRAFT_HEAD=1 保留作对照并恢复 FP16 oracle；greedy 等价
  （MTP on/off）逐 token 一致。
- 上游 Neroued/ninfer 查证（README + docs/performance/qwen3.8-flash-next.md）：head 同源、
  非 acceptance 差异来源；99.1-99.6% 只在随机采样饱和口径，greedy 单请求 Structured 91%、
  Code 84%、Translation 72%、AIME 50%、Story 35%；我们 greedy 0.860 与上游同口径一致，
  greedy 契约下 99% 不可达；120/99% 口径错位（见 4.1）。
- 验证：构建 OK，git diff --check 干净；mtp_verify / fold_commit / mtp_state /
  gdn_replay_storage 通过；draft / chain / confidence / greedy_equivalence 因缺隐藏
  fixture SKIP（内核 oracle 仍运行）；历史 38/38 ctest；NLL greedy 11751、
  mean_nll 3.7311021849；mtp_verify parity、mtp_state bit-exact、MoE T=4 oracle 全绿。
- 画像已解决 cache 冷启动；「每轮 2 次顺序 verify 只 commit ~1.44 token 必然输给 base」
  的旧根因已被 fold commit + 多列权重复用 kernel 推翻。
- greedy 下 batch verify 必须 bit-exact；概率 rejection sampling 退化为严格比较，不能替代。
- GDN qkvz 归约顺序已修为 bit-exact，PLE history 等仍未完全等价，decode_span 仍 0.56。
- CUDA Graph 负收益根因：sm_70 上 cudaGraphLaunch 的 O(N) 节点提交（3054 节点 →
  ~0.6 ms/round）超过已掩盖的 host 间隙；Strata 同为 V100 能用 graph 靠 kernel 内
  t 权复用（eager 侧属性），我们已在 eager 侧领取该收益。

任务（用户已裁决 Q4 默认，无需再确认验收口径）：
1. 120 tok/s 仍未达成（2× gate = 141），剩余差距为 HBM 地板 + 单 head 顺序 draft；
   draft expert 7-kernel 合并、跨轮 gating 为增量杠杆，2× 需结构上更便宜的 draft head
   或多 head/并行 draft artifact。
2. 保持 greedy 验收（base 70.54 + 质量门禁），不与随机采样 saturation 数字混用；
   若另要随机采样性能，需引入 stochastic sampling + speculative accept（1-TV）并重定 Gate。
3. 负结果历史保留；不切生产 verify 到 decode_span batch、不动并行架构。

不要主动 commit 或推送（由控制者统一提交）。构建 CPU 并发不超过约 85% 上限。
```
