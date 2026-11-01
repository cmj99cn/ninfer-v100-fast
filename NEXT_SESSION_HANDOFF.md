# NInfer 新会话交接文档

> 最后更新：2026-11-01
> 仓库：`/home/cmj/ninfer-v100-fast`
> 分支：`main`
> HEAD：`78cc79d6`（`docs(readme): restore project links in acknowledgements`）
> 工作区：**73 项未提交改动/未跟踪文件**，没有 commit。
> 产品状态：BF16 文本路径可用；FP16 数学层已完成，完整 FP16 文本 runtime 未接入。

---

## 0. 一句话结论

不要在当前 BF16 runtime 上继续零散接 FP16。FP16 attention/GDN/conv 等叶子已经有完整实现和 oracle 测试，但文本 runtime 的 workspace、residual、三个 leaf family、FP8 head、sampling 仍全部按 BF16 定义。下一步必须做一次完整的 runtime dtype 重写，或者明确停止 FP16 产品化、保留现有 BF16 产品。

---

## 1. 当前目标

用户要求的是：

- 真正全 FP16 文本推理：FP16 activation/residual/workspace；
- FP16 attention、FP16 GDN、FP16 conv；
- 层间不允许 `FP16→BF16→FP16` 边界转换；
- 用 BF16 原版和 1cat-vLLM 做端到端对比。

当前结论：

- FP16 数学组件已齐全；
- 完整 FP16 Engine 路径尚未实现；
- 产品仍走 BF16；
- 不能把当前半成品接进产品。

---

## 2. 机器、artifact 与关键路径

| 项 | 值 |
|---|---|
| GPU | 2× Tesla V100-SXM2 32GB，NVLink |
| 构建工具 | `/home/cmj/sglang-v100-venv/bin/ninja` |
| CUDA | 12.8，`sm_70` |
| 产品 artifact | `/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer` |
| 量化源 | `/home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4` |
| legacy Q4_K_M | `/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer` |
| BFCL 结果根目录 | `/home/cmj/eval-runs/bfcl/` |
| 诊断/临时证据 | `/home/cmj/tmp_layer_diag/` |
| GPU 当前 | 空闲，两张卡各约 4 MiB；没有 server 在跑 |

当前产品契约见 `AGENTS.md`：

- 当前 release target：2×32GB、NVFP4、262144 context、INT8 group-64 KV、MTP-3、eager decode；
- 4×16GB / Q4_K_M / 180000 / TP4 是 legacy regression；
- 不要把 legacy 目标当成当前验收门槛。

---

## 3. 本次会话已经完成的内容

### 3.1 FP16 数学层

以下均已有代码和测试：

- FP16 cast
- FP16 RMSNorm
- FP16 gated RMSNorm
- FP16 SiLU / silu_and_mul
- FP16 residual add
- FP16 allreduce
- FP16 embedding（FP8 table → FP16 直出）
- FP16 NVFP4 GEMM
- FP16 dense attention
- FP16 paged attention + cached attention
- FP16 KV append
- FP16 GDN recurrent
- FP16 RoPE
- FP16 GDN gating projection
- FP16 causal conv1d

测试名称：

```text
ninfer_gqa_attention_f16_dense_test
ninfer_gqa_attention_f16_paged_test
ninfer_gqa_attention_f16_cache_test
ninfer_gdn_f16_test
ninfer_rope_f16_test
ninfer_gqa_kv_append_f16_test
ninfer_gdn_gating_proj_f16_test
ninfer_causal_conv1d_f16_test
ninfer_linear_nvfp4_f16_test
```

测试结果摘要：

```text
gqa_attention_f16 cached prefill T=64 rel≈6.1e-5
gqa_attention_f16 cached decode N=512 rel≈2.8e-4
gdn_f16 T=1/4 out_rel≈3.6e-4/4.2e-4, state_rel≈2.2e-4/2.7e-4
rope_f16 T=1 rel=0, T=32 rel≈1.6e-4
gqa_kv_append_f16 T=1/32/64 exact
gdn_gating_proj_f16 rel≈1.1e-6
causal_conv1d_silu_f16 T=1/5/7 rel≈1.8e-4~2.0e-4, state_fail=0
nvfp4_cutlass_sm70_f16 rel L2≈1.4e-4~2.9e-4
```

### 3.2 conv state 布局修正

发现并修复了一个真实的布局错误：

- 原 `LinearAttentionStatePool` conv 布局是 `[C,3,S]`；
- conv kernel 自己按 `[3,C]` 索引，语义不一致；
- 现已改为 `[3,C,S]`，`conv_slot` 返回 `[3,C]`；
- `src/ops/wrapper/gdn_input_proj.cpp` 的 snapshot/record 校验已同步改为 `[3,C,S]`；
- `causal_conv1d_silu` wrapper 已同步；
- `src/ops/linear_attention/gated_delta_net/replay.cpp` 的 replay 校验已同步改为 `[3,C,S]`；
- 相关测试张量形状已同步；
- `state_store`、`causal_conv1d_silu`、`gdn_input_proj_conv_snapshot`、`gdn_input_proj_conv_record`、`gdn_replay_fold`、`mtp_split` 均通过。

证据：

- `ninfer_state_store_test` 通过；
- `ninfer_causal_conv1d_silu_test` 通过；
- BF16 模型可以启动；
- #117 输出 `40 3172 1438 440`，与修复前一致。

BFCL 影响：

- 修复后跑到 160/200，`54/160 = 33.8%`；
- baseline 前 160 条也是 `54/160`；
- 逐条通过数完全相同，说明 conv 布局修复是正确性修复，但不是 11.5pp 质量差距的原因。

### 3.3 诊断 CLI / 输入

保留的未提交诊断改动：

- `--print-prompt-token-ids`
- `--top-logits`
- `--preserve-thinking`
- `prompt_input.cpp` 使用 `ordered_json`

这些不是产品功能，交接时应决定是否提交或清理。

---

## 4. BFCL 质量证据

| 配置 | 结果 |
|---|---|
| NInfer BF16 baseline | `54/200 = 27.0%` |
| NInfer `--preserve-thinking` | `57/200 = 28.5%` |
| NInfer BF16 embedding/head | `56/200 = 28.0%` |
| NInfer greedy | `57/200 = 28.5%` |
| 1cat-vLLM | `77/200 = 38.5%` |
| layer0 FP16 MLP partial | `55/200 = 27.5%`（已移除） |

结论：

- 不是简单的 endpoint 转换；
- 不是单层 MLP 转换；
- 剩余差距主要是 BF16 vs FP16 数值路径差异，以及低 margin 决策；
- conv state 修复也不是主因。

相关结果目录：

```text
/home/cmj/eval-runs/bfcl/ninfer_sm70fix_s42_n200/20261026_094605
/home/cmj/eval-runs/bfcl/1cat_full/20261018_214318
/home/cmj/eval-runs/bfcl/ninfer_preserve_s42_n200/20261031_174521
/home/cmj/eval-runs/bfcl/ninfer_bf16ep_s42_n200/20261031_201424
/home/cmj/eval-runs/bfcl/ninfer_greedy_s42_n200/20261101_061933
/home/cmj/eval-runs/bfcl/ninfer_convfix_s42_n200（未跑完，160/200 后停止）
```

`--top-logits` 目前只抓 prefill bonus token，不抓 decode 每轮 logits。要分析逐轮决策，需要扩展它。

---

## 5. 为什么还没有“全 FP16 产品路径”

剩余阻塞不是某个 Op，而是 runtime 契约：

- `src/targets/qwen3_6/impl/runtime/workspace_recipe.h` 全部按 `DType::BF16` 分配；
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` 中约 279 处 `DType::BF16`；
- attention、GDN、post-mixer 三个 leaf family 的接口全部是 BF16；
- `Variant` 没有文本 dtype 参数；
- FP8 output head 仍是 BF16 输出；
- sampling/logits 仍按 BF16；
- KV/conv state 的 dtype 需要贯穿 request plan、graph frontier、workspace layout。

如果只替换部分 leaf，就会在层间产生 `FP16→BF16→FP16`，已被用户明确否定。因此要么完整重写，要么不接。

---

## 6. 构建与测试

构建：

```bash
/home/cmj/sglang-v100-venv/bin/ninja -C build ninfer -j 8
```

只构建某个测试：

```bash
/home/cmj/sglang-v100-venv/bin/ninja -C build ninfer_causal_conv1d_f16_test -j 8
./build/tests/ninfer_causal_conv1d_f16_test
```

常用测试：

```bash
./build/tests/ninfer_state_store_test
./build/tests/ninfer_causal_conv1d_silu_test
./build/tests/ninfer_gdn_input_proj_conv_snapshot_test
./build/tests/ninfer_gdn_input_proj_conv_record_test
./build/tests/ninfer_gdn_replay_fold_test
./build/tests/ninfer_mtp_split_test
./build/tests/ninfer_causal_conv1d_f16_test
./build/tests/ninfer_gdn_f16_test
./build/tests/ninfer_gqa_attention_f16_cache_test
./build/tests/ninfer_gqa_attention_f16_paged_test
./build/tests/ninfer_gqa_kv_append_f16_test
./build/tests/ninfer_rope_f16_test
./build/tests/ninfer_gdn_gating_proj_f16_test
./build/tests/ninfer_linear_nvfp4_f16_test
```

BF16 模型快速 smoke：

```bash
./build/apps/ninfer \
  /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --messages /home/cmj/tmp_layer_diag/bfcl117_tools_turn0.json \
  --no-thinking --greedy --max-new 4 \
  --tp 1 --device 0 --max-context 5120 --kv-dtype bf16 \
  --prefill-chunk 8192 --no-cuda-graph --print-token-ids
```

期望生成 token：

```text
40 3172 1438 440
```

BFCL 全量启动参数记录在 `BFCL_REPRODUCTION.md`。当前没有 server 在运行。

---

## 7. 未提交文件

`git status --short` 共 73 项。主要分三类：

1. 已有文件的 FP16/诊断改动（`M`）
2. 新增 FP16 实现与测试（`??`）
3. 1cat 适配代码（`third_party/1cat_flash_attention/`）

关键新增：

```text
include/ninfer/ops/causal_conv1d_silu_f16.h
include/ninfer/ops/gdn_gating_proj_f16.h
include/ninfer/ops/gqa_attention_f16.h
src/ops/attention/
src/ops/gdn_gating_proj/fp16/
src/ops/launcher/fp16/
third_party/1cat_flash_attention/
tests/ops/linear/test_nvfp4_f16.cpp
tests/ops/test_causal_conv1d_f16.cpp
tests/ops/test_gdn_f16.cpp
tests/ops/test_gdn_gating_proj_f16.cpp
tests/ops/test_gqa_attention_f16_cache.cpp
tests/ops/test_gqa_attention_f16_dense.cpp
tests/ops/test_gqa_attention_f16_paged.cpp
tests/ops/test_gqa_kv_append_f16.cpp
tests/ops/test_rope_f16.cpp
```

关键修改：

```text
src/core/linear_attention_state.cpp
src/ops/linear_attention/gated_delta_net/replay.cpp
src/ops/wrapper/gdn_input_proj.cpp
src/ops/wrapper/causal_conv1d_silu.cpp
src/ops/kernel/{cast,embed_gather,residual_add,rmsnorm,rope,silu_and_mul}.cuh
src/ops/launcher/{cast,embed_gather,residual_add,rmsnorm,rope,silu_and_mul}.{cu,h}
src/ops/linear/linear.cpp
src/ops/linear/nvfp4/
include/ninfer/types.h
src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h
src/targets/qwen3_6/impl/runtime/program_impl.h
apps/cli/
tests/
```

未跟踪的 `docs/superpowers/` 是旧过程文件，不应盲目提交。

---

## 8. 新会话建议执行顺序

### 方案 A：真正完成全 FP16 runtime（推荐，如果用户坚持）

1. 先决定当前 73 项改动是提交、stash 还是保留工作区。不要丢失 FP16 实现。
2. 新建专用分支，不直接改产品 main。
3. 在 `Variant` / `TextConfig` 中引入编译期文本 dtype，而不是运行时字符串分支。
4. 先改 `workspace_recipe` 和 layout simulation，让 residual/workspace 全部使用 FP16。
5. 按 leaf family 逐个替换：
   - attention projection / attention result / output projection
   - GDN input projection / conv / recurrent / gating / normalized output
   - post-mixer
6. 每个 family 替换后立刻用 BF16 原版做数值对比；不允许中间边界 cast。
7. 最后处理 FP8 output head 和 sampling/logits。
8. 跑 `ninfer` smoke、单测、BFCL 200；只有端到端结果支持时才声称完成。

### 方案 B：停止 FP16 产品化

保留当前 BF16 产品；把 FP16 实现作为独立分支/实验代码；清理诊断 CLI 和过期文档。不要再接半成品路径。

---

## 9. 已知坑

- `gdn_input_proj` 的 conv snapshot/record 形状必须与 state pool 一致：`[3,C,S]`。
- conv kernel 按 `[3,C]` 访问 state，不要改回 `[C,3]`。
- `KvCacheStorage::Float16` 枚举已存在，但尚未暴露 CLI，也未接模型。
- `ops::linear_add_row_parallel` 的 FP16 路径尚未实现；全 FP16 时不要直接复用 BF16 融合 down projection。
- FP8 output head 仍是 BF16 输出；全 FP16 不能只改 embedding。
- `--top-logits` 只抓 prefill bonus token，decode 逐轮 logits 需要另加。
- 当前 2×32GB 目标不要用 4×16GB legacy 的 180000 capacity 做验收。
- 构建 CPU 并发不要超过用户约 85% 上限，使用 `-j 8`。
- 不要提交当前工作区，除非用户明确要求。

---

## 10. 可直接粘贴给新会话的提示词

```text
你接手 /home/cmj/ninfer-v100-fast，先完整阅读 NEXT_SESSION_HANDOFF.md、AGENTS.md、
BFCL_QUALITY_ISSUE.md、BFCL_REPRODUCTION.md，以及 docs/maintainer/ 中与当前任务相关的文档。
不要读取旧对话，不要重复已经做过的 BFCL 诊断。

当前事实：
- 分支 main，HEAD 78cc79d6；工作区有 73 项未提交的 FP16/诊断改动，尚未 commit。
- 产品仍是 BF16；FP16 数学层（attention/GDN/conv/RoPE/KV append/NVFP4 GEMM 等）已完成并有测试。
- conv state 已修正为 [3,C,S]，BF16 模型可运行；BFCL 修复前后逐条结果相同，不是质量差距主因。
- 剩余差距是 BF16 vs FP16 数值路径；完整 FP16 文本 runtime 尚未接入。
- 禁止层间 FP16→BF16→FP16 边界转换。
- 目标硬件：2×V100-SXM2-32GB，NVFP4 artifact，262144 context，INT8 group-64 KV，MTP-3，eager decode。
- 不要用 4×16GB legacy 目标做验收。
- 不要用卡片提问；自行评估并选择技术上最优的方案，直接执行。

任务：
在保留现有 BF16 产品可用的前提下，设计并实现全 FP16 文本 runtime。
先给出一个最小但完整的实现边界和分阶段顺序，然后直接开始：
1. 处理当前未提交改动（提交/分支/stash 中选最安全的一种，不要丢代码）。
2. 引入编译期文本 dtype，改 workspace_recipe 和 layout simulation。
3. 按 attention、GDN、post-mixer 三个 leaf family 逐个替换，禁止中间 dtype cast。
4. 处理 FP8 output head 和 sampling/logits。
5. 每个阶段用独立 oracle/BF16 对比验证；最后跑 BF16 smoke、FP16 smoke、BFCL 200。
6. 只有端到端证据支持时才声称完成；失败就明确记录负结果。

先报告你确认的当前状态和第一阶段改动范围，然后开始改代码。不要重复询问用户。
```
