# Qwen3.8-Flash-Next M3：MTP-3 投机解码实现计划

> **面向 AI 代理的工作者：** 必需子技能：subagent-driven-development（推荐）或 executing-plans。步骤使用复选框（`- [ ]`）。

**目标：** 在现有 TP2 48 层路径上接入 MTP-3 投机解码，使 committed 吞吐按接受草稿数放大约 3×，且**贪心模式下提交的 token 序列与非 MTP 路径逐 token 相同**。

**架构：** 工件自带一个完整的 MTP 层（`mtp/layers/0/*`，含独立专家库 `weight/000096/000097`），加 `fc_embedding`/`fc_hidden` 融合投影、`pre_fc_*_norm` 与 `final_hc/*`。草稿链：以上一步的隐藏态 + 下一 token 的嵌入作输入，MTP 层预测 $\hat t_{n+1}$；链式最多 3 步。验证：把 3 个草稿拼到目标模型的当前上下文后做一次 T=4 的验证前向（复用 prefill 路径），按贪心逐位比较接受最长前缀，提交接受的 token，并把 KV/GDN/QSA/PLE 状态回滚到接受位置。

**技术栈：** C++20、CUDA 12.8、sm_70、FP16 激活（`NINFER_TEXT_DTYPE=f16`）、NVFP4 专家、TP2、eager decode。

**规格：** `docs/maintainer/qwen3.8-flash-next-artifact.md`（§1–§13，含 §13.1 实测表）、`.superpowers/sdd/2026-11-14-flash-next-m4-tp2-full-model/progress.md`（M4/M5 全部裁决）、`docs/maintainer/concurrent-inference-architecture.md`（`RoundBudget`/decode 批语义）。

## 全局约束

- 硬件：2×Tesla V100-SXM2-32GB（每卡 32 GiB）、NV6 互连、CUDA 12.8、NCCL 2.21.5+cuda12.4。
- 工件：`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`（102 GB，只读；缺失返回 77）。
- **数值判据（硬性）**：贪心 MTP 的提交序列必须与非 MTP 贪心序列**逐 token 相同**；被拒绝草稿的状态必须完全丢弃（KV/GDN/QSA tail/PLE history），不得残留。
- **逐位约束的延续**（M5 沿用）：改动不得改变既有非 MTP 路径的数值；`engine_route` golden `4818 84666 96991` 不变；`full_model` 两次逐位。
- 内部实现文件不写解释性注释（公开头接口契约注释除外）；C++20；无第三方测试框架；`src/targets/qwen3_6*` 一字不改；家族内无 target-dependent 分支；不做运行期权重搬运/通用抽象/占位。
- 构建：`/home/cmj/strata/.venv/bin/ninja -C build -j 8`；测试前
  `export LD_LIBRARY_PATH=/home/cmj/nccl-lib/whl/nvidia/nccl/lib:/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/cuda128/targets/x86_64-linux/lib:$LD_LIBRARY_PATH`。

## 起点实测（M5 结束时，未开 MTP）

| 口径 | ms/token | tok/s |
|---|---:|---:|
| 512 占用，0-miss | 31.0 | 32.2 |
| 512 占用，warm | 42–62 | 16–23 |
| 59K 占用 | 62.0 | 16.1 |
| prefill（512 chunk / 59K） | ≈5.05 s / 381 s | ≈101 / 131–146 |

每卡每 token 约 5.1 GB 权重；带宽下限 7–9.5 ms/token（105–140 tok/s）。已知剩余热点（M5-8 未做）：
router 读 2.6 MB 用 3.8 ms；routed 专家 15% 峰值；每层小算子合计约 14 ms。

## 审查重点

- **接受/拒绝的状态回滚**：拒绝的草稿必须在 KV、GDN recurrent、QSA tail/records、PLE history 里都不留痕；回滚后重跑必须逐位等于非投机路径。
- **提交序列的贪心等价**：MTP 不得改变采样分布或 tie-break；贪心必须逐 token 相同。
- **MTP 层的专家缓存**：MTP 层有自己的 512 专家库（约 2 GB），必须计入容量与 miss 统计，不能挤掉主模型。
- **验证前向的 KV 分配**：T=4 的验证要多占 3 个 KV 位置；容量边界（262144 档）与 32K 常驻窗口必须拒绝而不是静默越界。
- **不引入第二套 trunk**：MTP 层应复用家族的层执行器（同一 `run_gdn/run_qsa/run_ple/run_moe` 路径），只是层参数不同；不得复制 48 层的主干逻辑。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/targets/qwen3_8_flash_next/impl/load/mtp_bindings.{h,cpp}` | MTP 角色的绑定映射与装载期几何（独立专家库、共享 embedding/head） |
| `src/targets/qwen3_8_flash_next/impl/variant.h` | MTP 层视图与 `MtpConfig`（草稿窗口、层定义） |
| `src/targets/qwen4_exp/impl/runtime/speculative.*` | 草稿链、接受判定、提交/回滚事务（家族所有） |
| `src/targets/qwen4_exp/impl/runtime/text_context_impl.h` | 单层执行器复用（MTP 层走同一条层路径） |
| `src/targets/qwen4_exp/impl/runtime/state_transaction.*` | KV/GDN/QSA/PLE 的快照/回滚 |
| `src/targets/qwen4_exp/impl/runtime/program_impl.h` | `RoundBudget` 接线、metrics（acceptance） |
| `tests/targets/qwen3_8_flash_next/test_mtp_*.cpp` | 见各任务 |
| `tools/reference/qwen3_8_flash_next/mtp.py` | FP64 MTP oracle（草稿链 + 接受） |

---

### 任务 1：MTP 绑定与单步草稿

**文件：** 创建 `impl/load/mtp_bindings.{h,cpp}`、`impl/variant.h` 的 `MtpConfig`；测试 `test_mtp_bindings.cpp`、`test_mtp_draft.cpp`。

- [ ] **步骤 1：失败的测试**（真实工件，缺 77）：36 条 `mtp/*` 绑定全部解析；MTP 专家库 `matrices==512`、`expert_begin` 语义与任务 4 一致；`mtp/token_embedding`/`mtp/output_head` 与文本路径指向同一对象；`MtpConfig::draft_window == 3`（≤7 契约）。
- [ ] **步骤 2：RED**：`./build/tests/ninfer_flash_next_mtp_bindings_test` 编译失败。
- [ ] **步骤 3：实现绑定 + 单步草稿**：给定 `(hidden, next_token)`，跑一次 MTP 层（attention+MoE+HC，复用家族层执行器）→ 输出草稿 logits；贪心取 argmax。
- [ ] **步骤 4：oracle**：`tools/reference/qwen3_8_flash_next/mtp.py` 用 FP64 实现同一层；单步草稿 token 与 C++ 一致。
- [ ] **步骤 5：GREEN + 提交** `feat(flash-next): bind and run one mtp draft step`。

### 任务 2：草稿链（≤3）与贪心等价

**文件：** `src/targets/qwen4_exp/impl/runtime/speculative.{h,cpp}`；测试 `test_mtp_chain.cpp`。

- [ ] **步骤 1：失败的测试**：对同一 8-token 提示，`draft_chain(3)` 产出的 3 个草稿与 FP64 oracle 相同；当目标模型对草稿全部接受时，提交序列 == 非 MTP 贪心序列（逐 token）。
- [ ] **步骤 2：RED**。
- [ ] **步骤 3：实现链**：第 $i$ 步的输入是 $(h_i, \hat t_i)$，$h_i$ 来自上一步 MTP 层输出；不写任何持久状态。
- [ ] **步骤 4：GREEN + 提交** `feat(flash-next): draft a three-token mtp chain`。

### 任务 3：状态事务与回滚

**文件：** `state_transaction.{h,cpp}`（KV 页、GDN recurrent、QSA tail/records、PLE history）；测试 `test_mtp_state.cpp`。

- [ ] **步骤 1：失败的测试**：构造"接受 2 个 / 拒绝 1 个"的场景，回滚后状态哈希 == 非投机路径在 2 个 token 处的状态；再继续 8 步 decode，输出与非投机路径逐 token 相同。
- [ ] **步骤 2：RED**。
- [ ] **步骤 3：实现**：验证前向 T=4 后，记录各状态的"提交点"（按 token 数），回滚即截断到提交点；不复制整份状态（页表指针 + 长度即可）。
- [ ] **步骤 4：GREEN + 提交** `feat(flash-next): roll back rejected mtp drafts`。

### 任务 4：验证前向与提交策略

**文件：** `program_impl.h`（`RoundBudget` 接线）、`speculative.cpp`（接受判定）；测试 `test_mtp_verify.cpp`。

- [ ] **步骤 1：失败的测试**：贪心接受 = 最长公共前缀；T=4 验证的 logits 与单独 T=1 前向在同一 token 上逐位相同（验证前向不得改变数值）。
- [ ] **步骤 2：RED**。
- [ ] **步骤 3：实现**：一次 T=1+k 的验证前向复用 prefill 路径；接受 k′ 个后提交；`RoundBudget` 的预算/`finish` 语义与 27B 家族一致。
- [ ] **步骤 4：GREEN + 提交** `feat(flash-next): verify and commit mtp drafts in one round`。

### 任务 5：MTP 专家缓存与容量

**文件：** `expert_cache.*`/`expert_runtime.*`（MTP 库纳入同一容量策略）；测试 `test_mtp_cache.cpp`。

- [ ] **步骤 1：失败的测试**：MTP 专家库的 miss/hit 统计独立可观测；容量不足以容纳主模型 + MTP 时，`auto` 的覆盖率目标按两者合计的 routed mass 计算；显式容量过小时装载期拒绝。
- [ ] **步骤 2：RED**。
- [ ] **步骤 3：实现**：MTP 的 (layer=MTP, expert) 进入同一槽位表，`expert_begin` 区分来源。
- [ ] **步骤 4：GREEN + 提交** `feat(flash-next): cache the mtp expert bank`。

### 任务 6：端到端吞吐与质量

**文件：** `tools/bench/run_flash_next_bench.py`（`--mtp` 模式）、`docs/maintainer/qwen3.8-flash-next-artifact.md` §13.2；测试 `test_mtp_throughput.py`。

- [ ] **步骤 1：失败的测试**：`--mtp 3` 的 committed 序列 == 非 MTP 贪心序列（逐 token，512 与 59K 各一次）；acceptance 统计正确。
- [ ] **步骤 2：RED**。
- [ ] **步骤 3：实现测量**：512/3.7K/15K/59K 各测 committed tok/s、TTFT、acceptance、miss；JSON 到 `profiles/bench/`。
- [ ] **步骤 4：GREEN + 更新 §13.2 + 提交** `bench(flash-next): measure mtp-3 committed throughput`。

---

## 自检记录

- **验收口径**：MTP 的核心正确性是"贪心等价"，不是"接近"；放弃该判据就没有质量保证。
- **不复制主干**：MTP 层必须是家族同一条层执行器的不同层参数（任务 1/3 的测试会暴露复制）。
- **容量**：MTP 专家库约 2 GB，必须与主模型共用同一预算模型（任务 5）。
- **依赖**：M5-8（router/专家路径/小算子）与本计划独立；它决定 MTP 的底座，先做哪个由控制者按当时 GPU 可用性决定。
- **风险**：T=4 验证前向在 262144 容量边界上可能越界（审查重点第 4 条），任务 4 的测试必须覆盖。
