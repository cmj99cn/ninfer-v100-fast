# DFlash2 优化向 backlog（2026-11-27）

当前活跃优化目标。MTP 清单（`mtp-optimization-backlog.md`）已冻结为历史记录。
口径：2×V100-SXM2-32GB、TP2、`sm_70`、CUDA 12.8、greedy、eager；target
`Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` + companion
`Qwen3.8-Flash-Next-DFlash2-Infernix/...infernix`，`--expert-cache-mib 15000` +
`tools/bench/expert-profile-flash-next.txt`（0 miss）。

## 基线（已独立复现，HEAD 5ac8666c）

| 配置 | committed tok/s | acceptance | ×base |
|---|---:|---:|---:|
| base plain | 71.0 | — | 1.0 |
| MTP-3 | 112.2 | 0.859 | 1.58 |
| **DFlash2-4** | **118.9** | **0.937** | **1.67** |

目标 2×（142）还差 ~20%。方法沿用 MTP：轮次分相分解 → 定位主导相 → 受控根因实验 → 实现
（独立 oracle + 受控 A/B）→ 记录负结果。不做外推。

## D0 结果（2026-11-27，已独立复核）

512 prompt / 128 greedy / TP2 / cache 15000 + profile / min_p 0.8：

| K | tok/s | acceptance | ms/token（按 1+accepted/轮折算） |
|---:|---:|---:|---:|
| 4 | 118.8 | 0.936 | 8.35 |
| **5** | **138.4（reps2 复现 138.42）** | **0.989** | **7.17** |
| 6 | 135.7 | 0.989 | 7.33 |
| 7 | 133.9 | 0.982 | 7.42 |

- K=5 与 plain greedy **逐 token 相同**（lossless），两次进程确定性；K=5 在 64-token prompt 上也赢
  （116.4 vs 113.2），非单一 prompt 巧合。
- `min_p=0.9`（K=5）= 126.7 tok/s、acceptance 1.0——截断掉 draft 反而更慢，故保留 0.8。
- 结论：Flash-Next companion 在 V100X2 上的最优是 **K=5 + min_p=0.8**，≈1.95× base，逼近 2×。
  命令行默认仍保留 4（避免影响 27B DFlash2），文档推荐 Flash-Next 显式 `--draft-tokens 5`。

## D1 结果（轮次分相 + T=6 verify per-op，2026-11-27）

分相（K=5，761 轮稳态，5.668 token/轮；插桩对吞吐 −0.12%）：

| 相位 | ms/轮 | % | ms/token |
|---|---:|---:|---:|
| propose | 9.61 | 24.3 | 1.70 |
| verify（T=6 目标前向） | 28.73 | 72.8 | 5.07 |
| commit | 0.20 | 0.5 | 0.04 |
| materialize（特征→ring） | 0.88 | 2.2 | 0.16 |

T=6 verify per-op（两卡合计 ms/round，括号为 T6/T1 缩放）：

| op | ms | ×T1 | 备注 |
|---|---:|---:|---|
| routed | 14.99 | 4.13 | 最大项（22.7%）；多列专家 GEMV |
| qsa_proj | 6.27 | 3.49 | T>1 多列几何退化 |
| gdn_proj | 5.63 | 2.23 | |
| attn_hc | 5.14 | 1.88 | |
| moe_hc_mix | 4.86 | 1.71 | 已融合（G1） |
| moe_reduce | 4.39 | 2.83 | TP2 allreduce，随 T 线性 |
| moe_shared | 3.99 | 2.02 | |
| gdn_out | 3.82 | 1.71 | row-parallel allreduce |
| head_linear | 3.09 | 2.10 | 6 列 logits，必要 |
| qsa_attn | 2.73 | 1.10 | 近固定（上下文 KV 主导），摊薄好 |

判断：verify 是首要杠杆；但 MTP 时期已排除的项（routed 单列 GEMV、qsa_proj 小几何、moe_shared
列复用、allreduce、graph）不能无证据重开。唯一 DFlash2 专属的新假设是 **D3a：高 acceptance 使
6 列路由高度重叠**，若重叠率远高于 MTP 的 ~6%，grouped multi-column routed 才可能首次为正。先测
重叠率，再决定。

## D3a 结果（T=6 路由重叠，2026-11-27）

`NINFER_DFLASH2_ROUTE_OVERLAP=1` 诊断（门控关闭时 tok/s 不变 138.44/138.49）：

- 全体跨列重叠率 **39.08%**（逐层 0.158–0.592，后段 35–47 多在 0.42–0.59；rank0 37.6% / rank1 40.3%）。
- `distinct/total = 0.609`（T=6）；T=5 为 0.548，T=4 为 0.682。
- 结论：DFlash2 高 acceptance 使 6 列路由显著重叠，约为 MTP 旧测 ~6% 的 6.5 倍；**grouped multi-column routed 具备立项证据**，但性能收益必须实现后受控 A/B，严禁从读取量外推。
- 注：MTP 旧排除（P2a）基于 ~6% 重叠与 regs/CTA 门禁，条件不同，故不直接复用其否定结论；实现时仍须过 oracle/占用率门禁。

## D3a 负结果复盘（2026-11-27）

- 设计：每组专家按列**顺序**循环、单套 gate/up 累加器 + L1 权重复用，避免 6 份累加器；REG 50/52→48、down 76→56，CTA/SM 未降（10/9；6→9），资源门禁通过。
- Oracle：`ninfer_moe_nvfp4_test` 新增 T=2/6/8 grouped 用例（独立 FP32/FP64 oracle + raw-bit 比对）全绿；所有 token 与 plain/基线逐 token 相同。
- 端到端：关门 136.99、开门 119.23 tok/s（−13%）；verify 28.68→34.96 ms/轮。仅 group gate_up 125.95、仅 group down 125.64，均更慢。
- 根因：**gate_up 并发 CTA 从 9600 降到 1280**，权重 L2 复用收益不足以抵消并行度损失。说明"高重叠"在 sm_70 的 eager 逐层会合下不等于正收益；P2a 的占用率门禁虽过，但并行度门禁未过。
- 结论：默认关闭，保留代码作负结果证据。不再沿"分组减少 CTA"方向重开 routed。

## 本轮实施记录（2026-11-27）

已执行顺序：D0（K=5）→ D1（分相）→ D3a（负结果，默认关）→ D2（正结果，已接受）。
- env 门控诊断（默认零行为）：`program.h`/`program_impl.h`（轮次分相）、
  `expert_runtime.cpp/.h`（路由重叠）。
- D3a 负结果保留但默认关：`moe_nvfp4_launch.h`、`grouped_multi.cu`、`moe_nvfp4_test.cpp`
  （env `NINFER_DFLASH2_GROUPED_ROUTED=0`）。
- D2 正结果：`src/ops/kernel/dflash2_gemm.cu`（列数/warp 模板化，位级不变）。
下一步：D3b（qsa_proj/gdn_proj 在 T=6 的缩放）或暂缓项按需。

## DFlash2 优化项

| # | 项 | 状态 | 依据/下一步 |
|---|---|---|---|
| D0 | draft 数扫描 4→5→6→7（+min_p） | **完成：K=5 胜** | 配置级、零风险；512/128：K4 118.8、**K5 138.4**、K6 135.7、K7 133.9；ms/token 8.35/7.17/7.33/7.42 |
| D1 | 轮次分相：propose / T=6 verify / commit / materialize | **完成** | verify 28.73 ms/轮（72.8%）主导；propose 9.61（24.3%）；commit 0.5%；materialize 2.2% |
| D2 | drafter propose GEMM（SIMT warp-per-row） | **完成·正结果** | 列数/每 CTA warp 数模板化（位级不变）：propose 9.584→7.669 ms/轮（−20%），端到端 139.07→149.25 tok/s（+7.3%，独立复核 146.44）；小 n CTA 128→512、80→320 |
| D3 | T=6 目标 verify | **首要** | 占 72.8%；T=6 per-op：routed 14.99（22.7%）、qsa_proj 6.27、gdn_proj 5.63、HC ~10、moe_reduce 4.39 |
| D3a | T=6 routed 跨列专家复用 | **已实施·负结果（保留但默认关）** | 重叠 39.1% 成立、oracle/资源门禁均过，但 gate_up 并发 CTA 9600→1280，L2 收益抵不过并行度损失：119.2 vs 137.0 tok/s（−13%）、verify +6.28 ms/轮。env `NINFER_DFLASH2_GROUPED_ROUTED=0`。详见 `dflash2-grouped-routed-report.md` |
| D3b | qsa_proj / gdn_proj 在 T=6 的缩放 | 待测 | T6/T1 = 3.49× / 2.23×；先按 per-op 剖面判断是否有 T=6 专属退化 |
| D4 | DFlash2 CUDA Graph | 已排除 | DFlash2 强制 eager；sm_70 graph 已两次受控 A/B 为负 |
| D5 | feature materialize / ring(2048) | 视 D1 | 每轮 committed 特征的投影开销，量化后再定 |

## 暂缓项（用户裁决：先记录，稍后再看是否实施）

### 部署路径（不影响带 profile 的栅门，但影响无 profile 部署的真实性能）

- **专家缓存准入/替换抖动**：同一硬件/artifact，带 `--expert-cache-profile`（frozen）= 71.08
  tok/s / 0 miss；无 profile、cache 15000 = 18.89 / 7099 miss；cache 20000 + warmup 160 =
  37.66 / 1271 miss；cache 20000 + warmup 512 在 30 min 内未完成（miss/同步过重）。工作集在
  15 GB 内放得下（profile 0 miss 证明），故为**策略抖动**而非容量。潜在 2–4×，但只作用于部署路径。
  下一步（若实施）：追踪 miss/evict 模式，确认抖动，A/B 保留策略（pin 热集 / LFU / 复用保留）。

### base 设备侧（带 profile 栅门，剩余 ~3–6%，风险高）

- **hc_mix down→up 依赖气泡**：隔离微基准（真实形状，sm_70）down 741 GB/s、up 678 GB/s、
  down→up 序列 566 GB/s、依赖间隙 ~4.7 µs/层；流式地板 904–955 GB/s。`attn_hc`/`moe_hc_mix`
  每 token 48 层 × 2。若做 cooperative/persistent 融合，可去气泡，但占用率可能回退，需真 A/B。
- **hc_combine_norm 4-CTA 延迟**：grid=(1,4)，两趟 RMS，~11.5 µs/次 × 95/token（op 剖面
  2.2 ms 两卡合计）；属小核延迟，fold 兑现率未知，需受控 A/B。
- **qsa_attn 内部归因**：2.49 ms 两卡合计（12 次），KV 字节未计入，内部未拆；唯一可能藏 >5% 处，
  需先做 op 内 kernel 拆分。

### 已排除（勿重复，均有实测/机制证据）

- routed NVFP4 R=1 GEMV（A1 占用率局部最优）、qsa_proj 小几何（A2）、moe_shared 列复用
  （2026-11-26 复核）、gdn_out RowsPerCta（allreduce 主导；其低 GB/s 是 `row_parallel_linear`
  的 K 切 + TP2 allreduce，不是 GEMM）、moe_reduce（关全部 allreduce 仅 +3.5%）、CUDA Graph
  （sm_70 O(N) 节点）、双 stream（A4/A5/A6）、launch-merge（兑现率低）。

## 质量验证（2026-11-27）

用户要求"像 27B 一样跑同一套 BFCL 质量测试集"。frontend 已补齐 chat/function call 能力后，**Flash-Next 可以在该套 HTTP 路由上评估 BFCL**：
- 27B 的 BFCL-v3 `multi_turn_base` 走 `ninfer-serve` 的 `/v1/chat/completions` + `tools`，依赖 family 的 chat-template frontend。Flash-Next（qwen4_exp）的 `Frontend::prepare()` 现已实现同一条 chat-template/function-call 渲染路径（共享 `src/frontend/chat_template.cpp`），不再抛 `logic_error`，因此与 27B 走同一套 serving 路由。
- 复核证据：普通 chat、function call、多轮 tool result 请求均返回 HTTP 200；1 条 BFCL slice 的 accuracy 为 1.0。
- 口径：20 条用例尚未跑完，n=1 不构成质量结论；**全量 BFCL 评估仍未完成**，不能据此宣称整体质量等价。README 仍将 Flash-Next 标为"开发中，非发布产品"，但这不再阻碍用上述路由做 BFCL 评估。

可执行的等价质量证据（已完成）：
- DFlash2-5 与 plain greedy 在 512/128 上逐 token 相同（lossless），且改动前后相同。
- `ninfer_flash_next_text_nll` capital5（5 token）：改动前 HEAD `5ac8666c` 与本轮 HEAD `7cc80a2f` 均为 `greedy_first=11751`、`greedy_next=11751`、`top1_matches=2`、`mean_nll=3.73905609471`（各 2–3 次稳定复现）。**本轮优化质量零变化**；该 NLL 值与更早记录的 3.7311 的差来自更早的 prefill/config 路径，非本轮。
- 所有 oracle 测试（fused/draft/moe_nvfp4/mtp_*/load/options）绿。

## P1：prefill GDN 投影根因与优化（2026-11-27，已接受，默认开）

根因：`GdnProjectionLeaf::project` 在 T>4 走 `w8_f16_dispatch → launch_w8_simt_r8_c8_f16`，grid.y=T/8，每个 8 列 CTA 都重读整行 qkvz W8 权重（每卡每层 1.43 GB × 64 次，T=512）。决定性实验：prompt T=8→9 计算量 +12.5%，耗时 0.116→0.331 ms/rank/call（+2.85×），按 8 列台阶增长。

优化：扩展已验证的 sm_70 融合 tensor-core W8 GEMM 支持 FP16，新增 `ops::linear_split_fused_f16`；prefill T>4 走该路径，列 tile 8→32、权重每 32 列暂存一次。env `NINFER_FLASH_PREFILL_GDN_FUSED`（**默认开**，`=0` 可关闭对照）。T≤4 / MTP / DFlash2 路径不变。

证据：
- oracle `ninfer_linear_w8_f16_test` 相对 L2=2.4e-4（阈值 2e-2）；gdn_qkvz/mtp_*/gdn_replay 全绿。
- 质量（独立复核，capital5）：门关 `mean_nll 3.7391`、门开 `3.7469`（Δ+0.0079 ≤0.01），`greedy_first/greedy_next=11751`、`top1_matches=2` 不变；多 prompt N=128..512 greedy token 全等。
- A/B：gdn_proj 两卡 663.7→87.0 ms（−86.9%）；prefill 1.753→1.464 s（−16.5%，292→350 tok/s），TTFT −0.289 s；decode tok/s 不变。
- 决定：按项目 Δ≤0.01 + greedy 不变的门槛接受为默认，以换取 prefill 极致速度；`=0` 保留对照。

### P1 裁决修正（端到端净负，默认恢复关闭）

把 `NINFER_FLASH_PREFILL_GDN_FUSED` 设默认开后做受控 A/B（同二进制、门 0 vs 1、reps 2、DFlash2-5）：

| 指标 | 门关 (=0) | 门开 (=1) | 变化 |
|---|---:|---:|---:|
| committed tok/s | **149.48** | 137.29 | −8.2% |
| prefill s | 1.748 | **1.466** | −0.282 s |
| decode s | **0.850** | 0.925 | +0.075 s |
| acceptance | 0.9903 | 1.0000 | 模式改变 |
| expert misses | 0 | 0 | 无差异 |

prefill 单项确实 −16%，但 decode 变慢 +9%，端到端净 −8%。两者 expert misses 都为 0，故不是缓存抖动；acceptance 分布从"前高后低"变成"均匀"，greedy token 仍全等，说明是 prefill 结束状态的数值微扰传导到 decode/GDN 路径。按 AGENTS.md"端到端为准"与"局部优化不得损害整体"，**恢复默认关闭**（`=0`），保留实现与测量作证据。教训：prefill 与 decode 共享状态，prefill 的数值路径变更必须做端到端 A/B，不能只看 prefill 单项。

## 转向服务端性能（2026-11-27）

预填充与缓冲命中成为首要优化点，见 `docs/maintainer/serving-performance-backlog.md`。
关键事实：qwen4_exp 前缀复用完全未实现（`prefix_probe`/`has_retained_lane`/`evict_retained_lane` 均为 stub，
实测 `cached_tokens=0`）；冷专家缓存使 4096 prefill 慢 ~2.5×（130 vs 322 tok/s）；预填充本身线性 ~320 tok/s。
BFCL 全量因"无前缀复用 + 冷缓存"单轮极慢（0/200 在 7 分钟内未完成），暂停待优化后再跑。
