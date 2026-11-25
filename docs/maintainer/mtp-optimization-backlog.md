# MTP 剩余可优化点清单（转向 base 前的冻结记录）

日期：2026-11-25。当前已提交推送：HEAD `234768d4`，110.3–110.5 tok/s（1.56× base 70.54），
配置 Q4 head + min_p=0.8 + draft=3 + 0 miss + TP2 + greedy。

每轮 29.27 ms：verify 21.61（73.8%）、draft 6.94（23.7%）、commit+tail 0.72（2.5%）。

## 状态定义
- **未做·可落地**：有量级、有路径、无阻断。
- **已排除**：实测或机制证明在当前栈下不可行，附原因，避免重复。
- **已做**：本轮已落地并计入 110 tok/s。
- **部分完成**：代码在树里但未做端到端 A/B。

---

## A. verify 侧（73.8%，主战场）

| # | 项 | ms/轮 潜力 | 状态 | 证据/原因 |
|---|---|---:|---|---|
| A1 | `routed` 专家 GEMV 效率（4.28 ms、~100 GB/s vs 850 floor） | 理论 3.7 | **已排除** | `analysis-routed-truth.md`：Strata 的 −6.8% 是 IQ2_XS/q8_1 LUT 格式专属；我们低效率是软件 NVFP4 decode GEMV（R=1、regs 限制），加宽掉 occupancy；P2a 按 expert 分组实测 +12%（top-10 跨列重叠仅 ~6%）。结论：当前栈下接近最优，勿再投。 |
| A2 | `qsa_proj` 多列效率（1.78 ms、120 GB/s） | 理论 1.46 | **已排除（本轮两方案）** | `qsa-route-opt-report.md`：实际几何是 256/512/128 不是 1664；登记新几何走 GEMV multi 回归 3–7%（CTA 数降 1/4），batched MMA decode 回归 6–10%。未放宽 oracle。若未来有 sm_70 专用小几何多列 kernel 可重评。 |
| A3 | `route` 4-warp 并行 `route_select` | 0.8/rank | **部分完成·未 gate** | `qsa-route-opt-report.md`：T4−T1 Δ −82%（+0.954→+0.164 ms/rank），bit-exact、测试绿；代码在树里，未跑 engine_bench A/B。 |
| A4 | `moe_shared ∥ routed` 双 stream | 0.6–1.0 | **未做·可落地** | `analysis-parallel-next.md`：二者无真依赖；需第二 scratch + 规约后 add；并行报告有 lxe/volta `VOLTA_SHARED_EARLY` 佐证。 |
| A5 | `qsa_proj` 两半双 stream | 0.2–0.4 | **未做·可落地** | 同上；需两份 workspace，注意 NCCL 排序。 |
| A6 | `qsa_tail` ∥ `qsa_index/attn` | 0.05–0.15 | **未做·低优先** | 同上。 |
| A7 | `moe_reduce` NCCL（1.07 ms、48 次/层） | 0.3–0.5 | **已排除（大改）/未做（小改）** | 关掉全部 allreduce 仅 +3.5%（`progress.md`），层间依赖难合并；PP 契约禁止。 |
| A8 | verify CUDA Graph | 0（负） | **已排除** | sm_70 O(N) 节点提交：3054 节点 → 0.6 ms/round，超过可削的已掩盖 host 间隙；两次受控 A/B −4.7% / −4.4%。Strata 同 V100 能用是因其 kernel 内 t 复用属 eager 属性，我们已领取。 |
| A9 | 成本感知 verify 长度（HybridSpec/EcoSpec，复用 draft top-10 路由做代价预测） | 1–3 | **未做·最有潜力** | `analysis-industry-breakthroughs.md`：低置信度轮缩短 verify，无损；行业 1.19→1.36× 证据。注意当前 min_p=0.8 已提前终止 draft，需评估叠加收益。 |

## B. draft 侧（23.7%）

| # | 项 | ms/轮 潜力 | 状态 | 证据/原因 |
|---|---|---:|---|---|
| B1 | 专家 7-kernel 融合（decode→GEMV×2、down GEMV→scale_add epilogue） | 0.15–0.3/step | **未做·可落地** | `analysis-draft-next-levers.md`；gate_up→down 真依赖不融；bit-exact oracle 需守。 |
| B2 | device router top-k 去 host D2H+sync | 0.10–0.25/step | **未做·可落地** | 同上；与 B1 配套否则收益受限。 |
| B3 | 跨轮门控（上轮 step-0 置信度低则 K=0） | 条件性 | **未做·低优先** | min_p=0.8 下多为 0；Strata cost model 思路。 |
| B4 | draft head 词表子集（Strata 106299） | 0.07–0.23/step | **已排除** | Q4 全词表已 0.397 ms（带宽地板），子集再省有限；且 greedy argmax 在子集上会改 token，需改 artifact 语义。 |
| B5 | 更大 draft window（5/7） | 负 | **已排除** | 实测 window=5 → 20 tok/s（draft 步数线性增长压垮收益）。 |
| B6 | Q4 proposal head | 2.8/step | **已做** | 110 tok/s 的主要来源；oracle 已更新为 Q4 语义，FP16 开关保留。 |
| B7 | confidence min_p 提前终止 | 0.4/step | **已做** | 88→110 路径的一部分。 |

## C. 跨阶段 / 其它

| # | 项 | 状态 | 证据 |
|---|---|---|---|
| C1 | draft ∥ verify 逐 token 流水线 | **已排除** | verify 是一次 T=4 forward 不是逐 token；单头自回归 draft 无法拆分；需要改 artifact。 |
| C2 | TP2 verify 角色再平衡 / EP | **已排除** | 契约禁止；SM121 的 EP +9% 不可移植 sm_70。 |
| C3 | 跨请求并发复用 | 附带观察 | 产品 bench 单请求；非本目标。 |
| C4 | base 优化向 verify 传导 | **转向执行** | `analysis-base-to-verify-transfer.md`：共享 helper 收益 ×1.0–3.1 传导到 verify；base 70.54→80 ⇒ MTP 116.7–131.9。这是当前最有价值的新方向。 |

## 执行排序（若回到 MTP）
A9 → A4 → A3 的 gate A/B → A5 → B1+B2 →（A8/A2/A1 不再尝试）。

## 硬上限
k=3 达 141 tok/s 需每轮 ≤23.5 ms；4 token HBM floor 22.84 ms 已占 97%。
单靠 MTP 侧增量优化预计上限 ~127 tok/s（1.8×）；到 2× 需改 draft artifact（并行/多头 head）
或平台（graph 友好的 Ampere+），均为契约变更。
