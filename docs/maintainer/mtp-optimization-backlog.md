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
| A3 | `route` 4-warp 并行 `route_select` | 0.8/rank | **已做（提交 36279352）** | T4−T1 Δ −82%（−0.8 ms/rank），bit-exact；受控 gate 110.4→113.0 tok/s（+2.4%），acceptance 不变，已推送。 |
| A4 | `moe_shared ∥ routed` 双 stream | 0.6–1.0 | **已排除（根因实验）** | `debug-moe-shared-rootcause.md`：H2 成立——gate_up device kernel 与 routed 争 SM/带宽，rout ed +0.82 ms/token；H1（event 开销）证伪。第二 stream 只能跑 host I/O/H2D，不能切 device kernel。 |
| A5 | `qsa_proj` 两半双 stream | 0.2–0.4 | **已排除** | `verify-a5-a6-feasibility.md`：两半皆 device kernel、合计 ~303 GB/s 带宽饱和，与 A4 同类 H2；0.2–0.4 外推不可信。 |
| A6 | `qsa_tail` ∥ `qsa_index/attn` | 0.05–0.15 | **已排除** | `verify-a5-a6-feasibility.md`：层内无真依赖属实，但跨层 tail 发布是真依赖需 event，device kernel 仍争 SM，量级 < 噪声。 |
| A7 | `moe_reduce` NCCL（1.07 ms、48 次/层） | 0.3–0.5 | **已排除（大改）/未做（小改）** | 关掉全部 allreduce 仅 +3.5%（`progress.md`），层间依赖难合并；PP 契约禁止。 |
| A8 | verify CUDA Graph | 0（负） | **已排除** | sm_70 O(N) 节点提交：3054 节点 → 0.6 ms/round，超过可削的已掩盖 host 间隙；两次受控 A/B −4.7% / −4.4%。Strata 同 V100 能用是因其 kernel 内 t 复用属 eager 属性，我们已领取。 |
| A9 | 成本感知 verify 长度（HybridSpec/EcoSpec） | 1–3 | **已排除** | `verify-a9-feasibility.md`：与已落地 min_p=0.8 提前终止重叠；greedy 严格前缀不允许跳过 interior verify；draft top-10 路由无接受预测信号；期望收益 ≈0。 |

## B. draft 侧（23.7%）

| # | 项 | ms/轮 潜力 | 状态 | 证据/原因 |
|---|---|---:|---|---|
| B1 | 专家 7-kernel 融合 | 0.15–0.3/step | **已排除（根因确证）** | `debug-b1-rootcause.md`：根因是并行度/解码串行——256 线程×4 codes 的 decode 搬进 1-warp MMA 后每线程串行解 640 codes（×160）且与每轮 K-tile 屏障耦合；ldexpf 是放大器非根因（去掉仍 4.1×），m4 的全局原子根因不适用；当前形态不可修，需改独立高并行 decode kernel 或 grouped LUT 形态。 |
| B2 | device router top-k 去 host D2H+sync | 0.10–0.25/step | **已排除（实测前提不成立）** | `b1-b2-draft-report.md`：router sync host 阻塞实测仅 0.0050 ms/step，外推的 0.15–0.30 bubble 不存在；top-k+10 槽分发无收益且有舍入/选集风险。 |
| B3 | 跨轮门控（上轮 step-0 置信度低则 K=0） | 条件性 | **已排除** | `verify-b1-b3-feasibility.md`：无 step-0 分布统计；Q4 0.8929/FP16 0.9380 均 > 0.8；min_p=0.8 下条件收益 = 0。 |
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
| C4 | base 优化向 verify 传导 | **已执行·穷尽** | GDN extract 实得 +0.64 tok/s（预期外推错）；moe_shared 双流 H2 不可修；HC/qsa_proj 剩余外推均不成立（`verify-remaining-base-extrapolations.md`）。base 栈已近上限。 |

## 执行排序（已执行完毕）
所有"未做·可落地"项已实现或验证：A3 已落地；A4/A5/A6/A9/B1/B2/B3 均已排除（根因/实测）。
剩余到 2× 只有契约变更方向：并行/多头 draft artifact 或 graph 友好平台（见 C1/C2）。

## 硬上限
k=3 达 141 tok/s 需每轮 ≤23.5 ms；4 token HBM floor 22.84 ms 已占 97%。
单靠 MTP 侧增量优化预计上限 ~127 tok/s（1.8×）；到 2× 需改 draft artifact（并行/多头 head）
或平台（graph 友好的 Ampere+），均为契约变更。
