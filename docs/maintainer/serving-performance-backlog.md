# Flash-Next 服务端性能 backlog（2026-11-27）

目标：极致服务端性能，优先级 **缓冲命中（prefix/expert）→ 预填充 → 解码 → 首字延迟（TTFT）**。
硬件/口径：2×V100-SXM2-32GB、TP2、`sm_70`、CUDA 12.8、INT8 KV；target
`Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`；serve：`--max-concurrency 1`
（qwen4_exp 只服务一个活跃请求）、`--prefill-chunk 8192`、`--prefix-cache`、INT8 KV。

## 实测现状

### 预填充（暖专家缓存，no-spec，512..4096）

| prompt | prefill s | prompt tok/s |
|---:|---:|---:|
| 512 | 1.78 | 287 |
| 1024 | 3.12 | 328 |
| 2048 | 6.28 | 326 |
| 4096 | 12.74 | 322 |

预填充本身近似**线性 ~320 tok/s**，不是超线性。单 token 成本仍高（decode 为 149 tok/s，DFlash2-5）。

### 专家缓存冷/暖（4096，no-spec）

| 配置 | prefill s | prompt tok/s | misses |
|---|---:|---:|---:|
| 带 `--expert-cache-profile`（0 miss） | 12.74 | 322 | 0 |
| 无 profile（冷，动态准入） | 31.38 | 130 | 8360 |

冷专家缓存使 prefill **慢 ~2.5×**；且既有结论：无 profile 时动态准入会抖动（cache 20000 + warmup 160 仍 1271 miss）。

### 前缀缓存（prefix reuse）——**未实现**

受控实验（同一 serve，greedy）：
- req1 `prompt_tokens=24, cached_tokens=0`；
- req2（含 req1 相同前缀，`prompt_tokens=43`）仍 `cached_tokens=0`。

代码证据（`src/targets/qwen4_exp/impl/runtime/program_impl.h`）：
- `has_retained_lane()` 恒返回 `false`；
- `evict_retained_lane()` 是 no-op；
- `prefix_probe()` 返回空 `PrefixProbe`（无 block）。

即 qwen4_exp **完全没有前缀复用**：每个请求都冷启动全量 prefill。BFCL 多轮每轮 `reuse=full_reset`、TTFT ~88.8 s（6026 token ≈ 68 tok/s），是"无前缀复用 + 冷专家缓存"叠加的结果。

对照：qwen3_6（27B）在 4849 行 `program_impl.h` 中实现 retained lane / rewrite checkpoint / block snapshot / linear-state slot / dflash restore；qwen4_exp（1128 行）这些全是 stub。

## 优化项（按优先级）

| # | 项 | 现状 | 目标/路径 | 风险 |
|---|---|---|---|---|
| S1 | **qwen4_exp 前缀复用** | 未实现（stub） | 多轮/重复前缀只 prefill 新 suffix。**移植源 = `Neroued/ninfer` 的 `src/models/qwen4_exp/session/checkpoints.cpp` + `state/state_image`（与 qwen4_exp 同源），不是 qwen3_6**。前缀缓存**落 pinned 系统内存**：本 fork 已有 `src/runtime/prefix_cache`（`PrefixIndex` 不占 device、`SharedHostTier`/`HostSlabPool` pinned host，预算 `--host-cache-mib`）与 `state_transaction_impl`（GDN/QSA/PLE/route/KV/position 快照）。设计见 `.superpowers/.../qwen4exp-prefix-reuse-port.md`。 | 大改；分阶段 oracle + 端到端 A/B；不可改变 greedy 输出 |
| S2 | **专家缓存冷启动/抖动** | 已完成机制：`serve` 原先**没有** `--expert-cache-profile`（只有 bench 有）。已给 serve 增加该参数并映射到 `EngineOptions.expert_cache_profile`；DFlash2+KV 下可用显存 ~14.6GB，profile 需 `--expert-cache-mib` ≤12000 MiB 才放得下 | 服务端启动即加载 profile；目标 0 miss | 小；已验证 flag 生效 |
| S3 | 预填充吞吐（gdn_proj 权重每 8 列重读） | 已有 FP16 融合 GEMM 实现，默认关 | 该 op −87%、prefill −16.5%；但单请求 512/decode128 门禁下 decode 变慢 −8%（数值微扰改变 acceptance）。需在**服务端短 decode** 负载重估端到端；或先查 decode 变慢根因 | 中；已实现，需针对性 A/B |
| S4 | prefill routed（45%）/ moe_reduce | routed 按 token 重读，HBM 边际 ~719 GB/s | 需 grouped/tensor-core 大 T 专家 kernel（多次负结果，慎重）；moe_reduce 为 TP2 allreduce，随 T 线性 | 大；低优先 |
| S5 | 解码 | DFlash2-5 + GEMM ≈149 tok/s（2.1× base），lossless | 已较好；剩余 verify 72.8% | 中 |

## 结论

"缓冲命中没做好"属实且比"没做好"更严重：**qwen4_exp 的前缀缓存是从零未实现**。这直接决定多轮 TTFT 与 BFCL 吞吐。预填充本身 ~320 tok/s 线性，但冷专家缓存再叠 2.5×。

建议顺序：**S2（快速、低风险，先把冷缓存 2.5× 拿掉）→ S1（前缀复用，最大 TTFT/多轮收益，但大改）→ S3（预填充 op，针对服务负载重估）→ S4/S5**。

### 服务端实测日志（DFlash2 + profile，2026-11-27）

`ninfer-serve` 请求日志：
- `speculative=dflash2 1.67tok/round (50.0%)` —— 投机解码已生效（此前 BFCL 误用 `--spec none`）。
- `cache=0 reuse=full_reset reuse_reason=none` —— 前缀复用确认缺失。
- `prefill=46.3tok/s`（24 token）到 `79.3tok/s`（43 token）——短 prompt 固定开销占比高。

结论：服务端慢的三要素是 **无前缀复用（必须实现）+ 短/多轮反复全量 prefill + 冷专家缓存**；投机解码本身可用（`--spec dflash2 --draft-tokens 5 --draft-model <companion>`）。
