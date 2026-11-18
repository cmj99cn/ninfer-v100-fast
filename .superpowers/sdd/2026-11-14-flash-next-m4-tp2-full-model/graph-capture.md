# M5-14: flash-next decode 整模型 CUDA Graph

日期：2026-11-18　工作树：`.worktrees/flash-next`　HEAD（起始）：`32f17467`
硬件/工具链：2×Tesla V100-SXM2 32GB，sm_70，CUDA 12.8，TP2，262144 capacity，INT8 group-64 KV，
expert cache 20000 MiB，工件 `qwen3_8_flash_next.ninfer`。

## 结论（状态）

**阶段 A 的“capture 可行性”成立，但阶段性正确性/性能验收未完全达标。**

- 整 token 的 48 层主图 capture **成功**：TP2 两张 per-rank 图，`ThreadLocal` 同时 capture，
  `node_count=2954`（约 1477/rank），一次 `cudaGraphLaunch`/rank（2 launch/token，eager 为 2390）。
- replay **可运行**，wall `18.9 ms/token`（约 53 tok/s），相对 eager baseline `21.4–39.2 ms`
  下降约 25–50%，但**未达到 ≤14 ms（≥70 tok/s）目标**。
- replay token **大多数**与 eager 一致，但存在偶发分叉；logits **未做到逐位一致**，因此
  **未把图接入生产 Engine 的 decode 路径**，避免质量风险。

## 已做改动

1. `src/targets/qwen4_exp/impl/runtime/text_context_impl.h`
   - 新增 capture seam：`NINFER_QWEN4EXP_GRAPH_PREFETCH`；把 winner 回读改为
     `NINFER_QWEN4EXP_GRAPH_WINNER_COPY` seam（默认仍为同步 `cudaMemcpy`，生产语义不变）。
   - WIP 既有 seam（IDS/POSITIONS/SCALARS/PREPARE_PLE/PLE_AWAIT/ROUTED_FINISH/FINISH_SYNC/WINNER_PTR）保留。
2. `tools/bench/flash_next_graph_profile.cpp`（WIP 骨架补全）
   - 仅在 capture 期间把 host 指针换成 pinned 固定缓冲、把 host 同步/专家 miss/PLE host 读换成 no-op；
     prefill/warmup 走 eager。
   - winner 与 logits 经 pinned 缓冲 D2H（异步，可入图）；图外每轮补 PLE issue/collect。
   - 用 reset+prefill+warm 重建同一初始态，对比 eager 与 replay 的 token/logits 并测量 wall。

`ninfer_flash_next_package` 等其他生产库未改行为；`ninfer_flash_next_text_nll` 质量守卫通过。

## 捕获失败原因（逐个排除，已修）

| 失败点 | 原因 | 处理 |
|---|---|---|
| `cudaMemcpyAsync` H2D `cudaErrorInvalidValue` | WIP 把图 seam 用在 prefill（T=512，源缓冲仅 64 B） | 用 `g_capturing` 开关让非 capture 走 eager；缓冲扩到 8192 B |
| `expert_runtime.cpp:170 cudaStreamSynchronize` | `post_mixer_prefetch` 的 `prefetch_previous_round` 含 host 同步 | 新增 `GRAPH_PREFETCH` seam，capture 期间 no-op |
| 同步 `cudaMemcpy` D2H | 捕获流中禁止同步 memcpy | 新增 `GRAPH_WINNER_COPY` seam，capture 期间用 `cudaMemcpyAsync` |
| 捕获后 eager `expert_runtime.cpp:439 cudaEventSynchronize` | `probe_done_` 在 capture 中被 record，成为 capture event，外部不可同步 | 捕获后置 `g_graph_mode`，跳过 routed_finish/prefetch |
| replay 与 eager PLE 不一致 | `prepare_ple_rows` 含 `advance_ple_tail` 副作用，图内被 no-op | 每次 launch 前在图外显式 prepare + await |

## 验证数据与命令

构建：`/home/cmj/strata/.venv/bin/ninja -C build -j 8 ninfer_flash_next_graph_profile ninfer_flash_next_phase_profile ninfer_flash_next_text_nll`

Eager baseline（cache 20000）：
```
NINFER_FLASH_PHASE_CACHE_MIB=20000 ./build/tests/ninfer_flash_next_phase_profile $ARTIFACT 512 192
warm_misses≈7287; decode wall/token = 25.36, 21.40, 37.02, 25.43, 39.25, 32.78 ms
routed(host) 17–44 ms/token（每层 cudaEventSynchronize + miss rollback 重跑）
```

Graph capture（最终两次）：
```
./build/tests/ninfer_flash_next_graph_profile $ARTIFACT
{"capture_ok":true,"tp":2,"node_count":2954,"capture_position":704,
 "eager_token":271,"token_match":false,
 "replay_tokens":[271,271,271,271,248046,271],
 "replay_wall_ms_avg":18.93,
 "eager_vs_graph_logit_diff":248205,"eager_vs_graph_logit_maxdiff":49137}
{"capture_ok":true,"tp":2,"node_count":2954,"capture_position":704,
 "eager_token":271,"token_match":false,
 "replay_tokens":[271,248046,271,271,4097,271],
 "replay_wall_ms_avg":18.92,
 "eager_vs_graph_logit_diff":198278,"eager_vs_graph_logit_maxdiff":39371}
```
`token_match=false` 由偶发分叉（248046/4097）导致；logits 差异涉及几乎全部 248320 项，
最大原始 FP16 差约 4–5 万，**不是 1-ULP 舍入**，因此不能宣称逐位一致或质量等价。

质量与回归：
```
ninfer_flash_next_text_nll --tokens capital5.ids  -> greedy_first 11751, mean_nll 3.7311021849（不变）
ctest -R flash_next  -> 22 passed, 0 failed, 1 skipped（23）；full_model golden `504 359 81486` 不变
```

## 为什么未继续硬上（阶段 B/C 的阻塞）

1. **专家路由表是 host 可变的**：capture 时的 `lookup_` 由 eager warmup 的 miss 处理刷新；
   rebuild/replay 时 T=512 prefill 与 T=1 miss 处理会改变常驻集，图内 baked 的路由决策随之漂移。
   这是偶发 token 分叉和 logits 大差异的主因。需要阶段 B 的
   `ExpertCache::seed(profile)` 静态常驻 + 设备端 hit 决策 + 图外异步补 miss + 互补求和，
   并保持 lookup 表在 replay 窗口内不变。
2. **PLE 行源有 host 副作用**：`prepare_ple_rows` 会推进 `ple_tail_`，必须在图外以固定 tail 快照
   issue/collect；当前工具每轮 rebuild 恢复 tail，但未做静态 profile 化的 PLE 常驻。
3. **事件生命周期**：capture event 不能在后续 eager 路径同步；生产图路径需要独立的
   probe/down 事件集或完全移除 host 读取（阶段 B 的状态机）。
4. **性能上限**：即使完全跳过 miss 处理，纯图 GPU 工作仍约 18.9 ms（GDN proj ~4.4、
   MoE routed ~7、head linear ~3.7 ms/rank 等真实 kernel 时间）。14 ms 目标需要阶段 B 的
   “去同步 + 互补分区 + 零重跑”以及 kernel 级改动，单纯 capture 无法达成。

## 建议

不要在当前状态把图接入公共 Engine。先完成阶段 B：静态 profile 常驻使 lookup 表在 replay 窗口冻结、
设备端 hit 决策、图外异步补 miss + 设备端求和、PLE issue/collect 与事件集隔离；再做阶段 C
（两 rank 同时 capture + mailbox 入图）并以 `flash_next_graph_profile` 的 bit-exact token/logits
与 `phase_profile` 的 wall 为门槛。本次仅交付 capture 接缝与可复现的验证工具。
