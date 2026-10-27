# NInfer 多实例共享缓存

本文是多实例 serving 的稳定权威文档，描述同一进程内多个同构 Engine 的分组、共享边界、准入、
路由、共享 host tier 与 manifest 生命周期，以及当前验证状态。单实例执行语义仍由
[小规模并发推理架构](concurrent-inference-architecture.md)和
[Paged KV Context Store](paged-kv-cache.md)定义。

## 1. 范围

- 一个 `EnginePool` 拥有 N 个 Engine；所有组使用同一 artifact、同一 tensor-parallel 宽度和互不
  重叠的设备。
- 进程内只打开一份 artifact reader / mmap（`SharedModel`），并在启用 host tier 时只分配一份
  pinned host KV tier（`SharedHostTier`）。
- 每个 Engine 保留自己的 device 权重、device KV、GDN/backend 状态、snapshot slots、CUDA 流和
  executor。
- 前端只有一个 OpenAI/Anthropic 入口；`EnginePool` 是所有请求唯一的 admission 队列。

非目标：多进程/多机、运行中抢占或迁移、运行期改变组数或卡数、同进程混合 TP 宽度、TP8。
多组模式首版只支持 MTP 或无 draft；`--groups > 1` 与 `--spec dflash` 在启动时拒绝。

## 2. 分组与启动参数

| 参数 | 语义 | 默认 |
|---|---|---:|
| `--groups N` | 进程内 Engine 数 | `1`（单 Engine 路径） |
| `--group-devices "0,1;2,3"` | 按组显式列出设备，分号分组；省略 `--tp` 时由组宽推断 | 无 |
| `--devices A,B,...` | 未给 `--group-devices` 时，按组顺序列出全部 `groups * tp` 个设备 | `--device` 起连续 |
| `--tp K` | 每组张量并行宽度，必须为 1、2 或 4；所有组同宽 | `1` |
| `--max-concurrency N` | 每个实例的 lane 数；多组时按实例配置 | `tp=1:1`、`tp=2:4`、`tp=4:8` |
| `--kv-capacity N\|auto` | 每个实例的 Main Text KV 容量；多组省略时为 `align64(max_concurrency * max_context)` | 推导值 |
| `--global-pending-requests N` | 共享 admission 队列上限 | `16` |
| `--http-threads N` | HTTP worker 数；`0` 使用推导值 | 多组 `groups*max_concurrency + global_pending + 1` |
| `--affinity-slack-tokens N` | idle home 的软黏性加分（token） | `2048` |
| `--fairness-aging-ms N` | 队头 aging 阈值 | `1000` |
| `--host-cache-mib N` | 进程共享 pinned host tier 预算；需与 `--prefix-cache` 一起使用 | `0` |

`--group-devices` 与 `--devices` 同时给出时必须一致；设备不能重复。单组路径保持原有
`--tp`/`--devices` 语义，`--groups 1` 不构造 pool。

共享 host tier 由第 0 个实例在 `--prefix-cache --host-cache-mib N` 下创建，其余实例附着同一个
tier。未启用 prefix cache 或预算为 0 时不分配 host tier，跨实例 manifest 恢复不可用；device
内 prefix 命中仍然工作。

2 卡本机（正确性已验证、性能未测，见 §7）：

```bash
./build/apps/ninfer-serve /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --groups 2 --group-devices "0;1" --tp 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --prefix-cache --host-cache-mib 1024
```

8 卡机（本任务窗口不可访问；未验证）。profile 由脚本统一展开，避免文档与命令漂移：

```bash
bash tools/v100/bench_multi_instance_8gpu.sh 4xTP2
bash tools/v100/bench_multi_instance_8gpu.sh 2xTP4
```

| profile | `--groups` | `--group-devices` | `--tp` | `--max-concurrency`（每实例） | `--kv-capacity`（每实例） |
|---|---:|---|---:|---:|---:|
| `4xTP2` | 4 | `0,1;2,3;4,5;6,7` | 2 | 4 | 1048576 |
| `2xTP4` | 2 | `0,1,2,3;4,5,6,7` | 4 | 8 | 2097152 |

两份 profile 共用 `--max-context 262144`、`--global-pending-requests 16`、`--kv-dtype int8`、
`--spec mtp --draft-tokens 3`、`--prefix-cache --host-cache-mib 153600`（150 GiB）、
`--no-thinking --greedy`，并把 `--request-log-jsonl` 写到
`profiles/bench/multi_instance_<mode>.requests.jsonl`。默认可执行文件是
`build/apps/ninfer-serve`（可用 `NINFER` 覆盖），默认 artifact 是
`/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer`（可用 `NINFER_V100X2_ARTIFACT` 覆盖）。

`--dry-run` 只打印完整命令，不创建目录、不读取 artifact、不分配 GPU，因此可在 2 卡机上安全
执行：

```bash
bash tools/v100/bench_multi_instance_8gpu.sh 4xTP2 --dry-run
bash tools/v100/bench_multi_instance_8gpu.sh 2xTP4 --dry-run
```

压测由独立进程执行，避免 serve 与 benchmark 重启互相干扰；在仓库根目录另开一个终端：

```bash
# 4 × TP2
python3 tools/v100/bench_multi_instance.py --groups 4 \
  --request-log profiles/bench/multi_instance_4xTP2.requests.jsonl

# 2 × TP4
python3 tools/v100/bench_multi_instance.py --groups 2 \
  --request-log profiles/bench/multi_instance_2xTP4.requests.jsonl
```

默认 16 个并发请求、每组目标 50K-token 前缀、`max_tokens=128`；实际 prompt token 数由响应
`usage.prompt_tokens` 记录。脚本从
`http://127.0.0.1:8080/metrics?format=json` 读取 `instances` 数组和聚合计数器，记录
committed decode tok/s、客户端 TTFT、MTP acceptance（来自 `--request-log` 的
`request_done.speculative`）、queue wait 估计和采样到的队列压力，并把完整报告写到
`profiles/bench/multi_instance_<UTC>.json`。默认 bearer token 与启动脚本一致
（`ninfer-eval-key`），可用 `--api-key` 或 `NINFER_API_KEY` 覆盖。`--dry-run` 只打印请求计划
与目标 URL，不发请求、不写文件。两份 8 卡 profile 只是可复现配置，容量、pinned host 预算和
吞吐尚未实测；不得把 2 卡或单实例数据当作 8 卡验收。

## 3. 共享边界与所有权

| 对象 | 所有权 |
|---|---|
| `SharedModel`（artifact reader / mmap） | `EnginePool`，所有 Engine 共享只读 |
| `SharedHostTier`（pinned slab pool + manifest 目录） | `EnginePool`；第 0 个实例创建，其余实例附着 |
| device 权重、KV pools、GDN/backend state、snapshot slots | 每个 Engine 私有 |
| `PrefixIndex`（device/host 索引、foreign 节点、device LRU） | 每个 Engine 私有 |
| `ConcurrentExecutor`、CUDA 流、CUDA Graph | 每个 Engine 私有 |
| `Frontend`、media cache | 每个 Engine 私有 |
| 全局 pending 队列、permit、routing 计数 | `EnginePool` 私有 |

共享 host tier 的 slab 元数据、manifest 目录和引用计数由 tier 的一把 mutex 串行化；D2H/H2D
传输在锁外、在各自实例的 cache stream 上执行。foreign 节点只引用 tier 的 host 页，不拥有它们。

## 4. Admission、permit 与路由

请求准备只在实例 0 上执行（同一 artifact 的 tokenizer/frontend），`PreparedPrompt` 不含实例
身份，可提交给任意 Engine。

`EnginePool::submit`：

1. 把 prepared prompt 转成全局 pending entry；`required_pages =
   ceil((prompt_tokens + requested_output_tokens + 64) / 64)`。
2. 入全局 FIFO；队列已满立即返回 `RequestErrorKind::Overloaded`（HTTP 429，
   `server_overloaded`）。
3. `EngineRouter` 为每个 `(pending, instance)` 读取该实例的 prefix probe 和 capacity snapshot，
   在所有有 room 的候选中评分。
4. 选中后在 pool 锁内同时预留一条 lane 和 `required_pages` 个 KV 页，生成 permit；permit 是
   该请求在实例上的唯一 resource claim。
5. 锁外调用目标 Engine 的 `submit`。Engine 拒绝时归还 permit。

候选容量口径：

- `active_requests` 使用该实例当前持有的 permit 数，而不是 Engine 的内部计数；一个 permit
  就是一个 lane claim。
- `pending_requests` 在 pool 视图中为 0。多组时每个 Engine 的 `max_pending_requests` 强制为 1，
  其队列只承载已经路由到它的那一个请求；pool 的全局队列才是唯一的 admission 等待队列。
- `free_pages` 是该实例报告的 free pages 减去尚未被 Engine 映射的 permit 预留页，因此 permit
  不会被重复计算。

路由评分：

- 前缀深度取 `max(device_frontier, manifest_frontier)`；manifest 命中同时产生跨实例
  `PrefixReuseHint`。
- admission 时 home 是该 prompt 在 pool 内 device frontier 最深的实例；当 home 空闲且深度差距
  不超过 `--affinity-slack-tokens` 时优先 home。
- 等待超过 `--fairness-aging-ms` 的队头请求直接取自己的最佳候选，避免饿死。
- 同分依次比较 `pending_requests`、`active_requests`、实例序号。
- 不会为了等待更深的前缀而空闲 lane。

错误沿用现有路径：全局队列满为 429 `server_overloaded`；排队或准备阶段超过
`--pending-timeout-ms` 为 503 `request_queue_timeout`。permit 在 `PoolHandle::wait()` 完成后
释放；未完成就销毁 handle 同样释放 permit。

## 5. Manifest 生命周期与跨实例恢复

- 源请求正常完成时，若路径上的 block 和 snapshot 状态镜像都在共享 host tier 中 host-resident，
  Program 发布一个 manifest；发布失败只表示少一个可迁移点，不报错。
- manifest 记录 `tp`、64-token block 边界 frontier、block hash 链、各 block host page id 和
  state page id，不含第二份 KV 拷贝。
- 发布为 manifest 引用的 host 页加 pin；manifest 自身有引用计数和 30 分钟 TTL（`Config` 默认）。
  引用归零后 manifest 才可被淘汰；slab 分配失败时优先淘汰已过期 manifest，否则按频次/最近命中
  淘汰。淘汰或 owner 退出时释放页引用，页在引用归零后回到 slab pool。
- 目标实例收到 hint 后重新验证 live manifest 的 id 和 frontier；失效则清空 hint、记录
  `HintRejected` 并走冷路径，不恢复半状态。
- 恢复成功的请求在本地 `PrefixIndex` 建立 foreign 节点。foreign 节点引用 manifest 的 host 页，
  不进入本地 device LRU，也不通过本地 allocator 释放这些页。
- 请求正常完成时，恢复出的 device 页和 foreign 链提交到本地索引，后续同实例请求可直接 device
  命中；取消/错误路径释放 import 和 device 页，不提交。
- 同一 manifest 可以被多个实例引用；manifest 的 host 页只在 tier 淘汰或 erase 时回收，不会被
  消费实例释放。

v1 只发布完整 64-token 边界的 Tap snapshot；partial-tail 或没有完整 continuation state 的前缀
不进入共享 manifest。

## 6. 指标

`MetricsSnapshot` 的既有聚合字段保持不变：devices 逐卡拼接，scheduler 与 KV pages/counters
求和；共享 host tier 是进程级对象，只报告一份，不按实例求和。新增 `instances` 数组，每项为
`InstanceMetrics`：

| 字段 | 来源 |
|---|---|
| `free_kv_pages` | 该实例 `main_kv.free_pages` |
| `active_requests` | 该实例 `scheduler.running_requests`（prefilling 已包含在内） |
| `pending_requests` | 该实例 `scheduler.waiting_requests`（只含已路由请求） |
| `committed_tokens` | 该实例 `scheduler.committed_decode_tokens` |
| `prefix_hit_tokens` | permit 完成时累计的 `GenerationResult.reused_prompt_tokens` |
| `manifest_restore_tokens` | 带 manifest hint 且完成路径为 `ReusedBlockSnapshot` 时累计的 hint frontier |

`prefix_hit_tokens` 覆盖所有 prefix 复用路径；`manifest_restore_tokens` 是 pool 可观测的
跨实例 manifest 归属，若目标实例恢复的 live manifest 比 hint 更深，指标只记录 hint frontier。
两者都是进程生命周期累计计数器。

Prometheus 暴露 `ninfer_instance_free_kv_pages`、`ninfer_instance_active_requests`、
`ninfer_instance_pending_requests`、`ninfer_instance_prefix_hit_tokens_total`、
`ninfer_instance_manifest_restore_tokens_total` 和 `ninfer_instance_committed_tokens_total`，
均带 `instance` 标签。JSON 响应增加 `instances` 数组；单实例时该数组为空。`instances` 为空时
Prometheus 输出不增加任何 instance 行，单组文本输出与之前一致。

## 7. 验证状态

已运行的 host 证据（无 GPU、无 artifact）：

- `ninfer_engine_pool_test`：全局队列、permit 的 lane+KV 预留、queue-full 429、
  permit-timeout 503、取消释放。
- `ninfer_engine_router_test`：pairwise probe、home bonus、aging、tie-break。
- `ninfer_shared_host_tier_test`、`ninfer_prefix_cache_host_test`、
  `ninfer_prefix_cache_index_test`：manifest 注册/引用/淘汰、foreign import/commit/release、
  host 页所有权。
- `ninfer_metrics_format_test`：两个 fake 实例的聚合（总量等于分项和）以及 Prometheus/JSON 渲染。

2 卡真实集成（2026-10-24 补跑，2×V100-SXM2-32GB，NVFP4 artifact）**已运行并通过**：

- `ninfer_qwen3_8_27b_multi_instance_real_test` 通过（101.90 s）：2×TP1 的共享 host tier 只
  分配一次、两个 Engine 共享同一 `SharedModel`；跨实例 manifest 恢复与冷启动的 greedy 逐
  token 一致；`pool round2 reused=8128/8128`；16 个排队请求在全局队列中的 FIFO/aging/取消
  行为通过；2×TP1 的跨实例 MTP 通过。
- `ninfer_qwen3_8_27b_v100x2_concurrent_prefix_real_test` 通过（89.11 s）：1×TP2 四 lane
  并发共享同一前缀时，每个 lane 的 greedy 答案与该 lane 的单 lane reference 完全一致，且
  多 lane 批处理可复现。该模型与 prompt 下 reference 本身收敛为同一答案，因此测试只在
  reference 互异时才要求 lane 互异。
- 2 卡 TP2 单组前后对比（2026-10-24，NVFP4）：baseline `605f6c1` vs 当前工作树，同一 TP2
  profile、单请求、每点 2 次冷启动且 before/after 交替；实际 prompt 8133/49461 tokens。
  8K：TTFT 3894.1→3928.6 ms（+0.88%），decode-phase 102.49→102.48 tok/s，end-to-end
  committed 24.686→24.503 tok/s；50K：TTFT 27855.5→27864.5 ms（+0.03%），decode-phase
  90.72→90.74 tok/s，end-to-end committed 4.340→4.339 tok/s；acceptance 58.0%/54.9% 不变。
  单请求无 >2% 回归。这是 NVFP4，不是 85K occupied context 或多实例吞吐验收，也没有
  LM Studio 对照。
- 质量对比（2026-10-24，NVFP4，TP2，greedy）：4 个冷启动 prompt（算术、斐波那契、代码、
  8133-token prose）的 before/after `generated_token_ids` 完全一致；8K prefix reuse 第二次
  `prefix_cache_hit_tokens=8128`、`block_snapshot`，cold/warm 与 before/after 文本一致；
  MTP 3 drafts 下 8K warm 也一致，acceptance 20/30、10 rounds。未覆盖采样质量与 Vision。

8 卡机在本任务窗口不可访问，以下全部**未验证**：

- NCCL 多 communicator（每进程 4 个 TP2 或 2 个 TP4）在 8×V100 上的初始化、带宽与设备集合
  互斥。
- TP4 跨实例 host restore 的显存与正确性，包括恢复后的 MTP 与 Vision。
- 16×50K token 的聚合 committed decode tok/s（分别对 4×TP2 与 2×TP4）是否达到单实例 3.0×，
  以及实际 TTFT、queue wait 和 acceptance；由 `tools/v100/bench_multi_instance.py` 记录，
  尚无测量。
- 4×TP2 在 `kv_capacity=1048576` 下的 32GB 显存余量；8 卡下每实例 180000-token KV 的余量。
- 153600 MiB（150 GiB）pinned host tier 在目标主机上的实际可分配性。
- 多组 Vision：NVFP4 的 2×TP1/4×TP2 Vision prompt 未在本窗口验证。
- 多组 MTP：2×TP1 已通过（见上）；8 卡 4×TP2/2×TP4 的多组 MTP 未验证。

8 卡启动与压测脚本已通过静态检查（`bash -n`、`py_compile`、两条 `--dry-run`），但未在 8 卡
上执行。2 卡 TP2 单组已做上述前后对比，但未覆盖 85K occupied context、多实例吞吐或
LM Studio 对照；不得把静态检查、正确性测试、质量对比或这些单组数据当作多实例/验收性能证据。

本仓库当前没有可引用的多实例吞吐测量；不得把单实例、短 prompt、draft 吞吐或上游数据当作
多实例验收。8 卡验收通过前，`AGENTS.md` 的 4×16GB 单组路径仍是回归门禁。

## 8. 已知限制

- 不抢占、不迁移正在 prefill/decode 的请求；只在新请求 admission 时改变 placement。
- 多组不支持 DFlash；`--groups > 1 --spec dflash` 在启动时拒绝。
- 一个实例段错误会带走整个进程；本设计不提供进程级隔离。
- Vision 的 device 权重仍按实例私有，每个实例各持一份。
- 运行时不能改变组数、每实例 lane 数或 host tier 预算。
