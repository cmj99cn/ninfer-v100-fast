# Qwen3.8-Flash-Next M4（TP2 全模型 + 专家常驻缓存）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 2×V100-SXM2-32GB 上以 TP2 跑通完整 48 层 Qwen3.8-Flash-Next，专家按可配置的常驻缓存 + host 兜底，输出真实 committed tok/s 与语料级质量数字。

**架构：** 包侧新增 TP2 分片映射，把 50 个角色拆成每卡视图（head/专家 intermediate 切分，小权重复制）；家族 `Program` 从"前 4 层截断"升级为完整 48 层，两卡同层并行、每层一次 allreduce；专家以固定地址的 VRAM 槽位常驻（容量由 `--expert-cache-mib auto|<MiB>` 决定），未命中从工件 mmap 读取并填入空闲槽位；KV 全容量常驻显存（1.74 GB），嵌入表置 pinned host 按行取，PLE 表逐行（已实现）并在每轮开始异步预取。MTP 不在本计划（独立计划）。

**技术栈：** C++20、CUDA 12.8（sm_70）、NVLink（6×25.78 GB/s）、工件 `qwen3_8_flash_next.ninfer`（108.4 GB，v3 目录）。

**规格：** `docs/maintainer/qwen3.8-flash-next-artifact.md`（§1–§13）、`docs/maintainer/paged-kv-cache.md`、`docs/maintainer/storage-layouts.md` §4/§11、`.superpowers/sdd/2026-11-12-flash-next-m2-text-tp1/progress.md`（M2 账本与裁定）。

## 全局约束

- 工件：`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`（env `NINFER_FLASH_NEXT_ARTIFACT`），**只读**；缺失返回 77。
- 硬件：2×Tesla V100-SXM2-32GB（每卡 32 GiB）、NV6 互连（6×25.78 GB/s）、PCIe Gen3 ×8（实测 H2D 6.13 GB/s）、94 GB host RAM。
- 已实测的显存账：专家 67.95 GB / 48 层（每 expert-pair 2.765 MB）、稠密 5.20 GB、嵌入+主头 2.54 GB、KV(262144, INT8 g64, 12 QSA 层) 1.74 GB、PLE 表 28.80 GB（host）。
- 已实测的路由倾斜（4 层 × 512 token）：42.7% pairs → 99.0% mass，51.7% → 99.9%；以 Strata 的 48 层曲线（91.2% pairs → 99.7%）作保守上界。
- 已实测的速度口径（Strata 层切分 4bit 对照）：28.9 ms/decode window = 34.6 次前向/秒；TP2 的理论窗口上限约为其 2×。
- 仓库规则：内部实现不加解释性注释（公开头契约注释保留）；C++20；既有 v2/v3 行为与既有包（`src/targets/qwen3_6*`）一字不改；家族内不得出现 target-dependent 分支；不新增第三方测试框架。
- 构建：`/home/cmj/strata/.venv/bin/ninja -C build -j 8`，configure 需 `-DCURL_INCLUDE_DIR=/home/cmj/media-deps/root/usr/include/x86_64-linux-gnu`；测试前 `export LD_LIBRARY_PATH=/home/cmj/nccl-lib/whl/nvidia/nccl/lib:/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu:/home/cmj/cuda128/targets/x86_64-linux/lib:$LD_LIBRARY_PATH`；ctest 用 `/home/cmj/strata-tools/lib/python3.11/site-packages/cmake/data/bin/ctest`。
- Python：`/home/cmj/1cat-vllm/venv/bin/python`（torch 2.10、numpy 2.3.5；不安装任何依赖）。
- 并行宽度只做 TP2（1 与 2）；TP4 与 Vision 留给后续里程碑。

## 审查重点（Review Focus）

- **专家缓存容量越小、越接近预算边界**：必须仍然产生正确 logits，且未命中路径与常驻路径逐位同值；不能因未命中而跳过专家或静默降级。
- **`--expert-cache-mib` 显式值大于可用显存**：必须拒绝启动并报出可用预算，而不是 OOM 或静默裁剪。
- **TP2 分片边界不合法（head 数不能被 2 整除、K128 对齐不满足、词表不整除）**：必须精确报错并指出角色，而不是越界读或错位。
- **同一 token 在两卡上得到的路由/块选择不一致**：必须防止（复制的 router/indexer 保证一致）；出现不一致要显式断言，而不是各算各的。
- **单卡在 decode 中途被另一卡拖慢（allreduce 未重叠）**：`--devices` 顺序、NVLink 可用性、mailbox 初始化必须显式验证。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/targets/qwen3_8_flash_next/impl/load/sharding.h` / `.cpp` | 50 角色的 TP2 分片映射：每角色的轴、每 rank 范围、复制集合 |
| `src/targets/qwen3_8_flash_next/impl/load/bindings.h` / `.cpp` | 生成每 rank 的 `ModelView`（含 batched NVFP4 专家的 intermediate K 切分） |
| `src/targets/qwen3_8_flash_next/impl/load/expert_cache.h` / `.cpp` | 专家槽位表、容量策略、profile 种子、host mmap 兜底、命中/未命中统计 |
| `src/targets/qwen3_8_flash_next/impl/load/embedding_source.h` / `.cpp` | 嵌入表的 pinned-host 行源（按 token 取行） |
| `src/targets/qwen4_exp/impl/runtime/text_context_impl.h` | 48 层层循环、TP2 allreduce、专家 gather、PLE 预取 |
| `src/targets/qwen4_exp/impl/runtime/layouts.h` / `layouts_impl.h` | 两卡页池/workspace/状态规划 |
| `src/targets/qwen4_exp/impl/runtime/program_impl.h` | 完整 48 层的 prefill/decode 成功路径 |
| `src/targets/registry.h` / `registry.cpp`、`src/runtime/engine/engine.cpp` | flash-next 的 Engine 接线（按 R22 顺延） |
| `include/ninfer/types.h`、`src/serve/serve_options.{h,cpp}` | `expert_cache_mib` 选项与 `--expert-cache-mib` |
| `tools/bench/run_flash_next_bench.py`、`tools/reference/qwen3_8_flash_next/` | 速度与质量度量脚本 |
| `tests/targets/qwen3_8_flash_next/{test_sharding,test_expert_cache,test_full_model,test_engine_route}.cpp` | 对应测试 |

---

### 任务 1：TP2 分片映射

**文件：**
- 创建：`src/targets/qwen3_8_flash_next/impl/load/sharding.h`、`sharding.cpp`
- 测试：`tests/targets/qwen3_8_flash_next/test_sharding.cpp`

- [ ] **步骤 1：编写失败的测试**

`test_sharding.cpp`（无需工件）断言：
- 每个 `roles::Role` 都有映射；映射覆盖 `roles::kRoleCount` 全部项。
- `ShardAxis::Head` 的角色（GDN q/k/v/z、GDN output、QSA query/key/value/output_gate/output、index_query）在 `tp=2` 下每 rank 范围拼起来等于父范围，且起点按 head 宽度对齐。
- `ShardAxis::Intermediate` 的角色（moe shared gate/up/down、专家 gate/up/down）在 `tp=2` 下每 rank 覆盖一半，且 K128 对齐成立。
- 复制集合恰为：`token_embedding`、`output_head`、`final_hc/*`、全部 `norm`、`moe/router`、`moe/shared_score`、`gdn/a_log`、`gdn/dt_bias`、`gdn/a_projection`、`gdn/b_projection`、`gdn/convolution`、`attention/index_key`、`attention/index_key_norm`、`attention/index_query_norm`、`attention/key_norm`、`attention/query_norm`、全部 `attention_hc/*` 与 `moe_hc/*`、`ple/*`。
- `tp=1` 时所有角色返回单一全范围。

- [ ] **步骤 2：运行测试验证失败**

运行：`./build/tests/ninfer_flash_next_sharding_test`
预期：编译失败（`sharding.h` 不存在）。

- [ ] **步骤 3：实现 `sharding.h` / `sharding.cpp`**

```cpp
enum class ShardAxis { Replicated, Head, Intermediate };

struct Shard { std::int64_t begin; std::int64_t end; };

struct ShardMapping {
    ShardAxis axis;
    std::array<Shard, 2> ranks;
};

[[nodiscard]] ShardMapping shard_mapping(roles::Role role, int tp);
```

`Head` 轴按角色的 head 宽度（GDN key 128、value 128、QSA 256、indexer 128）切分；`Intermediate` 轴按 `moe_intermediate_size / tp` 且要求 128 对齐；越界或不可整除抛 `std::invalid_argument` 并带角色名。

- [ ] **步骤 4：运行测试验证通过**

运行：`./build/tests/ninfer_flash_next_sharding_test`；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen3_8_flash_next/impl/load/sharding.h src/targets/qwen3_8_flash_next/impl/load/sharding.cpp tests/targets/qwen3_8_flash_next/test_sharding.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): add the TP2 shard map for the text roles"
```

---

### 任务 2：每 rank 视图与专家 K 切分

**文件：**
- 修改：`src/targets/qwen3_8_flash_next/impl/load/bindings.h/.cpp`
- 测试：`tests/targets/qwen3_8_flash_next/test_sharded_bindings.cpp`

- [ ] **步骤 1：编写失败的测试**

真实工件门控（`NINFER_FLASH_NEXT_ARTIFACT`，缺失 77）：
- 对 `tp=2` 构造两个 `ModelView`；对每个 `ShardAxis::Head` 角色，两 rank 的 `Weight::data` 起点差等于 `begin * row_bytes`；对 `Intermediate` 角色等于按布局计算的字节偏移。
- 专家 `BatchedNvfp4Weight` 每 rank 的 `rows == 640/2`（gate/up）与 `columns == 640/2`（down），且两 rank 的 `scale_offset(row, group)` 指向各自矩阵内平面。
- 复制角色的两 rank 视图指向同一对象偏移。

- [ ] **步骤 2：运行测试验证失败**

运行：`NINFER_FLASH_NEXT_ARTIFACT=... ./build/tests/ninfer_flash_next_sharded_bindings_test`
预期：失败（无 `tp` 参数/仍是单 rank 视图）。

- [ ] **步骤 3：实现每 rank 视图**

`bindings.h` 增加

```cpp
struct RankViews {
    std::array<ModelView, kMaximumDevices> views;
};

[[nodiscard]] RankViews shard_views(const BindingPlan& plan,
                                    const artifact::MaterializedArtifact& materialized,
                                    const ShardPlanByName& shards, int tp);
```

`ShardPlanByName` 由任务 1 的 `shard_mapping` 生成；专家 bank 走 `BatchedNvfp4Weight` 的矩阵内行区间（gate 行 `[0,320)`/`[320,640)`，up 同理；down 走列切分并保证 K128 对齐）。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen3_8_flash_next/impl/load/bindings.h src/targets/qwen3_8_flash_next/impl/load/bindings.cpp tests/targets/qwen3_8_flash_next/test_sharded_bindings.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): build the per-rank weight views"
```

---

### 任务 3：TP2 数值等价（4 层）

**文件：**
- 修改：`src/targets/qwen4_exp/impl/runtime/text_context_impl.h`、`layouts_impl.h`、`workspace_recipe.h`
- 测试：`tests/targets/qwen3_8_flash_next/test_tp2_equivalence.cpp`

- [ ] **步骤 1：编写失败的测试**

真实工件 + 2 GPU：
- 用前 4 层分别在 `tp=1` 与 `tp=2` 上跑同一 8-token 提示，比较最后一步 `sample_hidden` 与 logits：相对 L2 误差 ≤ BF16 边界界（4e-3）；greedy token 相同。
- 用 `tp=2` 跑两次，logits 逐位相同（allreduce 顺序固定）。
- 计数器断言：每层恰有 1 次残差/allreduce 与 1 次下投影 allreduce（通过注入的 `CollectiveStats`）。

- [ ] **步骤 2：运行测试验证失败**

运行：`./build/tests/ninfer_flash_next_tp2_equivalence_test`
预期：失败（主干没有 TP2 分支）。

- [ ] **步骤 3：实现 TP2 主干**

在层循环中：每 rank 计算本地 head 的 attention/GDN 输出与本地 intermediate 的 MoE 输出，`allreduce_sum` 到完整 hidden；HC 与 router/indexer 复制（两 rank 各自算同一路由与块选择，断言一致）；残差以 BF16 写回两 rank。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen4_exp/impl/runtime tests/targets/qwen3_8_flash_next/test_tp2_equivalence.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): run the text trunk across two ranks"
```

---

### 任务 4：可配置专家常驻缓存

**文件：**
- 创建：`src/targets/qwen3_8_flash_next/impl/load/expert_cache.h`、`expert_cache.cpp`
- 修改：`include/ninfer/types.h`（`expert_cache_mib`）、`src/serve/serve_options.{h,cpp}`
- 测试：`tests/targets/qwen3_8_flash_next/test_expert_cache.cpp`

- [ ] **步骤 1：编写失败的测试**

- `ExpertCachePlan::auto_sized(available_bytes, pair_bytes)` 返回的 pair 数与覆盖率（用 `evidence/routing_skew.py` 产出的曲线）满足"≥99.5% mass"或"预算耗尽"二者之一，并报告二者。
- 显式容量大于可用显存 → 抛 `std::invalid_argument`，消息含可用字节与请求字节。
- 槽位表：固定的 `slots` 数量、每个 slot 的 device 地址稳定；`resolve(layer, expert)` 命中返回 slot，未命中返回 `std::nullopt` 并计入 `miss_count`。
- 替换策略确定性：同一命中序列产生同一 slot 分配。
- 未命中读取：`HostPairSource::read(layer, expert)` 从工件 mmap 取 2.765 MB，内容哈希与直接读工件一致。

- [ ] **步骤 2：运行测试验证失败**

运行：`NINFER_FLASH_NEXT_ARTIFACT=... ./build/tests/ninfer_flash_next_expert_cache_test`
预期：编译失败。

- [ ] **步骤 3：实现缓存**

```cpp
struct ExpertCacheOptions {
    std::optional<std::uint64_t> capacity_bytes; // nullopt => auto
    std::uint64_t vram_reserve_bytes = 0;
    std::optional<std::filesystem::path> profile; // (layer, expert, count) 文本，按 count 降序
};

class ExpertCache {
public:
    ExpertCache(const ExpertCacheOptions&, std::uint64_t pair_bytes,
                std::uint64_t available_bytes, HostPairSource&);
    [[nodiscard]] std::optional<std::uint32_t> find(std::uint32_t layer, std::uint32_t expert) noexcept;
    [[nodiscard]] std::uint32_t load(std::uint32_t layer, std::uint32_t expert); // 返回 slot
    [[nodiscard]] CacheStats stats() const noexcept;
};
```

槽位是固定数量的 device slab；`load` 先未命中统计再填 slot；替换用 clock（LRU 近似），profile 只决定初始填充顺序。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add include/ninfer/types.h src/serve/serve_options.h src/serve/serve_options.cpp src/targets/qwen3_8_flash_next/impl/load/expert_cache.h src/targets/qwen3_8_flash_next/impl/load/expert_cache.cpp tests/targets/qwen3_8_flash_next/test_expert_cache.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): add the configurable expert residency cache"
```

---

### 任务 5：完整 48 层装载与显存规划

**文件：**
- 修改：`src/targets/qwen3_8_flash_next/impl/config.h`（去掉 4 层截断）、`impl/load/bindings.cpp`、`src/targets/qwen4_exp/impl/runtime/layouts_impl.h`、`program_impl.h`
- 创建：`src/targets/qwen3_8_flash_next/impl/load/embedding_source.h/.cpp`
- 测试：`tests/targets/qwen3_8_flash_next/test_full_model.cpp`

- [ ] **步骤 1：编写失败的测试**

真实工件 + 2 GPU：
- 装载完整 48 层（专家走缓存，容量 `auto`），断言：装载成功、`memory_summary()` 的 device 占用 ≤ 每卡 32 GiB 且 host 占用 ≤ 94 GB 的 80%。
- 提示 8 token：greedy token 与参考模型（`tools/reference/qwen3_8_flash_next`，48 层）一致；若参考 48 层运行超时，用 `NINFER_FLASH_NEXT_REFERENCE_TOKEN` 占位并记录。
- 嵌入行源：取 token 248319 的行哈希与直接读工件一致。

- [ ] **步骤 2：运行测试验证失败**

运行：`NINFER_FLASH_NEXT_ARTIFACT=... ./build/tests/ninfer_flash_next_full_model_test`
预期：失败（仍只装载 4 层）。

- [ ] **步骤 3：实现装载与规划**

- `TextConfig::layers` 改为 48（唯一来源）；`kLoadedLayers` 的截断路径删除。
- `layouts_impl.h`：页池按 `max(kv_capacity, capacity)`；专家 slab 大小 = `auto` 预算；嵌入表放 pinned host（`cudaHostAlloc`），每 prefill chunk 按需取行。
- `memory_summary()` 报告 weights/experts-cache/kv/workspace/host 分项。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen3_8_flash_next src/targets/qwen4_exp/impl/runtime tests/targets/qwen3_8_flash_next/test_full_model.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): load and run the full 48-layer model"
```

---

### 任务 6：PLE 行预取与未命中路径

**文件：**
- 修改：`src/targets/qwen4_exp/impl/runtime/text_context_impl.h`、`src/targets/qwen3_8_flash_next/impl/load/expert_cache.cpp`
- 测试：`tests/targets/qwen3_8_flash_next/test_ple_prefetch.cpp`

- [ ] **步骤 1：编写失败的测试**

- 每轮 decode 的 16 个 PLE 行读取在第一个 GDN 层开始前完成（用注入的时序探针断言"发起时刻 < 层 0 结束时刻"）。
- 未命中专家读取与 PLE 行读取并发（`HostPairSource` 与 `PleRowSource` 的发起/完成计数满足重叠）。

- [ ] **步骤 2：运行测试验证失败**

运行：`./build/tests/ninfer_flash_next_ple_prefetch_test`；预期失败。

- [ ] **步骤 3：实现预取**

在 round 开始时用 `cudaMemcpyAsync` 在独立 stream 上发起 PLE 行与未命中专家的 H2D，事件在消费点等待；不改变数值语义。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen4_exp/impl/runtime/text_context_impl.h src/targets/qwen3_8_flash_next/impl/load/expert_cache.cpp tests/targets/qwen3_8_flash_next/test_ple_prefetch.cpp tests/CMakeLists.txt
git commit -m "perf(flash-next): overlap the PLE and expert-miss reads"
```

---

### 任务 7：Engine 接线（按 R22 顺延项）

**文件：**
- 修改：`src/targets/registry.{h,cpp}`、`src/runtime/engine/engine.cpp`、`src/serve/serve_options.{h,cpp}`
- 测试：`tests/targets/qwen3_8_flash_next/test_engine_route.cpp`

- [ ] **步骤 1：编写失败的测试**

真实工件 + 2 GPU：
- 以 `devices = {0,1}`、`expert_cache_mib = 0`（auto）、`kv_capacity` 262144 构造 Engine，提交一个 token-id 提示，取回 greedy token 与 `metrics()`；断言 token 与任务 5 一致、`metrics().committed_tokens == 1`。
- `--expert-cache-mib` 大于可用显存 → `Engine` 构造抛错且消息含预算。

- [ ] **步骤 2：运行测试验证失败**

运行：`NINFER_FLASH_NEXT_ARTIFACT=... ./build/tests/ninfer_flash_next_engine_route_test`
预期：失败（registry 无对应实例）。

- [ ] **步骤 3：实现接线**

`construct_target` 增加 flash-next 分支；`EngineOptions` 增 `expert_cache_mib`；CLI 增加 `--expert-cache-mib`（与 `--host-cache-mib` 并列）；Engine 只对 flash-next 传 2 个设备视图。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2；预期 PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/registry.h src/targets/registry.cpp src/runtime/engine/engine.cpp src/serve/serve_options.h src/serve/serve_options.cpp tests/targets/qwen3_8_flash_next/test_engine_route.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): route the engine to the two-rank package"
```

---

### 任务 8：速度与质量度量

**文件：**
- 创建：`tools/bench/run_flash_next_bench.py`
- 修改：`tools/reference/qwen3_8_flash_next/{config.py,model.py}`（支持 48 层）
- 测试：`tests/targets/qwen3_8_flash_next/test_reference_parity.py`（增加 48 层单点 parity）

- [ ] **步骤 1：编写失败的测试**

- `test_reference_parity.py` 增加 48 层 8-token 用例：参考 greedy token 与 C++（Engine 路径）一致；参考运行时间记录（预计 ~40 s/token）。
- `run_flash_next_bench.py`：以 token-id 语料在 512 / 3.7K / 15K / 59K occupied context 上测 committed tok/s（单请求、greedy）、TTFT、专家缓存命中率与未命中字节；输出 JSON 到 `profiles/bench/`。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/1cat-vllm/venv/bin/python tests/targets/qwen3_8_flash_next/test_reference_parity.py`
预期：48 层用例失败（参考仍按 4 层配置）。

- [ ] **步骤 3：实现参考 48 层与度量脚本**

`config.py` 增加 `layers=48` 模式（参考侧按需逐层物化，专家只解码被路由的矩阵）；基准脚本只通过 Engine 公共 API 驱动，不读内部状态；质量口径为 token-id 语料上的平均 NLL（同一语料在参考与 C++ 上各算一次，容忍度按 FP32/FP64 与 BF16 边界的实现配置给出）。

- [ ] **步骤 4：运行测试验证通过并记录数字**

运行：步骤 2 的脚本 + 基准脚本；把实测 tok/s、命中率、TTFT、NLL 差异写入 `docs/maintainer/qwen3.8-flash-next-artifact.md` §13（纯数字口径，不回退既有条款）。

- [ ] **步骤 5：Commit**

```bash
git add tools/bench/run_flash_next_bench.py tools/reference/qwen3_8_flash_next tests/targets/qwen3_8_flash_next/test_reference_parity.py docs/maintainer/qwen3.8-flash-next-artifact.md
git commit -m "bench(flash-next): measure the full-model speed and corpus quality"
```

---

## 自检记录

- **规格覆盖度**：M4 目标（完整 48 层、TP2、专家缓存、真实速度/质量）逐项对应任务 1–8；MTP（M3）与 Vision（M6）明确不在本计划；Engine 的聊天/服务前端（分词器、模板、detokenize）不在本计划，度量走 token-id 语料并已在任务 8 说明。
- **步骤扫描**：每步都有可检查结果；签名与阈值来自实测数字（2.765 MB/pair、99.5% mass、4e-3 BF16 界、6.13 GB/s、28.9 ms/window）。
- **类型一致性**：`ShardAxis`/`ShardMapping`（任务 1）、`RankViews`/`shard_views`（任务 2）、`ExpertCache`/`ExpertCacheOptions`（任务 4）、`expert_cache_mib`（任务 4/7）在后续任务同名同指。
- **审查重点**：五条分别由任务 3（两卡一致与 allreduce 计数）、任务 4（容量边界/拒绝启动/未命中等值/替换确定性）、任务 1（边界非法报错）、任务 7（预算拒绝）与任务 3/7（`--devices` 与 NVLink 验证）钉住。
- **比例**：计划只写决定（文件、接口、阈值、测试断言），不抄函数体。
