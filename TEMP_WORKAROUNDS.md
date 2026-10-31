# 临时改动 / 绕过 复核清单

> 生成时间：2026-10-14
> 用途：所有非正式修复、临时绕过、降级配置的记录，后续逐一复核处理。
>
> **历史记录（2026-10）：** 当前发布目标是 2×V100-SXM2-32GB / NVFP4 / 262144。
> 自转换 NVFP4 制品包含 Vision；GGUF Q4_K_M legacy 制品拒绝 Vision。以下条目保留为
> 历史记录，执行前请先核对当前源码和 active 文档。

---

## 1. 模型加载：跳过 vision 绑定（架构代差）

**文件**：`src/targets/qwen3_6_27b/impl/load/bindings.cpp`

**改动**：`!features.vision` 时不再调用 `bind_vision_backbone`，改用 `validate_unconsumed_matching("vision/")` 忽略所有 vision tensor。

**原因**：neroued 官方制品 `Qwen3.8-27B-nvfp4-NInfer` 的 vision tower 与本 fork 不兼容：
- 制品：q6/q5/q4/q8 group-wise 量化、RoPE attention、**无独立 qkv 投影**
- fork 契约：原生 FP16 linear + 传统 qkv 结构

**影响**：这条记录针对官方 `neroued` 制品；当前自转换 NVFP4 制品已包含兼容的 Vision
tower，`--vision` 可用。GGUF Q4_K_M legacy 制品仍拒绝 `--vision`。

**状态（2026-10）**：已被当前 NVFP4 制品取代；保留为历史记录。

**复核 TODO**：
- [ ] 如仍需支持官方 `neroued` 制品，移植新 vision 架构（RoPE attention、无 qkv、量化 linear 路径）
- [ ] 或明确拒绝该制品，而不是保留降级路径
- [ ] 保持 `--vision` 在 GGUF Q4_K_M 上明确报错

---

## 2. ~~Paged KV trim 与 prefix cache 冲突~~ → 已修复

**现象**：BFCL 多轮对话触发 server 错误：
```
Paged KV trim extent would discard cache-owned pages
```
之后所有请求返回 `503 service_unavailable`，引擎不可用。

**根因**：`src/core/paged_kv_cache.cpp:568` 的 trim 校验——prefix cache 复用（`reuse=block_snapshot`）产生的 cache-owned 页与后续 trim 冲突。当请求 B 的前缀比请求 A 的 promoted 页面短时，trim 试图丢弃 cache-owned 页。

**修复**（2026-10-14）：`trim_pages` 改为 clamp 到最高 cache-owned 页，保留这些页不 trim。多出的页是有效 KV，只是超出调用方的 frontier，安全。

**状态**：✅ 已修复并验证（15 轮长上下文压力测试通过，prefix cache 正常命中 `reuse=block_snapshot`）。

**复核 TODO**：
- [x] 修复 `PagedKVAllocation::trim_pages` 对 cache-owned 页的处理逻辑
- [ ] 重测带 prefix cache 的 BFCL（原 Strata 对照组也是无 prefix cache？需确认）
- [ ] 考虑是否需要更精细的处理（如 unpin cache-owned 页而不是保留）

---

## 3. 无 sudo 环境下的 ffmpeg/libcurl 依赖（本地 .deb 解包）

**方案**：`apt-get download` + `dpkg-deb -x` 解包到 `/home/cmj/media-deps/root/`，通过 `PKG_CONFIG_SYSROOT_DIR` + `PKG_CONFIG_PATH` 指向。

**涉及包**：libavcodec-dev/libavformat-dev/libavutil-dev/libswscale-dev/libswresample-dev/libcurl4-openssl-dev + 对应运行时包。

**CMake 配置**：
```
-PKG_CONFIG_SYSROOT_DIR=/home/cmj/media-deps/root
-PKG_CONFIG_PATH=/home/cmj/media-deps/root/usr/lib/x86_64-linux-gnu/pkgconfig
-DCMAKE_EXE_LINKER_FLAGS="-L.../media-deps/... -Wl,-rpath,..."
```

**复核 TODO**：
- [ ] 公司服务器如有 sudo，改用系统包管理器安装
- [ ] 或将 media-deps 打包为可复现的安装脚本

---

## 4. NCCL 库路径（/tmp symlink → 已持久化到 /home/cmj/nccl-lib）

**原始问题**：pip 包只有 `libnccl.so.2`，CMake 链接需要 `libnccl.so`；最初在 `/tmp/nccl-lib/` 建了 symlink（重启会丢）。

**已持久化**：`/home/cmj/nccl-lib/` 下 symlink 指向 pip 包实际路径。

**复核 TODO**：
- [ ] 确认公司服务器上 NCCL 的标准安装方式（可能已有系统 NCCL）
- [ ] 构建脚本里固化 NCCL 路径发现逻辑

---

## 5. 构建配置：NINFER_SKIP_FFMPEG 开关

**文件**：顶层 `CMakeLists.txt`、`src/CMakeLists.txt`、`src/media/decode/decode_stub.cpp`、`src/media/decode/acquire_stub.cpp`

**用途**：无 libav 头文件的测试环境跳过 media 依赖，用空桩编译。

**当前状态**：`NINFER_SKIP_FFMPEG=OFF`（多模态支持已编译），桩代码保留但未使用。

**复核 TODO**：
- [ ] 桩代码是否合入主分支（上游可能不需要此开关）
- [ ] 或仅保留在本地 patch 不提交

---

## 6. splitd 内核移植的适配改动（非临时，但需上游对齐）

**文件**：
- `src/ops/launcher/gqa_attention_volta_splitd.cu`（新增）
- `src/ops/launcher/gqa_attention_volta_flash.cu`（钩入）
- `src/CMakeLists.txt`（源文件 + include scope）
- `third_party/llama_cpp_sm70_d256/`（vendor 闭包）

**关键适配**（与原始 splitd 补丁的差异）：
1. geometry 改名 + 新增 `Gqa27Tp2Geometry(12/2)`、`Gqa27Tp4Geometry(6/1)`
2. static_assert 放宽到 QHeads∈{6,12,16,24}
3. **去掉 normalized D256 Hadamard**（tp4 int8 codec 无旋转）
4. `NINFER_VOLTA_SPLITD=0` kill switch（默认开）
5. SplitKV3 阈值 env `NINFER_SM70_SPLITKV3_MIN_KV`（默认 2048）

**实测**：174K prefill +7.5%（1059.7 vs 985.4 tok/s），远低于原作者 +36%——可能因 Qwen3.8-27B 仅 1/4 层为 full attention（splitd 只作用于该路径）。

**复核 TODO**：
- [ ] 确认 Hadamard 去掉对数值精度的影响（跑 AIME/GPQA 对比）
- [ ] +7.5% 是否符合预期，是否需要针对 linear-attention 层的优化
- [ ] vendor 闭包是否可精简（162 个 sm70-vendor 头文件）
- [ ] 与上游 splitd 补丁的合并策略

---

## 7. 模型制品来源

**当前**：`/home/cmj/models/ninfer-artifacts/qwen3_8_27b_nvfp4.ninfer`（22GB）
**来源**：`hf-mirror.com/neroued/Qwen3.8-27B-nvfp4-NInfer`

**注意**：官方制品 vision 不兼容（见 #1）；文本权重格式（nvfp4/q4/q5/q6）与 fork 主干 bindings 匹配。

**复核 TODO**：
- [ ] 确认是否应使用 fork 官方推荐的其他制品版本
- [ ] SHA256 校验（制品内有 `SHA256SUMS` 但未验证）

---

## 8. 评测配置对齐说明

| 项 | Strata | ninfer |
|---|---|---|
| 模型 | Qwen3.8-Flash-Next IQ2_XS | Qwen3.8-27B NVFP4 |
| prefix cache | 关（推测） | **开**（已修复 trim bug） |
| thinking | off | off |
| sampling | greedy | greedy |
| 上下文 | 262144 | 262144 |
| KV dtype | - | int8 |
| MTP | - | `--spec mtp --draft-tokens 3` |

**注意**：模型不同，质量分数**不可直接对比**，只能作各自基线。

---

## 9. Decode 性能调优发现

**LONG_SPLIT_KEYS env**（`NINFER_SM70_LONG_SPLIT_KEYS`）：已加到 `gqa_attention_decode.cu`，但 A/B 实测**无效**（480 vs 1920 无差异）。瓶颈不在 attention split 策略。

**MTP 是 decode 关键**：
- 无 MTP：~21 tok/s @ 31K
- 有 MTP：~36 tok/s @ 31K（+72%）
- 186K decode-only：~100-120 tok/s（与 tpx 101.9 持平）

**实测 decode 速度**（MTP on, 分离 prefill）：
| 上下文 | decode tok/s |
|---|---|
| 4K | ~119 |
| 8K | ~94 |
| 16K | ~100 |
| 186K | ~100-120 |

**结论**：decode 速度已达标，无需进一步优化 attention 内核。

---

## 10. 工具调用解析器严格性缺口（vs vLLM qwen3_coder）

**文件**：`src/serve/tool_call_parser.cpp`（解析）+ `src/serve/generation_service.cpp:411-418`（调用点，对整段文本解析）

**背景**：与 1cat-vLLM 的 `qwen3coder_tool_parser.py` 对比，两边对**标准 XML 输出**
`<tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>`
解析结果一致（实测 BFCL 轨迹里 `ls {"a":true}` 两边都被正确解析成结构化 tool_calls）。
但 ninfer 实现更严，存在以下缺口（当前 BFCL 口径下**不是**失败主因，非阻塞）：

| 维度 | ninfer | 1cat |
|---|---|---|
| 外壳 | 必须 `<tool_call>…</tool_call>` | `<tool_call>` **或裸 `<function=…>`**（back-off） |
| 体内格式 | 仅 XML | XML **+ `<tool_call>` 内 JSON body** `{"name","arguments"}` |
| 参数类型 | `trim`+`JSON.parse`（不看 schema） | 按工具 schema `coerce_to_schema_type` 强制转换 |
| 缺闭合标签 | 直接失败 | 可恢复缺失的 `</parameter>`/`</function>` |
| 值内假标签 | 贪婪取首个 `</parameter>`（值内同串会误切） | 结构化边界判断，忽略值内 `<parameter=` |
| 部分失败 | 任一 block 坏 → **整体回退成 content（丢弃全部 tool_calls）** | 过滤坏的、保留好的 |
| 尾部杂文本 | `</function>` 后必须仅空白 | 更宽松 |
| args 序列化 | nlohmann `dump()` → `{"a":true}` | `json.dumps` → `{"a": true}` |

**影响**：在非标准/轻微畸形输出下，ninfer 会把本该是工具调用的响应退化成普通 content，
导致 agent 循环/任务失败；1cat 能救回。当前模型+模板产出的都是标准 XML，故未暴露。

**复核 TODO**：
- [ ] 支持裸 `<function=…>`（无 `<tool_call>` 外壳）的 back-off
- [ ] 支持 `<tool_call>` 内 JSON body `{"name","arguments"}`
- [ ] 参数值按工具 schema 做类型转换（而非一律 JSON.parse）
- [ ] 缺失 `</parameter>`/`</function>` 时尽量恢复，而非整体回退
- [ ] 部分 block 解析失败时保留已成功的 tool_calls
- [ ] 参数值内联 `<parameter=`/`</parameter>` 的结构化边界处理
- [ ] 加单测覆盖上述畸形输入（当前无 tparser 单测）

---

## 11. ~~NVFP4 预填充数值缺陷~~ → 已定位并修复：Split-D staging 跨 rank 竞态

**真因（2026-10-18 复核后修正）**：`src/ops/launcher/gqa_attention_volta_splitd.cu` 的 Split-D 预填充路线把
staging 缓冲、smem 属性开关、pinned positions 探针都放在**函数级 `static`** 里。本引擎由**单个主机线程**
驱动两个 TP rank（`ops::detail::for_each_rank` 按 rank 设当前设备），于是 rank 1 复用 rank 0 的设备内存：
两个 rank 的 splitd 启动并发读写**同一组** staging 缓冲；当 `kv_len` 越过 `kSplitKv3MinKv=2048` 后，
split3 的 `partial_out/partial_max/partial_sum` 也落在同一组缓冲上。

**现象**（TP2）：prefill 从**第 8 个 chunk（KV=2048）**起 mixer 输出非确定；首 token argmax 在
`<tool_call>` / `I` 间抖动；BFCL `multi_turn_base` 默认配置 20–35% 且每进程不同，chunk=128
（split3 不触发）为 50% = 1cat 水平。

**修复**（commit `486ba771` 修复 + `f86b63ad` 改用既有约定）：Split-D 的启动期状态一律**按 CUDA 设备**键控
—— staging 与 pinned 探针用 `std::map<int, …>`，smem 开关改用仓库既有的 `ensure_func_attr_per_device`
（`kernel_attr_once.h`），设备切换用 `ops::detail::CurrentDeviceScope`。

**验证**（修复后，TP2、int8 KV，每格全新进程）：chunk=256 → `Let` 8/8（首版）+ 4/4（重构版）；
chunk=8192（生产默认）→ `Let` 4/4 + 3/3。单独关掉该路线（`NINFER_VOLTA_SPLITD=0`）同样 `Let` 4/4，
与修复后一致。**BFCL**（limit 20）默认 chunk=8192：修复后两次全新进程均为 **7/20 = 35%**，且
**通过集合完全相同** `[2,4,6,12,13,14,20]` ⇒ 非确定已消除（这是本次修复的目标）。

**遗留（真实且巨大；非 chunk 尺寸所致）**：改用**全量** BFCL `multi_turn_base`（200 条、逐样本配对、同一序）重测：

| config（修复后，每格全新进程） | acc |
|---|---|
| ninfer 默认（chunk 8192） | 41/200 = 20.5% |
| ninfer chunk 512 | 50/200 = 25.0% |
| ninfer chunk 128 | 46/200 = 23.0% |
| 1cat vLLM（APC 开、temp=1.0） | 77/200 = 38.5% |

> **更新（2026-10-27）**：SM70 修复后的最新全量为 **54/200 = 27.0%**，仍低于 1cat
> 38.5%；这是当前最高记录，但只比此前最佳 25.0% 高 2pp。当前状态、已采取措施和社区
> 请求见 [`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md)。上表是竞态修复后的历史
> 对照，不应作为当前分数。

配对 exact McNemar：chunk 128 vs 8192 → only-128=13 / only-8192=8，**p=0.38**；chunk 512 vs 8192 → 18/9，**p=0.12**；
chunk 128 或 8192 vs 1cat → **only-1cat=39/40，p<1e-5**。⇒ **竞态修复并未关掉质量差距，且 chunk 尺寸无显著效应**
（limit-20 的"50% vs 35%"是噪声）。**残余主因不在预填充 chunk**；嫌疑转向：KV dtype（1cat `fp8_e5m2` vs 本引擎 int8/g64）、
fork 自有的 Volta 路由选择（`NINFER_SM70_PREFILL_GEMM` cuBLAS 预填充 GEMM / Split-D / decode attn v2 / nvfp4 dispatch）、
以及 fork 的 all-NVFP4 转换（`6f8f60fa`）。另：**最早一次匹配参数测试（2026-10-16，n=80，thinking 开）即为 18.75% vs 38.75%**，
且两次 harness 都设了同一组快速路由 ⇒ 该差距**早于本文件所有修复，非本会话改动引入**。

此前 §11 基于 limit-20 的"无可分辨差距 (p=0.69)"是子集假象（那 20 题偏易：35% vs 全量 20.5%），
**limit-20 的 chunk 扫描结论全部作废**（全量重做见上表）。方法论教训：质量 A/B 必须用全量 200 条，
`--limit 20` 既高估分数又看不出 18 点的差距。见调查记录 Session 4 后续(3)(4)。

**之前的两条错误结论（已作废）**：
- ~~"注意力内核 splitd 与抖动无关"~~ —— 当时用 `env NINFER_VOLTA_SPLITD=0` 测试，但 harness 脚本**硬设**
  `export NINFER_VOLTA_SPLITD=1` 把开关覆盖了，实验从未真正关闭该路线（脚本已改为 `${NINFER_VOLTA_SPLITD:-1}`）。
- ~~"输入投影 CUTLASS 路由是决定因素"~~ —— 那些 env（`CUTLASS_MIN_T`/`INPUT_CAP`/`FORCE_QPN`/`SM70_PREFILL_GEMM`）
  确实生效，但它们只是**改变了时序**，让这个时序相关的竞态不再触发；与 arena 容量/地址、GEMM 后端无关。

**规避**：竞态本身不再需要规避（`--prefill-chunk 128` 的 3.3× 冷预填充代价仅在此前必要）。
全量已验证 chunk 尺寸无显著效应，故 `--prefill-chunk 128` **不再**作为质量选项（只会徒增 3.3× 冷预填充开销）。

**诊断开关**（dev 已有）：`NINFER_VOLTA_SPLITD`（0=关闭该路线）、`NINFER_ATTN_SYNC` / `NINFER_ATTN_SLEEP`
（attention mixer 各阶段流同步/停顿）、`NINFER_STREAM_TRACE` / `NINFER_TENSOR_TRACE` / `NINFER_DUMP_ATTN`、
`NINFER_DUMP_LAYERS`、`NINFER_VOLTA_CUTLASS_MIN_T`、`NINFER_VOLTA_INPUT_CAP`、`NINFER_NVFP4_FORCE_QPN`、
`NINFER_VOLTA_MMA_SPLITS`、`NINFER_SM70_PREFILL_GEMM`。

**路由数值等价性（2026-10-19 验证）**：单请求逐层残差 dump（`NINFER_DUMP_LAYERS`，同一配置跑两次
逐位相同，故差异是因果的）显示：每条 fork 自有路由的单算子差异仅 **1e-4~2e-3 relative L2**（≈1 个
BF16 ULP），但到第 63 层被放大到 **0.15–0.50**；`NINFER_SM70_ATTN_V2` 对 prefill 完全无影响
（decode-only）；Split-D 从 layer 3（第一个 full-attention 层，0-2 为 GDN）才开始有差异。
⇒ 没有可“修正”的算术 bug，数值对齐只能靠选用与上游相同的路由（把亚 ULP 差异归零）。
`NINFER_QUALITY_PARITY=1`（`tools/v100/ninfer-quality-off.sh`）一键切到上游 842078a1 的内核选择
（Split-D / ATTENTION_V2 / cuBLAS 预填充 / cutlass 输入投影全关，QPN 默认）；细节见
`bench-results/ninfer-quality-investigation.md`。

**复核 TODO**：
- [x] 定位并修复竞态根因。
- [ ] 同类审计：全仓 `static` 却持有按设备状态之处（单主机线程驱动多 rank 的约定）——首轮扫描已排除
  `vision_attention_volta_flash.cu` / `gqa_attention_volta_flash.cu`（按设备 `std::map` 键控 ✓）。

---

## 复核优先级建议

| 优先级 | 项 | 阻塞什么 |
|---|---|---|
| ~~P0~~ | ~~#2 paged KV trim bug~~ | ✅ 已修复 |
| P1 | #1 vision 架构 | 多模态能力 |
| ~~P1~~ | ~~#11 NVFP4 预填充数值缺陷~~ | ✅ 已修复（Split-D staging 跨 rank 竞态，`486ba771`/`f86b63ad`） |
| P2 | #6 Hadamard 精度 | 长期质量风险 |
| P2 | #10 工具调用解析器严格性（vs qwen3_coder） | agent 任务鲁棒性（非标准输出会丢工具调用） |
| P3 | #3/#4/#5 构建规范化 | 可移植性 |
