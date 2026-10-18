# 临时改动 / 绕过 复核清单

> 生成时间：2026-10-14
> 用途：所有非正式修复、临时绕过、降级配置的记录，后续逐一复核处理。

---

## 1. 模型加载：跳过 vision 绑定（架构代差）

**文件**：`src/targets/qwen3_6_27b/impl/load/bindings.cpp`

**改动**：`!features.vision` 时不再调用 `bind_vision_backbone`，改用 `validate_unconsumed_matching("vision/")` 忽略所有 vision tensor。

**原因**：neroued 官方制品 `Qwen3.8-27B-nvfp4-NInfer` 的 vision tower 与本 fork 不兼容：
- 制品：q6/q5/q4/q8 group-wise 量化、RoPE attention、**无独立 qkv 投影**
- fork 契约：原生 FP16 linear + 传统 qkv 结构

**影响**：文本推理正常；`--vision` 多模态不可用（加载仍会走 vision 分支并失败）。

**复核 TODO**：
- [ ] 要么移植新 vision 架构（RoPE attention、无 qkv、量化 linear 路径）
- [ ] 要么找匹配的老版本模型制品
- [ ] `--vision` 开关在 text-only 制品上的行为需要定义（报错 vs 警告降级）

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

## 11. NVFP4 预填充数值缺陷（`--prefill-chunk` ≥256 命中输入投影 CUTLASS 路径）

**文件**：`src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp` + `.../nvfp4_attn_input_cutlass_sm70.cu`；`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp` + `.../nvfp4_gdn_input_cutlass_sm70.cu`；共用 `src/ops/linear/nvfp4/nvfp4_cutlass_sm70.cu`。

**现象**：BFCL 多轮默认配置分数 20–35% 且**每个新进程结果不同**（`<tool_call>` / 自然语言在首 token 处抖动）；对照 1cat 同口径 45%。

**定位（2026-10-18）**：
- 首 token 由**最后一个 prefill chunk 末位置**的 logits 决定，故 `--prefill-chunk` 的 128↔256 边界直接改变数值。
- **未排除的混淆因素（重要）**：`NINFER_VOLTA_CUTLASS_MIN_T` **同时改变了输入投影的 workspace 容量**（`nvfp4_{attn,gdn}_input_plan.cpp` 的容量函数按同一阈值分支），即"输入投影后端"与"arena 容量/地址"两个变量尚未解耦。其余配置（`NINFER_NVFP4_FORCE_QPN`、`NINFER_SM70_PREFILL_GEMM=0`）只改后端、不改容量且仍抖，故与两种解释都相容。
- 已通读输入投影 CUTLASS 全路径（`split_attn_output` 下标、几何 `kOutputRows == w.n`（7168/8192）、dequant 覆盖 `n*k`、每 rank 独立 arena + 独立流）**未发现结构缺陷**，因此更倾向于**容量/地址相关**而非后端本身。
- chunk=256 首 token 配置矩阵（每格 3 个全新进程）：**只有把输入投影（attn+gdn）从 CUTLASS 切到 QPN 才变稳**（`NINFER_VOLTA_CUTLASS_MIN_T=1000000` 单独即稳 3/3；`NINFER_NVFP4_FORCE_QPN`/`NINFER_SM70_PREFILL_GEMM=0` 仍抖）；MLP GEMM 与注意力内核（`NINFER_VOLTA_SPLITD`、`NINFER_SM70_ATTN_V2`）与抖动**无关**。
- **时序敏感**（非单纯读未初始化内存）：开 `NINFER_DUMP_LAYERS`（每层 `cudaStreamSynchronize`+D2H）后 chunk=256 默认配置 5/5 `<tool_call>`；不开 dump 同 prompt 4 次里 1 次 `I`。两个同配置进程的 dump 前 4746 行一致、之后差 ~3e-3（top-1 未翻转）。
- 静态复核未发现越界/别名：QPN dequant 恰好覆盖 `n*k`；各 shard `kOutputRows == w.n`（attn 7168 / gdn 8192）；arena 超额抛 `std::bad_alloc` 而非静默重叠。可疑点在 **`DeviceArena::Scope::~Scope` 在 host 侧立即回收 bump 指针且不同步流**（`src/core/arena.cu:125-127`），正确性完全依赖单流有序，而 runtime 使用**每 rank `cudaStreamNonBlocking` 流**（`stream_for(rank)`）并行执行两个 rank。

**规避**：`--prefill-chunk 128`（结果确定，`multi_turn_base` 恢复到 50% = 1cat 水平）。代价：长提示冷预填充约 **3.3× 慢**（5060 tok TTFT 2.4s→8.0s；21620 tok 10.8s→34.6s），≤128 tok 无差异，BFCL 整轮因前缀缓存仅 +约 2min。

**诊断开关**（dev 已提供）：`NINFER_VOLTA_CUTLASS_MIN_T`（输入投影 cutlass→QPN 阈值）、`NINFER_NVFP4_FORCE_QPN`、`NINFER_VOLTA_MMA_SPLITS`、`NINFER_SM70_PREFILL_GEMM`、`NINFER_DUMP_LAYERS`。

**复核 TODO**：
- [ ] 修 T≥256 的输入投影 CUTLASS 路径：候选（a）把 arena 作用域延长到覆盖异步内核、（b）该路径加一次流同步、（c）直接把该路径切到 QPN（已验证稳定但不恢复质量）。
- [ ] 修好后以默认 chunk 复测，确认「稳且达 50%」后再决定是否保留 `--prefill-chunk 128`。

---

## 复核优先级建议

| 优先级 | 项 | 阻塞什么 |
|---|---|---|
| ~~P0~~ | ~~#2 paged KV trim bug~~ | ✅ 已修复 |
| P1 | #1 vision 架构 | 多模态能力 |
| P1 | #11 NVFP4 预填充数值缺陷（`--prefill-chunk` ≥256） | 默认配置的下游质量与可复现性 |
| P2 | #6 Hadamard 精度 | 长期质量风险 |
| P2 | #10 工具调用解析器严格性（vs qwen3_coder） | agent 任务鲁棒性（非标准输出会丢工具调用） |
| P3 | #3/#4/#5 构建规范化 | 可移植性 |
