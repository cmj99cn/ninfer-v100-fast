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

## 复核优先级建议

| 优先级 | 项 | 阻塞什么 |
|---|---|---|
| ~~P0~~ | ~~#2 paged KV trim bug~~ | ✅ 已修复 |
| P1 | #1 vision 架构 | 多模态能力 |
| P2 | #6 Hadamard 精度 | 长期质量风险 |
| P3 | #3/#4/#5 构建规范化 | 可移植性 |
