[English](#english)

# ninfer-v100-fast

基于 NInfer 的 Tesla V100 推理引擎：单进程 TP2（TP4 为 4×16GB legacy 回归）、NVLink 加速、多轮前缀缓存、长上下文 prefill 优化与 MTP 加速解码，目标是在 2×V100-SXM2-32GB 上把 Qwen3.8-27B NVFP4 的吞吐与延迟做到当前硬件的极限。

---

## 项目声明（请先读）

1. **本仓库全部代码与文档由 AI 编写生成。** 仓库所有者不具备大模型技术背景，不理解模型算法与推理引擎内部实现。**请不要向所有者提问任何技术问题**——所有者没有能力回答，也没有时间回复。
2. **问题、复现和社区协作通过 GitHub Issues/Discussions 与仓库文档进行。** 欢迎提交可复现的质量问题、logits 证据或转换校验；请附带步骤、日志、硬件/软件版本和环境信息。本项目不承诺修复周期或 SLA。
3. 本仓库是**全新项目**：以干净历史导入改进后的 NInfer 代码，不携带其他仓库的提交历史。许可与归属信息见 `NOTICE`。

---

## 发布版本

首个公开版本：**v0.1.0（2026-10-27）**。版本说明与已知限制见 [`CHANGELOG.md`](CHANGELOG.md)。

---

## 已知未解决问题：BFCL 多轮工具调用质量

本引擎最新全量结果为 **27.0%（54/200）**，是目前全量 200 的最高记录（此前最佳 25.0%），
但只比此前最佳高 2pp；与 c512 相比为 +14 / -10，净增 4 条、24 条翻转，单 seed 不能支撑
6pp 级归因。因此这是“没有回归、略有改善”，不是质量问题已解决。

我们已经做了全量 200 条多 seed、prompt 回退与 tokenizer 对齐、top_k 全词表、KV int8/bf16、
prefill chunk 128/512/8192、tool parser 严格性、不同转换路径制品对比，以及 SM70 数值/测试
修复；这些工作排除了大部分低成本假设，但没有解释剩余问题。目前**暂停盲目的主动修复**，
保留可复现实验、失败样本和证据采集入口；最值得查的是自转换 `.ninfer` 制品、NVFP4/attention
数值和 prefix-cache 状态。**公开请求社区协作定位质量问题**，具体要求见下。

- 问题说明、已采取措施、剩余假设、社区请求：[`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md)
- 详细复现、完整实验时间线、踩过的坑：[`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)

---

## 项目目标

在 2×Tesla V100-SXM2-32GB（NVLink）上，用**同一个引擎**同时满足：

| 目标 | 说明 |
|---|---|
| **TP2 / NVLink** | 单进程 TP2 张量并行，不依赖多进程代理；NVLink 上的轻量 all-reduce（mailbox）自动探测启用。TP4 仅保留为 4×16GB legacy 回归 |
| **缓冲命中做到极致** | `--prefix-cache` 内容寻址前缀缓存，多轮对话复用历史 KV，不重复 prefill |
| **预填充速度做到极致** | 长文档 splitd prefill 内核（174K 上下文实测 +7.5%） |
| **解码速度做到极致** | MTP 投机解码（`--spec mtp`）+ SM70 INT8 key-split 注意力内核 |

---

## 开发思路

1. **选基底**：选择带**单进程 TP2/TP4** 的 V100 NInfer 基底——多进程双卡方案架构脆弱、无法扩展到 TP4，被直接放弃；TP4 现仅保留为 4×16GB legacy 回归。
2. **移植长上下文 prefill**：引入 **splitd prefill 内核**（D256 Split-D）（geometry 映射、去掉 Hadamard 旋转、增加 `NINFER_VOLTA_SPLITD=0` 开关、`NINFER_SM70_SPLITKV3_MIN_KV` 阈值），实测 174K prefill +7.5%。
3. **修复前缀缓存**：生产化验证时发现 prefix cache 在多轮场景触发引擎崩溃（见下文问题 2），修复后 `--prefix-cache` 可长期稳定运行。
4. **速度验证与调参**：系统性分离 prefill / decode 测量，确认 MTP 是解码的主要杠杆；确认 `NINFER_SM70_LONG_SPLIT_KEYS` 类 attention 调参在本基底无效（瓶颈不在该处），避免无意义调优。
5. **质量评测**：跑 BFCL-v3 多轮函数调用评测，得到当前基线分数（见测试结果），后续迭代以该基线对照。

### 遇到的问题与解决思路

| # | 问题 | 解决 |
|---|---|---|
| 1 | 已有 NVFP4 制品的 Vision tower 与本 fork 契约不兼容（量化格式、无独立 qkv、RoPE attention），加载即崩 | 不使用该制品；当前自转换 NVFP4 制品包含 Vision，GGUF Q4_K_M 制品拒绝 Vision。多实例 Vision 未验证，见 `TEMP_WORKAROUNDS.md` |
| 2 | 开启 prefix cache 后多轮对话触发 `Paged KV trim extent would discard cache-owned pages`，之后所有请求 503，引擎不可用 | 两处修复：① `trim_pages` 遇到 cache-owned 页时收缩 trim 范围而不是抛异常；② `promote_page_at` 对已 cache-owned 的页幂等返回 |
| 3 | 怀疑 decode 慢是 attention split 策略问题，尝试 `LONG_SPLIT_KEYS=1920` | A/B 实测无效，确认瓶颈不在此，停止该方向调优 |
| 4 | decode 速度测量一直包含 prefill 时间，数字失真 | 改用 warmup + 多次迭代 + server 自报 timings 的方法分离测量 |
| 5 | 无 root 环境编译需要 ffmpeg/libcurl 头文件 | `apt-get download` + `dpkg-deb -x` 本地解包，`PKG_CONFIG_SYSROOT_DIR` 指向本地前缀；或直接 `-DNINFER_SKIP_FFMPEG=OFF` 见安装章节 |

全部临时改动与复核事项见 [`TEMP_WORKAROUNDS.md`](TEMP_WORKAROUNDS.md)。

---

## 当前进度（v0.1.0，截至 2026-10-27）

### 测试参数

| 引擎 | 参数 |
|---|---|
| ninfer（本引擎） | `--tp 2 --devices 0,1 --max-context 262144 --max-concurrency 2 --prefill-chunk 8192 --prefix-cache --kv-dtype int8 --no-thinking --greedy`，`NINFER_VOLTA_SPLITD=1` |

### 踩过的坑（已实测，避免重复）

- **stages=3 编译失败**：cutlass SM70 的 `DefaultGemm` 特化硬编码 2 stages，不可选。
- **swizzle 不可选**：`device::Gemm` 内核调用 `get_tile_offset(int)`，只有 identity swizzle 满足，horizontal 无法通过该 API 使用。
- **bench 方法论陷阱**：`speed_bench.py` 的 prefill 段各行共享前缀（部分缓存命中，非冷）；冷对比用 `tools/v100/bench_prefill_cold.py`（distinct seed）。
- **`#pragma unroll 4` 回退**：`__launch_bounds__(256,5)` 卡 ~51 regs/thread（实测 48）→ 溢出，27.27→29.86 回归，已回退。
- **attention 两轴已挖尽**：`NINFER_SM70_LONG_SPLITS_ONLY`（tier3 地板 129 splits）与 `NINFER_SM70_DECODE_MB`（mb5 最优）均达局部最优，splits-only 绕过地板全面更差。
- **`pkill -f` 自杀**：模式匹配到自身命令行会秒杀 shell，用 PID 或 bracket 技巧。
- **cuBLAS handle 必须按设备各持一个**：本引擎 TP2 单线程交替两卡调用（每次 `cudaSetDevice` 切换），单个 `thread_local` handle 会让部分 GEMM 报 `CUBLAS_STATUS_EXECUTION_FAILED`(13)。
- **dequant 缓冲布局**：`w_fp16` 是 out-major `[n][k]`（不是 `[k][n]`），`x_fp16` 是 token-major `[cols][k]`；cuBLAS 映射需 `transa=CUBLAS_OP_T, lda=k, transb=OP_N, ldb=k, ldc=n`，映射错会静默出垃圾结果（服务不崩，输出乱码）。
- **SM70 上 cuBLAS 不支持 bf16 输出**（`CUBLAS_STATUS_NOT_SUPPORTED`(15)）：bf16 输出走 fp16 GEMM + 转换 kernel。
- **strata 会抢占 GPU**：杀掉 strata engine 后其 serve 层（pid 2555）会在 GPU 释放后自动重启 engine 抢占显存；必须连 serve 层一起杀。
- **并发指标陷阱**：per-stream decode tok/s 的“合计”会惩罚早出首 token 的引擎（其解码期与另一流的 prefill 混批重叠）。并发对比应看**两流总 wall**，不能看每流 decode 均值之和。

---

## 测试环境

| 项 | 配置 |
|---|---|
| GPU | 2× Tesla V100-SXM2-32GB（SM70），NVLink 六链路 |
| CPU / 内存 | 96 GB RAM |
| 系统 | Linux x86_64，无 root 权限环境亦可构建（见安装） |
| CUDA | 12.8 |
| 编译器 | GCC 13.3，CMake ≥ 3.28，Ninja |
| 模型 | 自转换 `qwen3_8_27b_nvfp4.ninfer`（NVFP4，约 18 GiB；源 `unsloth/Qwen3.8-27B-NVFP4`） |
| 服务参数 | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## 测试用例与结果

以下全部为**本项目自身实测数据**（2026-10），不与其他引擎或 fork 对比。

### 速度（TP2，NVFP4，2×V100）

**Prefill**

| 用例 | 方法 | 结果 |
|---|---|---|
| 8K tokens 短 prompt | server 自报 | 最高约 **1780 tok/s** |
| 174K tokens 长文档（splitd 开启） | 专用 A/B 脚本，单请求 | **1059.7 tok/s** |
| 同上（splitd 关闭，对照） | 同上 | 985.4 tok/s（开启 splitd 相对提升 **+7.5%**） |

**Decode（生成 512–1024 tokens）**

对齐 profile，`long_gen_prompt` 四点（prompt 502 / 3.7K / 15K / 59K），`max_tokens=128`：

| 配置 | @502 | @3.7K | @15K | @59K |
|---|---|---|---|---|
| 无 MTP | 50.9 | 50.0 | 46.2 | 35.9 |
| **MTP（`--spec mtp --draft-tokens 3`）** | **106.8** | **99.2** | **99.7** | **84.4** |

MTP 为各点 **2.0–2.3×**，接受率 47–58%（2.4–2.7 tok/round）；TTFT 增加约 0.15s（draft 初始化）。

**多轮前缀缓存（`--prefix-cache`）**

15 轮长上下文压力测试（8K tokens 起步、每轮追加代码块至约 39K）：全部成功，server 日志确认持续命中 `reuse=block_snapshot`，末轮缓存命中约 36800 tokens，命中后 prefill 仍保持 1200–1600 tok/s。

### 质量（BFCL 多轮工具调用，未解决，公开请求社区协作）

**最新全量结果（2026-10-26/27）：54/200 = 27.0%**。这是目前全量 200 的最高记录
（此前最佳 25.0%），但只高 2pp；与 c512 逐样本相比为 +14 / -10，净增 4 条、24 条翻转。
单 seed 不能支撑 6pp 级归因，因此结论是“没有回归、略有改善”，不是“质量问题已解决”。

本次运行的设置：BFCL-v3 `multi_turn_base` 全量 200 条，thinking OFF，
`temperature=1.0`、`top_p=1.0`、`top_k=20`、`seed=42`，TP2，NVFP4 制品，
`--prefix-cache` 开启。实际发出 2404 个请求，耗时 5959 秒；平均延迟 2.471s，
TTFT 804.9ms，TPOT 13.8ms，吞吐 38.91 tok/s。结果目录：
`/home/cmj/eval-runs/bfcl/ninfer_sm70fix_s42_n200/20261026_094605`。

| 运行 | 分数 | 说明 |
|---|---:|---|
| ninfer，最新全量 | **27.0%（54/200）** | 2026-10-26/27，seed 42 |
| ninfer，此前最佳（chunk 512） | 25.0%（50/200） | 历史最佳 |
| ninfer，默认（chunk 8192，int8 KV） | 20.5%（41/200） | 历史默认 |
| ninfer，chunk 128 | 23.0%（46/200） | 无显著差异 |
| ninfer，bf16 KV | 23.0%（46/200） | 无显著差异 |

与 c512 对比：两边都过 40 条，本次新过 14 条、新丢 10 条。

我们为提升质量做过的工作：

1. **Prompt/tokenizer 对齐（`ce64d89a`）**：修复 OpenAI/Anthropic schema 的键序和
   注入的 `"strict": false`，使多轮历史逐 token id 稳定。
2. **采样对齐**：固定 `top_k=20, top_p=0.95, temperature=1.0`；显式 top_k=20、top_k=0
   与默认的差异不能解释问题。
3. **Prefill/KV 对照**：chunk 128/512/8192、KV int8/bf16 在全量 200 下均无显著差异
   （p=0.12/0.38/0.40）；`--limit 20` 的高分是小样本噪声。
4. **Tool parser 复核**：malformed tool call 离线恢复 0/9，不能解释主要失败样本。
5. **SM70 数值与测试修复**：修复 cuBLAS BF16 输出经 FP16 暂存的溢出（改为 FP32
   暂存），补齐 A4/A8 能力门控、Q4/Q5 workspace、零容量 arena 等；全量 ctest
   **135/135 通过**，15 个硬件/产物门控跳过。
6. **Prefix cache 与多实例**：完成共享 host tier、跨实例 manifest、2×TP1 真实测试；
   快速质量 A/B 的 cold/warm 文本与 MTP token 一致，未观察到回归。

**仍未排除，且最值得查的：自转换 `.ninfer` 制品。** 本引擎先把 HF safetensors 转换成
18 GiB 的 `.ninfer`（embedding encoder、NVFP4/FP8 布局、norm/lm_head 的 dtype 都可能
变化）。其次是 NVFP4 GEMM/attention 数值、prefix-cache 状态，以及失败样本的具体形态。
没有 logits/第一分叉 token 证据前，不应盲改 kernel 或继续扫参数。

**公开请求社区协作：** 如果你能复现，请提交 GitHub Issue/Discussion，并附以下证据：

- 同一 BFCL 样本的完整 prompt token IDs；
- 首轮 prefill 最后位置的 top-20 logits/probs，以及第一个分叉 token；
- 同一请求开/关 `--prefix-cache` 的第一分叉点；
- 不同转换路径的同层输出；
- 逐样本失败分类：格式错 / 函数名错 / 参数错 / 多轮状态错。

请不要只报“分数低”，不要用 `--limit 20` 下结论，也不要提交盲目的 kernel sweep。
完整清单、复现命令和已排除项见
[`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md) 与
[`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)。

> **旧 25% 单点不要追**：它来自已被覆盖的旧 binary，prompt A/B 已排除 prompt 原因；
> 用它做目标会重复我们走过的弯路。

`--prefill-chunk 128` 的代价：长提示冷预填充约 **3.3× 慢**（5060 tok TTFT 2.4s→8.0s；21620 tok 10.8s→34.6s）；≤128 tok 无差异；BFCL 整轮因前缀缓存仅约 +2min。

**历史基线（greedy，thinking 关闭，`--prefix-cache` 开启）**：

| Subset | 分数 |
|---|---:|
| multi_turn_base | 20% |
| multi_turn_miss_func | 10% |
| multi_turn_miss_param | 35% |
| multi_turn_long_context | 15% |
| **OVERALL（80 条）** | **20%** |

---

## 模型下载

本项目默认使用**自转换**的 NVFP4 `.ninfer` 制品（约 18 GiB）：以公开的 `unsloth/Qwen3.8-27B-NVFP4` 量化权重为源，套用官方 `Qwen/Qwen3.8-27B` 的结构与词表，经本仓库转换流水线 `tools/convert/qwen3_8_27b/convert_nvfp4.py` 生成。制品旁的 `qwen3_8_27b_nvfp4.ninfer.conversion.json` 记录了源仓库、revision 与转换参数（本制品固定：结构源 `1d4bf0f2…`、量化源 `60e813d4…`），可作溯源与校验。

两个来源均公开可下载，建议带 `--revision` 固定版本以复现：

```bash
# 国内环境建议使用镜像，并禁用 Xet 传输（否则大文件下载会失败）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 官方结构 / 词表 / chat 模板
huggingface-cli download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ./models/Qwen3.8-27B

# NVFP4 量化权重
huggingface-cli download unsloth/Qwen3.8-27B-NVFP4 \
  --revision 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 \
  --local-dir ./models/unsloth-qwen3.8-27b-nvfp4
```

转换（工具随 dev 分支维护，用法见 `tools/convert/qwen3_8_27b/convert_nvfp4.py` 顶部说明与 `--help`）：

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --model ./models/Qwen3.8-27B \
  --quantized-model ./models/unsloth-qwen3.8-27b-nvfp4 \
  --out out/qwen3_8_27b_nvfp4.ninfer
```

> 注意：
> - 已有 NVFP4 制品与本 fork 的 GDN `query_key_value_z` 加载契约不兼容（加载即报 tensor descriptor 契约错误），**请不要使用**。
> - 该制品为 text-only 路线，Vision tower 契约不兼容（见问题 1），请不要使用它；当前自转换 NVFP4 制品包含 Vision，GGUF Q4_K_M 制品拒绝 `--vision`。

---

## 安装

### 依赖

- NVIDIA CUDA Toolkit **12.8**（含 `nvcc`，支持 `sm_70`）
- CMake **≥ 3.28**、Ninja、GCC ≥ 13
- NCCL（`libnccl.so` + `nccl.h`，TP 必需）
- （可选）FFmpeg / libcurl 开发包——仅多媒体输入需要；纯文本服务可跳过

CUTLASS 由 CMake FetchContent 自动拉取，无需手动安装。

### 构建

```bash
git clone https://github.com/cmj99cn/ninfer-v100-fast.git
cd ninfer-v100-fast
git checkout dev

# 纯文本服务（推荐，依赖最少）
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_SKIP_FFMPEG=ON

cmake --build build -j"$(nproc)" --target ninfer-serve
```

产物：`build/apps/ninfer-serve`。

若需要多媒体（图片/音频/视频输入），去掉 `-DNINFER_SKIP_FFMPEG=ON`，并安装 FFmpeg 与 libcurl 开发包（`libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev` 等，或自行用 `apt-get download` + `dpkg-deb -x` 解包到本地前缀后设置 `PKG_CONFIG_SYSROOT_DIR` / `PKG_CONFIG_PATH`——本项目开发环境即采用此方式，细节见 `TEMP_WORKAROUNDS.md`）。

运行时若 NCCL 由 pip 包提供，需让链接器能找到它：

```bash
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH
```

---

## 使用

### 启动服务

```bash
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH   # 视你的 NCCL 位置

./build/apps/ninfer-serve \
  /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --model-id qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 --port 9010 \
  --api-key YOUR_API_KEY \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 \
  --max-context 262144 \
  --prefix-cache \
  --no-thinking --greedy
```

TP4（四卡 16GB legacy 回归）把 `--tp 2 --devices 0,1` 换成 `--tp 4 --devices 0,1,2,3`；当前 2×32GB 发布路径使用 TP2。

#### 多实例分组（部分验证）

同一进程可以启动多个 Engine，各占一组互不重叠的卡，共享一份只读 artifact mmap 与一份 pinned
host prefix-KV tier；device 权重与 device KV 仍每实例私有。host tier 只保存 prefix-cache 的
完整 64-token block 和 snapshot，不是所有活跃请求的 KV；没有磁盘 KV tier 或 KV offload。
前端只有一个 OpenAI/Anthropic 入口，`EnginePool` 按前缀命中、空闲容量和 aging 路由到实例，
不是简单轮询。`--group-devices` 用分号分组；省略 `--tp` 时由组宽推断。每实例的
`--max-concurrency` 与 `--kv-capacity` 独立配置，`--global-pending-requests` 是共享等待队列
上限（默认 16）。

| 卡数 | 组数 × TP | `--group-devices` | 状态 |
|---:|---|---|---|
| 2 | 2×TP1 | `"0;1"` | **真实集成测试已通过** |
| 4 | 4×TP1 | `"0;1;2;3"` | 未验证 |
| 4 | 2×TP2 | `"0,1;2,3"` | 未验证 |
| 8 | 8×TP1 | `"0;1;2;3;4;5;6;7"` | 社区/非契约；未验证 |
| 8 | 4×TP2 | `"0,1;2,3;4,5;6,7"` | 社区/非契约；未验证 |
| 8 | 2×TP4 | `"0,1,2,3;4,5,6,7"` | 社区/非契约；未验证 |

多组建议用 `--group-devices` 明确分组；`--devices` 也可按组顺序列出全部 rank，但 `ninfer`
CLI 最多接受 4 个 ID。多组模式不支持 `--spec dflash`。

2 卡本机（2×TP1；真实集成测试已通过）：

```bash
./build/apps/ninfer-serve /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --groups 2 --group-devices "0;1" --tp 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --prefix-cache --host-cache-mib 1024
```

8 卡 8×TP1（未验证；8 个完整实例，显存/内存压力最大）：

```bash
./build/apps/ninfer-serve /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --groups 8 --group-devices "0;1;2;3;4;5;6;7" --tp 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --prefix-cache --host-cache-mib 153600
```

对应脚本 profile（社区/非契约）：`bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1|4xTP2|2xTP4`
（加 `--dry-run` 只打印完整命令）。8 卡 4×TP2 / 2×TP4 以及 4 卡 4×TP1 / 2×TP2 的完整参数、
路由和验证边界见 [`docs/maintainer/multi-instance.md`](docs/maintainer/multi-instance.md)。
上述未验证 profile 没有吞吐或显存实测数据，不得引用为验收结果。

**社区测试请求（8 卡）：** 请在 8×V100-SXM2-32GB 上优先测试 8×TP1、4×TP2、2×TP4，并提交
启动命令、binary/artifact hash、`/metrics?format=json` 的 `instances` 数组、每实例 committed
decode tok/s、TTFT/TPOT、MTP acceptance、queue wait、free KV pages、host tier 实际分配量，以及
4×TP2/2×TP4 的 32GB 显存余量。4×TP1 与 8×TP1 都需要 32GB 卡；16GB V100 放不下一份完整的 27B 权重。
4 卡多实例（4×TP1/2×TP2）也未验证，欢迎一起提交。`--dry-run` 或静态检查不等于 8 卡验证；
本项目不承诺修复周期或 SLA。

### 关键参数

| 参数 | 说明 |
|---|---|
| `--tp 2\|4` + `--devices` | 张量并行度与设备列表（单进程） |
| `--prefix-cache` | 内容寻址前缀缓存（多轮对话强烈建议开启） |
| `--spec mtp --draft-tokens 3` | MTP 投机解码，解码吞吐显著提升 |
| `--kv-dtype int8` | INT8 KV cache（默认推荐） |
| `--max-context` | 最大上下文（本制品原生 262144） |
| `--no-thinking` / `--greedy` | 关闭思考模式、贪婪采样（评测常用） |

完整参数：`./build/apps/ninfer-serve --help`。参考启动脚本：`tools/v100/ninfer-aligned-tp2.sh`（对齐 A/B profile）、`tools/v100/ninfer-quality-matched.sh`（thinking/采样对齐）。

### 调用 API（OpenAI 兼容）

```bash
curl http://127.0.0.1:9010/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-nvfp4",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'
```

兼容 `/v1/chat/completions`（含 `tools` 函数调用）与 `/v1/models`。健康检查：`GET /health`。

### 环境变量（调优/排障）

| 变量 | 说明 |
|---|---|
| `NINFER_VOLTA_SPLITD=0` | 关闭 splitd prefill 内核（默认开启） |
| `NINFER_SM70_SPLITKV3_MIN_KV` | split-KV3 阈值（默认 2048） |
| `NINFER_SM70_ATTN_V2=0` | 回退到 phase-1 decode 注意力内核（默认用 v2） |
| `NINFER_SM70_LONG_SPLIT_KEYS` | 长窗口每 split key 数（本基底实测无效，保留作 A/B） |
| `NINFER_SM70_LONG_SPLITS_ONLY` | 绕过 tier 地板强制 long-split 策略（A/B 实测全面更差，保留作对照） |
| `NINFER_SM70_DECODE_MB` | T1-3 >6144 窗口 MinBlocks（默认 5；2/4/6 均更差） |
| `NINFER_SM70_PREFILL_GEMM` | 冷 prefill GEMM 路由：0=CUTLASS identity，1=cuBLAS（+20~23% 冷 prefill tps；`tools/v100/ninfer-aligned-tp2.sh` 默认 1） |
| `NINFER_SM70_GEMM_TRACE` | 设定后打印 prefill cuBLAS 调用维度/设备/线程诊断（排障用） |

---

## 已知限制

- **Vision**：当前自转换 NVFP4 制品包含 Vision；GGUF Q4_K_M 制品拒绝 `--vision`；多实例 Vision 未验证。
- 多实例只有 2×TP1 通过真实集成测试；4 卡（4×TP1/2×TP2）和 8 卡（8×TP1/4×TP2/2×TP4）未验证，8 卡为社区/非契约。4×TP1 与 8×TP1 需要 32GB 卡，16GB V100 放不下一份完整的 27B 权重。
- KV 只有每实例 device KV 和进程共享的 pinned host prefix-KV 两级；没有磁盘 KV tier、KV offload 或全局活跃 KV 池。
- prefix cache 修复后未做长时间浸泡测试；超高压并发下的缓存行为仍在观察。
- **BFCL 全量 200 条：ninfer 最新 27.0%（54/200）**。**该问题仍未解决，暂停盲目主动修复，公开请求社区协作**；已排除 prompt/top_k/chunk/KV/parser，剩余首要嫌疑是自转换 `.ninfer` 制品、NVFP4/attention 数值和 prefix-cache 状态（见“质量”节与 `BFCL_QUALITY_ISSUE.md`）。

---

## 许可协议

本仓库代码在适用许可条款下分发，**继承 Apache License 2.0**，完整声明见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。模型制品的许可以其发布仓库（Hugging Face）为准。

---

*本 README 及仓库内全部代码、文档由 AI 生成与维护（见「项目声明」）。*

---

# English

[中文](#ninfer-v100-fast)

A Tesla V100 inference engine based on NInfer: single-process TP2 (TP4 remains a 4×16GB legacy regression), NVLink acceleration, multi-turn prefix caching, long-context prefill optimization, and MTP-accelerated decoding. The goal is to push Qwen3.8-27B NVFP4 throughput and latency to the limit of 2×V100-SXM2-32GB hardware.

---

## Project Statement (read first)

1. **All code and documentation in this repository are generated by AI.** The repository owner has no large-model technical background and does not understand the model algorithms or the internals of the inference engine. **Please do not ask the owner technical questions**—the owner cannot answer them and has no time to reply.
2. **Problems, reproductions, and community collaboration go through GitHub Issues/Discussions and the repository documents.** Reproducible quality reports, logits evidence, and conversion checks are welcome; include steps, logs, hardware/software versions, and environment details. No fix timeline or SLA is promised.
3. This repository is a **new project**: improved NInfer code is imported with a clean history, without other repositories' commit history. See `NOTICE` for license and attribution information.

---

## Release

First public release: **v0.1.0 (2026-10-27)**. See [`CHANGELOG.md`](CHANGELOG.md) for release notes and known limitations.

---

## Known Unresolved Issue: BFCL Multi-Turn Tool-Calling Quality

This engine's latest full run scores **27.0% (54/200)**. This is the highest full-200
result so far (previous best 25.0%), but only 2pp above it; against c512 the per-sample
result is +14 / -10, net +4 with 24 flips. A single seed cannot support 6pp attribution.
The result therefore means “no regression, modest improvement”, not “quality issue solved”.

We ran full 200-sample multi-seed runs, prompt rollback and tokenizer alignment,
full-vocabulary top_k, KV int8/bf16, prefill chunk 128/512/8192, parser strictness,
different conversion-path artifacts, and SM70 numerical/test fixes. These ruled out most
low-cost hypotheses but did not explain the remaining issue. Active blind fixing is
**paused**; reproducible experiments, failing samples, and an evidence collection path are
kept. The highest-value remaining suspects are the self-converted `.ninfer` artifact,
NVFP4/attention numerics, and prefix-cache state. **We are publicly asking the community
to help localize the quality issue**; see below.

- Problem statement, work done, remaining hypotheses, community request: [`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md)
- Detailed reproduction, full experiment timeline, pitfalls: [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md)

---

## Goals

On 2×Tesla V100-SXM2-32GB (NVLink), use **one engine** to satisfy all of the following:

| Goal | Description |
|---|---|
| **TP2 / NVLink** | Single-process TP2 tensor parallelism without multi-process proxies; lightweight NVLink all-reduce (mailbox) is auto-detected and enabled. TP4 remains a 4×16GB legacy regression |
| **Maximum cache hits** | `--prefix-cache` content-addressed prefix cache reuses historical KV across multi-turn conversations without re-prefill |
| **Maximum prefill speed** | Long-document splitd prefill kernels (+7.5% measured at 174K context) |
| **Maximum decode speed** | MTP speculative decoding (`--spec mtp`) + SM70 INT8 key-split attention kernels |

---

## Development Approach

1. **Choose the base**: we selected a V100 NInfer base with **single-process TP2/TP4**. Multi-process dual-GPU designs were fragile and could not scale to TP4, so they were rejected; TP4 now remains a 4×16GB legacy regression.
2. **Port long-context prefill**: add the **splitd prefill kernels** (D256 Split-D) (geometry mapping, removing the Hadamard rotation, adding the `NINFER_VOLTA_SPLITD=0` switch and the `NINFER_SM70_SPLITKV3_MIN_KV` threshold). Measured +7.5% prefill at 174K.
3. **Fix prefix caching**: during production validation, prefix caching triggered engine crashes in multi-turn scenarios (see problem 2 below). After the fixes, `--prefix-cache` runs stably for long periods.
4. **Speed validation and tuning**: systematically separate prefill and decode measurements; confirm that MTP is the main decode lever; confirm that `NINFER_SM70_LONG_SPLIT_KEYS`-style attention tuning is ineffective on this base (the bottleneck is elsewhere) and stop meaningless tuning.
5. **Quality evaluation**: run BFCL-v3 multi-turn function-calling evaluation to establish the current baseline (see test results); future iterations use this baseline for comparison.

### Problems and Solutions

| # | Problem | Solution |
|---|---|---|
| 1 | The Vision tower of an existing NVFP4 artifact is incompatible with this fork's contract (quantization format, no separate qkv, RoPE attention); loading crashes immediately | Do not use that artifact; the current self-converted NVFP4 artifact includes Vision, while the GGUF Q4_K_M artifact rejects Vision. Multi-instance Vision is unverified; see `TEMP_WORKAROUNDS.md` |
| 2 | With prefix cache enabled, multi-turn conversations trigger `Paged KV trim extent would discard cache-owned pages`; all subsequent requests return 503 and the engine becomes unusable | Two fixes: (1) when `trim_pages` encounters cache-owned pages, shrink the trim range instead of throwing; (2) `promote_page_at` returns idempotently for pages already owned by the cache |
| 3 | Suspected that slow decode was caused by the attention split strategy; tried `LONG_SPLIT_KEYS=1920` | A/B showed no effect; confirmed the bottleneck is elsewhere and stopped tuning this direction |
| 4 | Decode speed measurements always included prefill time, distorting the numbers | Switched to warmup + multiple iterations + server-reported timings to separate the measurements |
| 5 | Building without root requires ffmpeg/libcurl headers | Use `apt-get download` + `dpkg-deb -x` to unpack locally, and point `PKG_CONFIG_SYSROOT_DIR` at the local prefix; or use `-DNINFER_SKIP_FFMPEG=OFF` as described in the installation section |

See [`TEMP_WORKAROUNDS.md`](TEMP_WORKAROUNDS.md) for all temporary changes and review items.

---

## Current Status (v0.1.0, as of 2026-10-27)

### Test Parameters

| Engine | Parameters |
|---|---|
| ninfer (this engine) | `--tp 2 --devices 0,1 --max-context 262144 --max-concurrency 2 --prefill-chunk 8192 --prefix-cache --kv-dtype int8 --no-thinking --greedy`, `NINFER_VOLTA_SPLITD=1` |

### Pitfalls (measured; avoid repeating)

- **stages=3 build failure**: the `DefaultGemm` specialization in cutlass SM70 hardcodes 2 stages; it is not selectable.
- **swizzle is not selectable**: the `device::Gemm` kernel calls `get_tile_offset(int)`; only identity swizzle satisfies it, so horizontal swizzle cannot be used through this API.
- **Benchmark methodology trap**: the prefill rows in `speed_bench.py` share a prefix (partial cache hits, not cold). Use `tools/v100/bench_prefill_cold.py` (distinct seed) for cold comparisons.
- **`#pragma unroll 4` rollback**: `__launch_bounds__(256,5)` is capped at ~51 regs/thread (measured 48) → spill, regression from 27.27 to 29.86; rolled back.
- **Both attention axes are exhausted**: `NINFER_SM70_LONG_SPLITS_ONLY` (tier3 floor, 129 splits) and `NINFER_SM70_DECODE_MB` (mb5 optimal) are both at local optima; splits-only bypassing the floor is uniformly worse.
- **`pkill -f` suicide**: if the pattern matches your own command line, it kills the shell instantly; use PIDs or a bracket trick.
- **Each device must have its own cuBLAS handle**: this engine's TP2 single thread alternates between two GPUs (calling `cudaSetDevice` each time); a single `thread_local` handle makes some GEMMs report `CUBLAS_STATUS_EXECUTION_FAILED` (13).
- **Dequant buffer layout**: `w_fp16` is out-major `[n][k]` (not `[k][n]`), and `x_fp16` is token-major `[cols][k]`. The cuBLAS mapping must use `transa=CUBLAS_OP_T, lda=k, transb=OP_N, ldb=k, ldc=n`; a wrong mapping silently produces garbage (the server does not crash, but the output is gibberish).
- **cuBLAS on SM70 does not support bf16 output** (`CUBLAS_STATUS_NOT_SUPPORTED` (15)): bf16 output goes through fp16 GEMM plus a conversion kernel.
- **strata preempts the GPU**: after killing the strata engine, its serve layer (pid 2555) automatically restarts the engine and grabs VRAM after the GPU is freed; kill the serve layer too.
- **Concurrency metric trap**: summing per-stream decode tok/s penalizes the engine that emits its first token earlier (its decode overlaps with the other stream's prefill in a mixed batch). For concurrency comparisons, use **total wall time for both streams**, not the sum of per-stream decode averages.

---

## Test Environment

| Item | Configuration |
|---|---|
| GPU | 2× Tesla V100-SXM2-32GB (SM70), six NVLink links |
| CPU / RAM | 96 GB RAM |
| OS | Linux x86_64; buildable without root (see Installation) |
| CUDA | 12.8 |
| Compiler | GCC 13.3, CMake ≥ 3.28, Ninja |
| Model | Self-converted `qwen3_8_27b_nvfp4.ninfer` (NVFP4, ~18 GiB; source `unsloth/Qwen3.8-27B-NVFP4`) |
| Server flags | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## Test Cases and Results

All data below are **measured on this project** (October 2026) and are not comparisons against other engines or forks.

### Speed (TP2, NVFP4, 2×V100)

**Prefill**

| Case | Method | Result |
|---|---|---|
| 8K-token short prompt | server-reported | up to ~**1780 tok/s** |
| 174K-token long document (splitd on) | dedicated A/B script, single request | **1059.7 tok/s** |
| Same (splitd off, reference) | same | 985.4 tok/s (splitd gives **+7.5%**) |

**Decode (generating 512–1024 tokens)**

Aligned profile, `long_gen_prompt` at four points (prompt 502 / 3.7K / 15K / 59K), `max_tokens=128`:

| Configuration | @502 | @3.7K | @15K | @59K |
|---|---|---|---|---|
| No MTP | 50.9 | 50.0 | 46.2 | 35.9 |
| **MTP (`--spec mtp --draft-tokens 3`)** | **106.8** | **99.2** | **99.7** | **84.4** |

MTP is **2.0–2.3×** at every point, with 47–58% acceptance (2.4–2.7 tok/round); TTFT increases by about 0.15s (draft initialization).

**Multi-turn prefix cache (`--prefix-cache`)**

15-turn long-context stress test (starting at 8K tokens, appending a code block each turn up to ~39K): all turns succeeded. Server logs confirm continuous `reuse=block_snapshot` hits; the last turn hit ~36800 cached tokens, and prefill still sustained 1200–1600 tok/s after the hit.

### Quality (BFCL multi-turn tool calling, unresolved; community help requested)

**Latest full result (2026-10-26/27): 54/200 = 27.0%**. This is the highest full-200
result so far (previous best 25.0%), but only 2pp above it; against c512 the per-sample
result is +14 / -10, net +4 with 24 flips. A single seed cannot support 6pp attribution.
The result means “no regression, modest improvement”, not “quality issue solved”.

Run settings: BFCL-v3 `multi_turn_base`, full 200 samples, thinking OFF,
`temperature=1.0`, `top_p=1.0`, `top_k=20`, `seed=42`, TP2, NVFP4 artifact,
`--prefix-cache` enabled. Actual requests: 2404; wall time 5959s; average latency
2.471s, TTFT 804.9ms, TPOT 13.8ms, throughput 38.91 tok/s. Results:
`/home/cmj/eval-runs/bfcl/ninfer_sm70fix_s42_n200/20261026_094605`.

| Run | Score | Notes |
|---|---:|---|
| ninfer, latest full run | **27.0% (54/200)** | 2026-10-26/27, seed 42 |
| ninfer, previous best (chunk 512) | 25.0% (50/200) | historical best |
| ninfer, default (chunk 8192, int8 KV) | 20.5% (41/200) | historical default |
| ninfer, chunk 128 | 23.0% (46/200) | not significant |
| ninfer, bf16 KV | 23.0% (46/200) | not significant |

Against c512: both pass 40, new passes 14, new losses 10.

Work done to improve quality:

1. **Prompt/tokenizer alignment (`ce64d89a`)**: fixed the OpenAI/Anthropic schema key
   order and the injected `"strict": false`, making the multi-turn history stable
   token by token.
2. **Sampling alignment**: fixed `top_k=20, top_p=0.95, temperature=1.0`; explicit
   top_k=20, top_k=0, and default differences cannot explain the issue.
3. **Prefill/KV A/Bs**: chunk 128/512/8192 and KV int8/bf16 show no significant
   difference on the full 200 (p=0.12/0.38/0.40); `--limit 20` scores were small-sample noise.
4. **Tool-parser recheck**: offline recovery of malformed calls was 0/9 and cannot explain
   the main failures.
5. **SM70 numerical/test fixes**: fixed cuBLAS BF16 output staged through FP16 (now FP32),
   added A4/A8 capability gating, Q4/Q5 workspace, zero-capacity arena fixes, etc.; full
   ctest **135/135 passed**, 15 hardware/artifact-gated skips.
6. **Prefix cache and multi-instance**: shared host tier, cross-instance manifests, and
   2×TP1 real tests completed; quick cold/warm and MTP quality A/B showed no regression.

**Still open, and the most important suspect: the self-converted `.ninfer` artifact.**
This engine first converts HF safetensors into an 18 GiB `.ninfer` (embedding encoder,
NVFP4/FP8 layouts, and norm/lm_head dtypes can all change). Other candidates are NVFP4
GEMM/attention numerics, prefix-cache state, and the concrete failure shape of failing
samples. Do not blindly change kernels or keep sweeping parameters without logits /
first-divergence evidence.

**Public request for community help:** if you can reproduce this, please open a GitHub
Issue/Discussion with:

- full prompt token IDs for the same BFCL sample;
- top-20 logits/probs at the last prefill position and the first divergent token;
- the first divergence with `--prefix-cache` on vs off;
- same-layer outputs for different conversion paths;
- per-sample failure taxonomy: format / function name / arguments / multi-turn state.

Do not report only a low score, do not use `--limit 20` for conclusions, and do not submit
blind kernel sweeps. Full details, reproduction commands, and ruled-out items are in
[`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md) and
[`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md).

> **Do not chase the old 25% single point**: it came from an overwritten old binary, and the
> prompt A/B already ruled out the prompt. Using it as a target would repeat our detours.

Cost of `--prefill-chunk 128`: long-prompt cold prefill is about **3.3× slower**
(5060 tok TTFT 2.4s→8.0s; 21620 tok 10.8s→34.6s); ≤128 tok shows no difference; the full
BFCL run costs only about +2 min thanks to prefix caching.

**Historical baseline (greedy, thinking off, `--prefix-cache` enabled)**:

| Subset | Score |
|---|---:|
| multi_turn_base | 20% |
| multi_turn_miss_func | 10% |
| multi_turn_miss_param | 35% |
| multi_turn_long_context | 15% |
| **OVERALL (80 samples)** | **20%** |

---

## Model Download

This project uses a **self-converted** NVFP4 `.ninfer` artifact (~18 GiB) by default. It is
generated from the public `unsloth/Qwen3.8-27B-NVFP4` quantized weights, using the structure
and vocabulary of the official `Qwen/Qwen3.8-27B`, through this repository's conversion
pipeline `tools/convert/qwen3_8_27b/convert_nvfp4.py`. The adjacent
`qwen3_8_27b_nvfp4.ninfer.conversion.json` records the source repositories, revisions, and
conversion parameters (this artifact pins structure source `1d4bf0f2…` and quantization source
`60e813d4…`) for traceability and verification.

Both sources are publicly downloadable. Pin versions with `--revision` for reproducibility:

```bash
# In mainland China, use the mirror and disable Xet transfer (otherwise large downloads fail)
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# Official structure / vocabulary / chat template
huggingface-cli download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir ./models/Qwen3.8-27B

# NVFP4 quantized weights
huggingface-cli download unsloth/Qwen3.8-27B-NVFP4 \
  --revision 60e813d4dbbdc5d64cf3f5a8caf2897bedf03679 \
  --local-dir ./models/unsloth-qwen3.8-27b-nvfp4
```

Conversion (the tool is maintained on the dev branch; see the header of
`tools/convert/qwen3_8_27b/convert_nvfp4.py` and `--help`):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 \
  --model ./models/Qwen3.8-27B \
  --quantized-model ./models/unsloth-qwen3.8-27b-nvfp4 \
  --out out/qwen3_8_27b_nvfp4.ninfer
```

> Notes:
> - An existing NVFP4 artifact is incompatible with this fork's GDN `query_key_value_z` loading contract (it fails immediately with a tensor descriptor contract error). **Do not use it.**
> - That artifact is text-only; its Vision tower contract is incompatible (see problem 1). Do not use it. The current self-converted NVFP4 artifact includes Vision; the GGUF Q4_K_M artifact rejects `--vision`.

---

## Installation

### Dependencies

- NVIDIA CUDA Toolkit **12.8** (with `nvcc`, supporting `sm_70`)
- CMake **≥ 3.28**, Ninja, GCC ≥ 13
- NCCL (`libnccl.so` + `nccl.h`, required for TP)
- (Optional) FFmpeg / libcurl development packages—only needed for multimedia input; pure-text serving can skip them

CUTLASS is fetched automatically by CMake FetchContent; no manual installation is needed.

### Build

```bash
git clone https://github.com/cmj99cn/ninfer-v100-fast.git
cd ninfer-v100-fast
git checkout dev

# Text-only serving (recommended, fewest dependencies)
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_SKIP_FFMPEG=ON

cmake --build build -j"$(nproc)" --target ninfer-serve
```

Output: `build/apps/ninfer-serve`.

For multimedia (image/audio/video input), remove `-DNINFER_SKIP_FFMPEG=ON` and install the
FFmpeg and libcurl development packages (`libavcodec-dev libavformat-dev libavutil-dev
libswscale-dev libcurl4-openssl-dev`, etc.). Alternatively, use `apt-get download` +
`dpkg-deb -x` to unpack them into a local prefix and set `PKG_CONFIG_SYSROOT_DIR` /
`PKG_CONFIG_PATH`—this is how the project's development environment does it; see
`TEMP_WORKAROUNDS.md` for details.

If NCCL comes from a pip package, make sure the linker can find it at runtime:

```bash
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH
```

---

## Usage

### Start the Server

```bash
export LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH   # depends on your NCCL location

./build/apps/ninfer-serve \
  /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --model-id qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 --port 9010 \
  --api-key YOUR_API_KEY \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 \
  --max-context 262144 \
  --prefix-cache \
  --no-thinking --greedy
```

For TP4 (four GPUs, legacy 4×16GB regression), replace `--tp 2 --devices 0,1` with `--tp 4 --devices 0,1,2,3`; the current 2×32GB release path uses TP2.

#### Multi-instance groups (partially validated)

One process can serve several Engines, each on a disjoint device group, sharing one read-only
artifact mmap and one pinned host prefix-KV tier; device weights and device KV remain private. The
host tier stores only complete 64-token prefix-cache blocks and snapshots, not all active KV; there
is no disk KV tier or KV offload. The single OpenAI/Anthropic frontend routes through `EnginePool`
by prefix hit, free capacity, and aging, not simple round-robin. `--group-devices` separates groups
with semicolons and `--tp` is inferred from the group width when omitted. `--max-concurrency` and
`--kv-capacity` are per instance; `--global-pending-requests` bounds the shared queue (default 16).

| Cards | Groups × TP | `--group-devices` | Status |
|---:|---|---|---|
| 2 | 2×TP1 | `"0;1"` | **real integration test passed** |
| 4 | 4×TP1 | `"0;1;2;3"` | unverified |
| 4 | 2×TP2 | `"0,1;2,3"` | unverified |
| 8 | 8×TP1 | `"0;1;2;3;4;5;6;7"` | community/out-of-contract; unverified |
| 8 | 4×TP2 | `"0,1;2,3;4,5;6,7"` | community/out-of-contract; unverified |
| 8 | 2×TP4 | `"0,1,2,3;4,5,6,7"` | community/out-of-contract; unverified |

Prefer `--group-devices` for multi-group launches. `--devices` can also list all ranks in group
order, but the `ninfer` CLI accepts at most four IDs. `--spec dflash` is rejected with multiple
groups.

Two-card host (2xTP1; the real integration test passed):

```bash
./build/apps/ninfer-serve /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --groups 2 --group-devices "0;1" --tp 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --prefix-cache --host-cache-mib 1024
```

Eight-card host (8xTP1; unverified, eight full instances, highest memory pressure):

```bash
./build/apps/ninfer-serve /home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --groups 8 --group-devices "0;1;2;3;4;5;6;7" --tp 1 \
  --max-context 262144 --kv-capacity 262144 --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 \
  --prefix-cache --host-cache-mib 153600
```

Script profiles (community/out-of-contract): `bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1|4xTP2|2xTP4`
(add `--dry-run` to print the full command). See
[`docs/maintainer/multi-instance.md`](docs/maintainer/multi-instance.md) for the 8xTP1, 4xTP2,
2xTP4, 4xTP1, and 2xTP2 parameters, routing, and validation boundaries. These unverified profiles
have no measured throughput or memory result and are not acceptance evidence.

**Community request (8 cards):** on an 8×V100-SXM2-32GB host, please prioritize 8xTP1, 4xTP2, and
2xTP4 and report the launch command, binary/artifact hashes, the `/metrics?format=json`
`instances` array, per-instance committed decode tok/s, TTFT/TPOT, MTP acceptance, queue wait,
free KV pages, actual host-tier allocation, and 32GB memory headroom for 4xTP2/2xTP4. 4xTP1 and
8xTP1 need 32GB cards; a 16GB V100 cannot hold one full 27B copy. The 4-card profiles (4xTP1/2xTP2)
are also unverified and welcome. `--dry-run` or static checks are not 8-card validation; no fix timeline or
SLA is promised.

### Key Flags

| Flag | Description |
|---|---|
| `--tp 2\|4` + `--devices` | Tensor-parallel degree and device list (single process) |
| `--prefix-cache` | Content-addressed prefix cache (strongly recommended for multi-turn conversations) |
| `--spec mtp --draft-tokens 3` | MTP speculative decoding; significantly improves decode throughput |
| `--kv-dtype int8` | INT8 KV cache (recommended default) |
| `--max-context` | Maximum context (this artifact natively supports 262144) |
| `--no-thinking` / `--greedy` | Disable thinking mode / use greedy sampling (common for evaluation) |

Full flags: `./build/apps/ninfer-serve --help`. Reference launch scripts:
`tools/v100/ninfer-aligned-tp2.sh` (aligned A/B profile) and
`tools/v100/ninfer-quality-matched.sh` (thinking/sampling alignment).

### API Calls (OpenAI-compatible)

```bash
curl http://127.0.0.1:9010/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-nvfp4",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 128
  }'
```

Compatible with `/v1/chat/completions` (including `tools` function calling) and `/v1/models`. Health check: `GET /health`.

### Environment Variables (tuning/troubleshooting)

| Variable | Description |
|---|---|
| `NINFER_VOLTA_SPLITD=0` | Disable the splitd prefill kernel (enabled by default) |
| `NINFER_SM70_SPLITKV3_MIN_KV` | split-KV3 threshold (default 2048) |
| `NINFER_SM70_ATTN_V2=0` | Fall back to the phase-1 decode attention kernel (v2 is the default) |
| `NINFER_SM70_LONG_SPLIT_KEYS` | Number of keys per split in long windows (measured ineffective on this base; kept for A/B) |
| `NINFER_SM70_LONG_SPLITS_ONLY` | Bypass the tier floor to force the long-split policy (measured uniformly worse; kept for comparison) |
| `NINFER_SM70_DECODE_MB` | MinBlocks for T1-3 >6144 windows (default 5; 2/4/6 are all worse) |
| `NINFER_SM70_PREFILL_GEMM` | Cold-prefill GEMM routing: 0=CUTLASS identity, 1=cuBLAS (+20~23% cold-prefill tps; `tools/v100/ninfer-aligned-tp2.sh` defaults to 1) |
| `NINFER_SM70_GEMM_TRACE` | When set, print prefill cuBLAS call dimensions/device/thread diagnostics (troubleshooting) |

---

## Known Limitations

- **Vision**: the current self-converted NVFP4 artifact includes Vision; the GGUF Q4_K_M artifact rejects `--vision`; multi-instance Vision is unverified.
- Only 2×TP1 multi-instance has passed a real integration test; the 4-card (4×TP1/2×TP2) and 8-card (8×TP1/4×TP2/2×TP4) profiles are unverified, and the 8-card profiles are community/out-of-contract. 4×TP1 and 8×TP1 require 32GB cards; a 16GB V100 cannot hold one full 27B copy.
- KV has two tiers only: per-instance device KV and the process-shared pinned host prefix-KV tier. There is no disk KV tier, KV offload, or global active-KV pool.
- After the prefix-cache fix, no long soak test has been run; cache behavior under very high concurrency is still being observed.
- **BFCL full 200: ninfer latest 27.0% (54/200)**. **The issue remains unresolved; blind active fixing is paused and community help is requested.** Prompt/top_k/chunk/KV/parser have been ruled out; the top remaining suspects are the self-converted `.ninfer` artifact, NVFP4/attention numerics, and prefix-cache state (see the “Quality” section and `BFCL_QUALITY_ISSUE.md`).

---

## License

This repository's code is distributed under the applicable license terms and **inherits the Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for the full text. Model artifacts are licensed under their publishing repositories (Hugging Face).

---

*This README and all code and documentation in this repository are generated and maintained by AI (see “Project Statement”).*
