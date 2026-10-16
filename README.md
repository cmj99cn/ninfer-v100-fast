# ninfer-v100-fast

基于 NInfer 的 Tesla V100 推理引擎：单进程 TP2/TP4、NVLink 加速、多轮前缀缓存、长上下文 prefill 优化与 MTP 加速解码，目标是在 2×V100 上把 Qwen3.8-27B NVFP4 的吞吐与延迟做到当前硬件的极限。

---

## 项目声明（请先读）

1. **本仓库全部代码与文档由 AI 编写生成。** 仓库所有者不具备大模型技术背景，不理解模型算法与推理引擎内部实现。**请不要向所有者提问任何技术问题**——所有者没有能力回答，也没有时间回复。
2. **提交的 Issue 会由 AI 定期复现并修复。** 提交问题时请附带可复现的步骤、日志与环境信息，否则 AI 无法复现。修复周期不固定，不提供任何 SLA。
3. 本仓库是**全新项目**：以一次干净提交导入改进后的 NInfer 代码，不携带上游或其他 fork 的提交历史。上游与相关修改的历史见文末「鸣谢」与 `NOTICE`。

---

## 项目目标

在 2×Tesla V100-SXM2-32GB（NVLink）上，用**同一个引擎**同时满足：

| 目标 | 说明 |
|---|---|
| **TP2 / TP4 / NVLink** | 单进程张量并行，不依赖多进程代理；NVLink 上的轻量 all-reduce（mailbox）自动探测启用 |
| **缓冲命中做到极致** | `--prefix-cache` 内容寻址前缀缓存，多轮对话复用历史 KV，不重复 prefill |
| **预填充速度做到极致** | 长文档 splitd prefill 内核（174K 上下文实测 +7.5%） |
| **解码速度做到极致** | MTP 投机解码（`--spec mtp`）+ SM70 INT8 key-split 注意力内核 |

---

## 开发思路

1. **选基底**：对比多个 NInfer / vLLM / TRT fork 后，选择带**单进程 TP2/TP4** 的 `ninfer-v100`（jason-ni/ninfer-v100-tp4）作为基底——多进程双卡方案架构脆弱、无法扩展到 TP4，被直接放弃。
2. **移植长上下文 prefill**：将 1CatAI/1Cat-vLLM 的 **splitd prefill 内核**（D256 Split-D）移植到本基底（geometry 映射、去掉 Hadamard 旋转、增加 `NINFER_VOLTA_SPLITD=0` 开关、`NINFER_SM70_SPLITKV3_MIN_KV` 阈值），实测 174K prefill +7.5%。
3. **修复前缀缓存**：生产化验证时发现 prefix cache 在多轮场景触发引擎崩溃（见下文问题 2），修复后 `--prefix-cache` 可长期稳定运行。
4. **速度验证与调参**：系统性分离 prefill / decode 测量，确认 MTP 是解码的主要杠杆；确认 `NINFER_SM70_LONG_SPLIT_KEYS` 类 attention 调参在本基底无效（瓶颈不在该处），避免无意义调优。
5. **质量评测**：跑 BFCL-v3 多轮函数调用评测，得到当前基线分数（见测试结果），后续迭代以该基线对照。

### 遇到的问题与解决思路

| # | 问题 | 解决 |
|---|---|---|
| 1 | 官方 NVFP4 制品的 Vision tower 与本 fork 契约不兼容（量化格式、无独立 qkv、RoPE attention），加载即崩 | 加载时跳过 vision 绑定（text-only 制品）；多模态暂不可用，见 `TEMP_WORKAROUNDS.md` |
| 2 | 开启 prefix cache 后多轮对话触发 `Paged KV trim extent would discard cache-owned pages`，之后所有请求 503，引擎不可用 | 两处修复：① `trim_pages` 遇到 cache-owned 页时收缩 trim 范围而不是抛异常；② `promote_page_at` 对已 cache-owned 的页幂等返回 |
| 3 | 怀疑 decode 慢是 attention split 策略问题，尝试 `LONG_SPLIT_KEYS=1920` | A/B 实测无效，确认瓶颈不在此，停止该方向调优 |
| 4 | decode 速度测量一直包含 prefill 时间，数字失真 | 改用 warmup + 多次迭代 + server 自报 timings 的方法分离测量 |
| 5 | 无 root 环境编译需要 ffmpeg/libcurl 头文件 | `apt-get download` + `dpkg-deb -x` 本地解包，`PKG_CONFIG_SYSROOT_DIR` 指向本地前缀；或直接 `-DNINFER_SKIP_FFMPEG=OFF` 见安装章节 |

全部临时改动与复核事项见 [`TEMP_WORKAROUNDS.md`](TEMP_WORKAROUNDS.md)。

---

## 当前进度（开发分支，截至 2026-10-16）

### 对齐测试参数（A/B 基准，两引擎同参）

| 引擎 | 参数 |
|---|---|
| ninfer（本引擎） | `--tp 2 --devices 0,1 --max-context 262144 --max-concurrency 2 --prefill-chunk 8192 --prefix-cache --kv-dtype int8 --no-thinking --greedy`，`NINFER_VOLTA_SPLITD=1` |
| 1cat-vLLM 1.5.1（对照） | `--max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8_e5m2 --enable-prefix-caching`，model `qwen38-27b-nvfp4` |

### 与 1cat 的当前差距（2026-10-16 全套 A/B，同参同模型；冷 prefill 为 cuBLAS 路由落地后）

| 项 | ninfer | 1cat | 结论 |
|---|---|---|---|
| 冷 prefill @6683 tok | **3.076 s / 2172 tps** | 3.344 s / 1998 tps | **ninfer +8.7%** |
| 冷 prefill @13344 tok | **6.299 s / 2118 tps** | 7.855 s / 1699 tps | **ninfer +24.7%** |
| 冷 prefill @53142 tok | **29.867 s / 1779 tps** | 35.686 s / 1489 tps | **ninfer +19.5%** |
| decode @502 / 3.7K（无 MTP） | **50.9 / 50.0** | 49.6 / 49.0 | ninfer 赢 |
| decode @15K / @59K（无 MTP） | 46.2 / 35.9 | **46.7 / 39.1** | 平 / −8% |
| **decode @502 / 3.7K / 15K / 59K（MTP）** | **106.8 / 99.2 / 99.7 / 84.4** | 49.6 / 49.0 / 46.7 / 39.1 | **ninfer +80~116%** |
| prefix warm（29.5K 命中后） | **0.34 s** | 1.08 s | ninfer 赢 |
| conc2 总 wall（两流完成） | **9.96 / 10.00 s** | ~10.1 s | 持平 |
| conc2 首流 TTFT | **3.6 s** | 6.7 s | ninfer 赢 |
| conc2 共享前缀总 wall | **6.6 s**（借运行中 lane，`cache=7360`） | — | ninfer 赢 |

结论：
- **冷 prefill 缺口（−23~−25%）已翻盘**：GEMM 占 prefill 72%，cuBLAS 比 CUTLASS 快 18–27%；`NINFER_SM70_PREFILL_GEMM=1` 落地后三点全线反超（+8.7~+24.7%）。
- **@59K decode 的 −8% 由 MTP 反超**：84.4 vs 39.1（+116%）；MTP 各点 2.0–2.3×，接受率 47–58%（2.4–2.7 tok/round）。
- **conc2 的旧 −22% 是指标假象**：原 bench 两流 seed 不同，prompt 在 token 3（块 0 内）即分叉，内容寻址前缀缓存对**两引擎都**不可复用；1cat 把两流 TTFT 都拖到 ~7s 后纯解码，使每流 decode 均值偏高，而 ninfer 首流 3.6s 即出首 token、其均值被“另一流 8192-token 预填块共享混批单元”稀释。**两边总 wall 均为 ~10s，ninfer 首流 TTFT 更好**。共享前缀场景下 ninfer 借运行中 lane 的 KV（`reuse=block_snapshot cache=7360`），第二路 prefill 3.5s→~0.2s，总 wall 6.6s。

### 与 1cat 的当前差距（历史，cuBLAS 路由之前的 2026-10-16 全套 A/B，保留作对照）

| 项 | ninfer | 1cat | 结论 |
|---|---|---|---|
| 冷 prefill TTFT @14794 | 8.62 s | 6.68 s | −23%（已解决，见上表） |
| 冷 prefill TTFT @59020 | 41.70 s | 31.34 s | −25%（已解决，见上表） |
| conc2 合计 | 62.3 | 83.7 | −26%（后确认该指标为假象，见上表说明） |

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

### 质量

**BFCL-v3 多轮函数调用**（evalscope，`multi_turn_base` / `multi_turn_miss_func` / `multi_turn_miss_param` / `multi_turn_long_context` 各 20 条，greedy，thinking 关闭，`--prefix-cache` 开启）：

| Subset | 分数 |
|---|---|
| multi_turn_base | 20% |
| multi_turn_miss_func | 10% |
| multi_turn_miss_param | 35% |
| multi_turn_long_context | 15% |
| **OVERALL（80 条）** | **20%** |

该分数为当前基线，后续迭代在此基础上对照改进。函数调用质量与采样参数、工具模板、thinking 开关高度相关，迭代空间大。

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
> - 官方 `neroued/Qwen3.8-27B-nvfp4-NInfer` 制品与本 fork 的 GDN `query_key_value_z` 加载契约不兼容（加载即报 tensor descriptor 契约错误），**请不要使用**。
> - 该制品为 **text-only** 路线，Vision tower 契约不兼容（见问题 1），请不要使用 `--vision` 参数。

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
  ./models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer \
  --model-id qwen3.8-27b-nvfp4 \
  --host 0.0.0.0 --port 9010 \
  --api-key YOUR_API_KEY \
  --tp 2 --devices 0,1 \
  --kv-dtype int8 \
  --max-context 262144 \
  --prefix-cache \
  --no-thinking --greedy
```

TP4（四卡）把 `--tp 2 --devices 0,1` 换成 `--tp 4 --devices 0,1,2,3` 即可，其余参数不变。

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

- **多模态不可用**（问题 1）：当前制品 vision tower 不兼容，仅文本。
- prefix cache 修复后未做长时间浸泡测试；超高压并发下的缓存行为仍在观察。
- BFCL 基线 20%，质量迭代是下一步重点。

---

## 鸣谢

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — NInfer 原始项目，本仓库的引擎基座。
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) — 双卡张量并行与 YaRN 扩展。
- [jason-ni/ninfer-v100-tp4](https://github.com/jason-ni/ninfer-v100-tp4) — V100 TP4 单进程并行基底（本仓库代码直接来源）。
- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — splitd（D256 Split-D）长上下文 prefill 内核的来源，已 vendored 到 `third_party/llama_cpp_sm70_d256/`。
- [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) — V100 上的 NInfer 单卡部署与跨引擎基准 fork（调研参考，未直接复用其代码）。
- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — 默认模型的 NVFP4 量化权重源。
- [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) — `.ninfer` 制品格式的参考（本 fork 契约不兼容，仅作格式参照）。
- Qwen / NInfer 生态的所有贡献者。

## 许可协议

本仓库代码在上游许可条款下分发，**继承 Apache License 2.0**，完整声明见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。模型制品的许可以其发布仓库（Hugging Face）为准。

---

*本 README 及仓库内全部代码、文档由 AI 生成与维护（见「项目声明」）。*
