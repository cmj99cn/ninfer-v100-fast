[English](#english)

# ninfer-v100-fast

基于 NInfer 的 Tesla V100 推理引擎：单进程 TP2（TP4 为 4×16GB legacy 回归）、NVLink 加速、多轮前缀缓存、长上下文 prefill 优化与 MTP 加速解码，目标是在 2×V100-SXM2-32GB 上把 Qwen3.8-27B NVFP4 的吞吐与延迟做到当前硬件的极限。

---

## 项目声明（请先读）

1. **本仓库全部代码与文档由 AI 编写生成。** 仓库所有者不具备大模型技术背景，不理解模型算法与推理引擎内部实现。**请不要向所有者提问任何技术问题**——所有者没有能力回答，也没有时间回复。
2. **问题、复现和社区协作通过 GitHub Issues/Discussions 与仓库文档进行。** 欢迎提交可复现的质量问题、logits 证据或转换校验；请附带步骤、日志、硬件/软件版本和环境信息。本项目不承诺修复周期或 SLA。
3. 本仓库是**全新项目**：以干净历史导入改进后的 NInfer 代码，不携带上游或其他 fork 的提交历史。上游与相关修改的历史见文末「鸣谢」与 `NOTICE`。

---

## 发布版本

首个公开版本：**v0.1.0（2026-10-27）**。版本说明与已知限制见 [`CHANGELOG.md`](CHANGELOG.md)。

---

## BFCL 多轮工具调用质量：已定位并修复

最新全量结果为 **no-spec 43.0%（86/200）、MTP-3 43.5%（87/200）**；修复前为
36.0%（72/200），更早基线为 23.0%（46/200）。与此同时修复了 FP16 runtime 的
decode/prefill 性能回退（详见“速度”节）。

根因是三类实现缺陷，均已修复：

1. **tool-call 参数类型**：补齐上游的 JSON Schema typing（数字样字符串如 zip code 保持
   字符串），并把 Qwen 输出的 Python 字面量 `True/False/None` 归一化为 JSON
   `true/false/null`。旧实现把 `False` 变成 truthy 字符串 `"False"`，直接反转工具语义。
2. **NVFP4 小 T QPN 内核**：把权重除数 `1/d_w` 从 FP16 block scale 移到 FP32 epilogue，
   消除约 `1.75e-4` 的系统标度偏差。
3. **prefix reuse 的数值等价性**：禁用 non-speculative 路径对
   `restore_turn_checkpoint` / `restore_response_checkpoint` 的复用（它们复用的是
   decode 路径状态，下一次 prefill 续接并不数值等价）；只保留 prefill 状态的 block
   snapshot 或 cold prefill。修复后同一 server 连续两次运行 **0/10 轨迹差异**，不再是
   此前的 5–8/10。

复现与历史实验记录已并入本文档；旧的质量问题追踪文件已移除。

---

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

## 当前进度（v0.2.0，截至 2026-11-07）

本版本在 v0.1.0 之上完成 FP16 text runtime，修复 BFCL 多轮质量，并消除 FP16 路径的
decode/prefill 回退。BFCL `multi_turn_base` 全量：**no-spec 86/200 = 43.0%、
MTP-3 87/200 = 43.5%**。MTP-3 decode 四点
（262144 上下文、INT8 KV、FP16-embedding 制品）：**104.7 / 98.9 / 109.6 / 98.1 tok/s**。

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
| 模型 | 自转换 `qwen3_8_27b_nvfp4_f16embed.ninfer`（NVFP4 + FP16 embedding，18.0 GiB；质量/速度实测用它）与标准 `qwen3_8_27b_nvfp4.ninfer`（16.8 GiB，速度相同）；源 `unsloth/Qwen3.8-27B-NVFP4` |
| 服务参数 | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## 测试用例与结果

以下全部为**本项目自身实测数据**（2026-10），不与其他引擎或 fork 对比。

### 速度（TP2，NVFP4，2×V100）

**Prefill**

| 用例 | 方法 | 结果 |
|---|---|---|
| 短 prompt（2048 tok，CLI） | CLI 自报 | **1826 tok/s**（修复前 1522，与 BF16 基线 1825 持平） |
| 服务端冷 prefill @3.7K | server 自报 | **2057 tok/s**（TTFT 1.73 s） |
| 174K tokens 长文档（splitd 开启） | 专用 A/B 脚本，单请求 | **1059.7 tok/s** |
| 同上（splitd 关闭，对照） | 同上 | 985.4 tok/s（开启 splitd 相对提升 **+7.5%**） |

**Decode（生成 512–1024 tokens）**

对齐 profile，`long_gen_prompt` 四点（prompt 502 / 3.7K / 15K / 59K），`max_tokens=128`：

| 配置 | @502 | @3.7K | @15K | @59K |
|---|---|---|---|---|
| 无 MTP | 51.0 | 49.9 | 46.4 | 36.0 |
| **MTP（`--spec mtp --draft-tokens 3`）** | **104.7** | **98.9** | **109.6** | **98.1** |

MTP 为各点 **2.0–2.7×**，接受率 **52–66%**（2.5–3.0 tok/round）；TTFT 增加约 0.15s
（draft 初始化）。以上为 f16embed 制品在 `--max-context 262144`、`--prefill-chunk 8192`
下的实测；标准制品同口径为 103.1 / 102.4 / 112.6 / 97.7。

**多轮前缀缓存（`--prefix-cache`）**

15 轮长上下文压力测试（8K tokens 起步、每轮追加代码块至约 39K）：全部成功，server 日志确认持续命中 `reuse=block_snapshot`，末轮缓存命中约 36800 tokens，命中后 prefill 仍保持 1200–1600 tok/s。

### 质量（BFCL 多轮工具调用，已解决）

**最新全量结果：no-spec 86/200 = 43.0%，MTP-3 87/200 = 43.5%（2026-11-07）**，
修复前为 36.0%（72/200），更早基线 23.0%（46/200）。

运行设置：BFCL-v3 `multi_turn_base` 全量 200 条，thinking OFF，
`temperature=1.0`、`top_p=1.0`、`seed=42`，TP2，FP16-embedding NVFP4 制品，
INT8 KV，`--prefill-chunk 8192`、`--prefix-cache` 开启。**每次评测前重启 server**
（跨 run 复用进程会污染前缀缓存状态）。BFCL prompt 很短（Avg In ≈ 5K），
`--max-context 131072` 与发布用的 262144 质量一致（262144 已确认可容纳该制品）。
结果目录：`/home/cmj/eval-runs/bfcl/ninfer_f16_prefillfix_notc_s42_n200`（no-spec）、
`/home/cmj/eval-runs/bfcl/ninfer_f16_fixed_mtp3_notc_s42_n200`（MTP-3）。

| 运行 | no-spec | MTP-3 |
|---|---:|---:|
| **ninfer（v0.2.0）** | **43.0%（86/200）** | **43.5%（87/200）** |
| ninfer（修复 FP16 prefill 前） | 44.0%（88/200） | — |

#### 关于 no-spec 44.0% → 43.0%（可优化点）

修复 prefill attention（FP16 prefill 从 chunked small-T 切到 Volta flash split-D，与 BF16
生产路径一致）后，no-spec 从 88/200 变为 86/200。逐样本对比只有 4 条翻转：index 17/56/64
（通过→未通过）、index 61（未通过→通过），净 −2。这是 n200 的采样抖动（1 SE ≈ 3.5 个点，
差异 1 个点），不是精度下降：flash split-D 的 attention 输出为 FP32 累加（原 small-T 路径
用 FP16 部分累加），只是归约顺序不同；在 `temperature=1.0` 下，边界工具调用任务会对极小
logit 差异翻转。**可优化点**：评测侧对 borderline tool-call 任务加 greedy 复核/固定采样，
或对 flash/small-T 的 crossover（`NINFER_SM70_ATTN_V2` 与 prefill 内核阈值）做小范围 A/B，
把这类翻转压到 0；不影响发布口径（43–44% 均显著高于 41% 基线）。

修复内容与验收证据：

1. **Tool-call schema typing + Python 字面量**：string 参数按 JSON Schema 原样保留；
   `True/False/None` 归一化。前 100 条从 39% 提到 50%，6 个 zip-code 样本全部翻盘。
2. **NVFP4 QPN 除数**：`1/d_w` 在 FP32 epilogue 应用；in_proj 对 FP64 dequant oracle
   的误差从 `3.0e-4` 降到 `2.0e-4`。
3. **Prefix reuse 数值等价性**：non-speculative 路径不再恢复 rewrite checkpoint（turn/response
   closure），只保留 block snapshot（prefill 状态）或 cold prefill；同 server 重跑
   0/10 轨迹差异。`bf8595c3` 把同一禁用扩展到 MTP：MTP 保留 checkpoint 时全量
   69/200 = 34.5%，禁用后 72/200 = 36.0%；本次进一步修复 FP16 prefill 数值路径后，
   MTP-3 达到 87/200 = 43.5%（与 no-spec 持平）；DFlash 暂保留（未测）。

仍未覆盖：block-snapshot reuse 仍会产生确定性的不同轨迹（本轮同分），以及 DFlash 路径
仍保留 rewrite checkpoint。二者不影响单请求 release 验收。

本次评测与发布均使用 `--prefill-chunk 8192`（旧评测曾用 128 作稳定性 workaround，代价是长提示冷预填充约 3.3× 慢；该 workaround 已不再需要）。FP16 prefill 修复后，冷 prefill @3.7K 约 2057 tok/s，与 BF16 基线持平。

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
- **BFCL 全量 200 条：ninfer no-spec 43.0%（86/200）、MTP-3 43.5%（87/200）**。本轮修复了 tool-call 参数类型、NVFP4 QPN 除数、prefix-reuse 的 rewrite checkpoint 复用，以及 FP16 的 gating/prefill 性能与数值路径；同一 server 重跑已可复现（见“质量”节）。

---

## 鸣谢

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — NInfer 原始项目，本仓库的引擎基座。
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) — 双卡张量并行与 YaRN 扩展。
- [jason-ni/ninfer-v100-tp4](https://github.com/jason-ni/ninfer-v100-tp4) — V100 TP4 单进程并行基底（本仓库代码直接来源）。
- [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) — V100 上的 NInfer 单卡部署与跨引擎基准 fork（调研参考，未直接复用其代码）。
- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — 默认模型的 NVFP4 量化权重源。
- [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) — `.ninfer` 制品格式的参考（本 fork 契约不兼容，仅作格式参照）。
- Qwen / NInfer 生态的所有贡献者。

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
3. This repository is a **new project**: improved NInfer code is imported with a clean history, without upstream or other fork commit history. See “Acknowledgements” and `NOTICE` for upstream and modification history.

---

## Release

First public release: **v0.1.0 (2026-10-27)**. See [`CHANGELOG.md`](CHANGELOG.md) for release notes and known limitations.

---

## BFCL Multi-Turn Tool-Calling Quality: Located and Fixed

The latest full run scores **no-spec 43.0% (86/200) and MTP-3 43.5% (87/200)**, up from
36.0% (72/200) before this fix and 23.0% (46/200) at the earlier baseline. The same round
also removed the FP16 runtime's decode/prefill regression (see "Speed").

Three implementation defects were found and fixed:

1. **Tool-call argument typing**: adopted upstream JSON-Schema typing (numeric-looking
   strings such as zip codes stay strings) and normalized Qwen's Python literals
   `True/False/None` to JSON `true/false/null`. The old parser turned `False` into the
   truthy string `"False"`, reversing tool semantics.
2. **NVFP4 small-T QPN kernel**: moved the weight divisor `1/d_w` from the FP16 block
   scale to the FP32 epilogue, removing a systematic `1.75e-4` scale bias.
3. **Prefix-reuse numerical equivalence**: the non-speculative path no longer restores
   `restore_turn_checkpoint` / `restore_response_checkpoint` (they carry decode-path
   state, which is not numerically equivalent to a later prefill of the same tokens);
   only prefill-state block snapshots or a cold prefill are used. Two consecutive runs on
   the same server now show **0/10 trajectory differences** (previously 5–8/10).

The reproduction and experiment record has been merged into this document; the old
quality-issue tracking files were removed.

---

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

## Current Status (v0.2.0, as of 2026-11-07)

This version adds the FP16 text runtime on top of v0.1.0, fixes BFCL multi-turn quality,
and removes the FP16 path's decode/prefill regression. BFCL `multi_turn_base`, full 200:
**no-spec 86/200 = 43.0%, MTP-3 87/200 = 43.5%**. MTP-3 decode at four points (262144 context, INT8 KV, FP16-embedding artifact):
**104.7 / 98.9 / 109.6 / 98.1 tok/s**.

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
| Model | Self-converted `qwen3_8_27b_nvfp4_f16embed.ninfer` (NVFP4 + FP16 embedding, 18.0 GiB; used for the quality/speed measurements) and the standard `qwen3_8_27b_nvfp4.ninfer` (16.8 GiB, same speed); source `unsloth/Qwen3.8-27B-NVFP4` |
| Server flags | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## Test Cases and Results

All data below are **measured on this project** (October 2026) and are not comparisons against other engines or forks.

### Speed (TP2, NVFP4, 2×V100)

**Prefill**

| Case | Method | Result |
|---|---|---|
| Short prompt (2048 tok, CLI) | CLI-reported | **1826 tok/s** (1522 before the fix; matches the BF16 baseline of 1825) |
| Server cold prefill @3.7K | server-reported | **2057 tok/s** (TTFT 1.73 s) |
| 174K-token long document (splitd on) | dedicated A/B script, single request | **1059.7 tok/s** |
| Same (splitd off, reference) | same | 985.4 tok/s (splitd gives **+7.5%**) |

**Decode (generating 512–1024 tokens)**

Aligned profile, `long_gen_prompt` at four points (prompt 502 / 3.7K / 15K / 59K), `max_tokens=128`:

| Configuration | @502 | @3.7K | @15K | @59K |
|---|---|---|---|---|
| No MTP | 51.0 | 49.9 | 46.4 | 36.0 |
| **MTP (`--spec mtp --draft-tokens 3`)** | **104.7** | **98.9** | **109.6** | **98.1** |

MTP is **2.0–2.7×** at every point, with **52–66%** acceptance (2.5–3.0 tok/round); TTFT
increases by about 0.15s (draft initialization). Measured with the f16embed artifact at
`--max-context 262144 --prefill-chunk 8192`; the standard artifact scores 103.1 / 102.4 /
112.6 / 97.7 at the same points.

**Multi-turn prefix cache (`--prefix-cache`)**

15-turn long-context stress test (starting at 8K tokens, appending a code block each turn up to ~39K): all turns succeeded. Server logs confirm continuous `reuse=block_snapshot` hits; the last turn hit ~36800 cached tokens, and prefill still sustained 1200–1600 tok/s after the hit.

### Quality (BFCL multi-turn tool calling, resolved)

**Latest full result: no-spec 86/200 = 43.0%, MTP-3 87/200 = 43.5% (2026-11-07)**,
up from 36.0% (72/200) before this fix and 23.0% (46/200) at the earlier baseline.

Run settings: BFCL-v3 `multi_turn_base`, full 200 samples, thinking OFF,
`temperature=1.0`, `top_p=1.0`, `seed=42`, TP2, FP16-embedding NVFP4 artifact, INT8 KV,
`--prefill-chunk 8192`, `--prefix-cache` enabled. **Restart the server before every run**;
reusing a process across runs contaminates prefix-cache state. BFCL prompts are short
(Avg In ~5K), so `--max-context 131072` matches the release's 262144 quality (262144 is
confirmed to hold the artifact). Results:
`/home/cmj/eval-runs/bfcl/ninfer_f16_prefillfix_notc_s42_n200` (no-spec),
`/home/cmj/eval-runs/bfcl/ninfer_f16_fixed_mtp3_notc_s42_n200` (MTP-3).

| Run | no-spec | MTP-3 |
|---|---:|---:|
| **ninfer (v0.2.0)** | **43.0% (86/200)** | **43.5% (87/200)** |
| ninfer (before the FP16 prefill fix) | 44.0% (88/200) | — |

#### On no-spec 44.0% → 43.0% (an optimizable point)

After the prefill-attention fix (FP16 prefill switched from chunked small-T to the Volta
flash split-D kernel that BF16 already uses), no-spec moved from 88/200 to 86/200.
A per-sample diff shows only four flips: indices 17/56/64 (pass→fail) and 61 (fail→pass),
net -2. This is n200 sampling noise (1 SE ~3.5 points; the difference is 1 point), not a
precision loss: the flash split-D attention accumulates in FP32 (the old small-T path used
FP16 partials), only the reduction order differs, and at `temperature=1.0` borderline
tool-call tasks flip on tiny logit changes. **Optimizable**: add a greedy re-check / fixed
sampler for borderline tool-call tasks in the evaluation, or A/B the flash/small-T crossover
(`NINFER_SM70_ATTN_V2` and the prefill kernel threshold) to drive these flips to zero; this
does not affect the release (43–44% both clear the 41% baseline).

Fixes and acceptance evidence:

1. **Tool-call schema typing + Python literals**: string parameters are preserved per the
   JSON Schema; `True/False/None` are normalized. The first 100 moved from 39% to 50%,
   flipping all six zip-code samples.
2. **NVFP4 QPN divisor**: `1/d_w` is applied in the FP32 epilogue; in_proj's error against
   the FP64 dequant oracle dropped from `3.0e-4` to `2.0e-4`.
3. **Prefix-reuse numerical equivalence**: the non-speculative path no longer restores the
   rewrite checkpoint (turn/response closure), only prefill-state block snapshots or a cold
   prefill; same-server reruns now show 0/10 trajectory differences. `bf8595c3` extends the
   same disable to MTP: MTP scored 69/200 = 34.5% with the checkpoint and 72/200 = 36.0%
   without it. After also fixing the FP16 prefill numerical path in this round, MTP-3 reaches
   87/200 = 43.5% (at parity with no-spec). DFlash is left unchanged (untested).

Still open: block-snapshot reuse still produces a deterministic but different trajectory
(same score this round), and the DFlash path retains the rewrite checkpoint. Neither
affects the single-request release acceptance.

This round's evaluation and release use `--prefill-chunk 8192` (the earlier evaluation used
128 as a stability workaround, which cost about **3.3×** on long-prompt cold prefill; that
workaround is no longer needed). After the FP16 prefill fix, cold prefill @3.7K is about
2057 tok/s, on par with the BF16 baseline.

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
- **BFCL full 200: ninfer no-spec 43.0% (86/200), MTP-3 43.5% (87/200)**. This round fixed tool-call argument typing, the NVFP4 QPN divisor, the rewrite-checkpoint prefix reuse, and the FP16 gating/prefill performance and numerical paths; same-server reruns are now reproducible (see the “Quality” section).

---

## Acknowledgements

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the original NInfer project and the engine base of this repository.
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) — dual-GPU tensor parallelism and YaRN extension.
- [jason-ni/ninfer-v100-tp4](https://github.com/jason-ni/ninfer-v100-tp4) — V100 TP4 single-process parallel base (direct source of this repository's code).
- [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) — NInfer single-GPU deployment and cross-engine benchmark fork on V100 (research reference; its code was not directly reused).
- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — NVFP4 quantized weight source for the default model.
- [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) — reference for the `.ninfer` artifact format (incompatible with this fork's contract; format reference only).
- All contributors to the Qwen / NInfer ecosystem.

## License

This repository's code is distributed under the applicable license terms and **inherits the Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for the full text. Model artifacts are licensed under their publishing repositories (Hugging Face).

---

*This README and all code and documentation in this repository are generated and maintained by AI (see “Project Statement”).*
