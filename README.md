[English](#english)

# ninfer-v100-fast

基于 NInfer 的 Tesla V100 推理引擎：单进程 TP2（TP4 为 4×16GB legacy 回归）、NVLink 加速、多轮前缀缓存、长上下文 prefill 优化与 MTP 加速解码，目标是在 2×V100-SXM2-32GB 上把 Qwen3.8-27B NVFP4 的吞吐与延迟做到当前硬件的极限。

---

## 分支定位（请先读）

本分支是 **`feat/qwen3-8-flash-next`** Flash-Next 开发分支，基于 main 的 **v0.2.5**
（27B NVFP4 发布版）。下文保留 main 的 v0.2.5 27B 发布内容，均标注为「main 分支发布状态」；
**本分支主体是 Flash-Next——仍在开发中的 target，不是发布产品**。

- **模型与术语**：Flash-Next 即 **Qwen3.8-Flash-Next**（Qwen4Exp 系列），亦可写作
  `qwen3.8-flash-next`；48 层（36 GDN + 12 QSA）、512 routed experts、NVFP4 权重 +
  FP8 draft bank、TP2、INT8 KV、262144 上下文。
- **本轮 MTP 结论（诚实记录）**：Flash-Next MTP 现已快过 base——base no-spec **70.73–70.82 tok/s**；
  最终配置（Q4 proposal/head 默认 + draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + 专家画像
  0 miss，TP2 / `sm_70` / CUDA 12.8 / greedy / eager 单请求 / window=3 / 128 token / 2 reps）多次实测
  **111–113 tok/s**、acceptance **0.858–0.862**，即 **1.60× base**。历史 110.3–110.5 tok/s /
  0.860–0.861 是同一配置的早期实测，属正常波动，不写死单点；早期 base 记录 70.54 tok/s 同样保留
  作历史。**120 tok/s 目标仍未达成**（2× base = 141，本轮约 1.60×，非 2×）。acceptance 的根因是
  min_p=0.8 下低置信度 draft 本身就错：只发 1 个 draft 的轮次命中 **41/213 = 19.2%**，发 3 个
  draft 的轮次命中 **1014/1051 = 96.5%**，整体 step0 1086/1366 = 79.5%——是经济权衡不是 bug，
  不改 artifact 无法突破 0.79，2× 需要 artifact 层更强/多层 MTP head。「MTP 落后 base / 负收益」
  的旧结论已失效——起点是 30.33 tok/s（fold commit 接线前），本轮 30 → 约 113 提升约 **3.6×**。
  注意变量名必须完整写成 `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`；漏掉 `NEXT` 会静默退回默认 0，
  使跨时段 A/B 口径失效。120 / 99% 属随机采样口径，greedy 下不可达（见「口径澄清」）。详见
  「当前进度 → Flash-Next 分支实测」与「速度」节。

---

## 项目声明（请先读）

1. **本仓库全部代码与文档由 AI 编写生成。** 仓库所有者不具备大模型技术背景，不理解模型算法与推理引擎内部实现。**请不要向所有者提问任何技术问题**——所有者没有能力回答，也没有时间回复。
2. **问题、复现和社区协作通过 GitHub Issues/Discussions 与仓库文档进行。** 欢迎提交可复现的质量问题、logits 证据或转换校验；请附带步骤、日志、硬件/软件版本和环境信息。本项目不承诺修复周期或 SLA。
3. 本仓库是**全新项目**：以干净历史导入改进后的 NInfer 代码，不携带上游或其他 fork 的提交历史。上游与相关修改的历史见文末「鸣谢」与 `NOTICE`。

---

## 发布版本

当前版本：**v0.2.5（2026-11-13）**（首个公开版本 v0.1.0，2026-10-27）。版本说明与已知限制见 [`CHANGELOG.md`](CHANGELOG.md)。v0.2.5 是 **main 分支的发布状态**；本分支 `feat/qwen3-8-flash-next` 在此之上新增 Flash-Next 开发 target（见顶部「分支定位」），不产生新发布版本。

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
| **Flash-Next target（开发中，非发布产品）** | Qwen3.8-Flash-Next / Qwen4Exp：48 层（36 GDN + 12 QSA）、512 routed experts、NVFP4 权重 + FP8 draft bank、TP2、INT8 KV、262144 上下文；artifact `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`（v3 容器，108 GB，分 4 个 part） |

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
| 1 | 已有 NVFP4 制品的 Vision tower 与本 fork 契约不兼容（量化格式、无独立 qkv、RoPE attention），加载即崩 | 不使用该制品；当前自转换 NVFP4 制品包含 Vision，GGUF Q4_K_M 制品拒绝 Vision。多实例 Vision（2×TP1）与 TP2 Vision 均已通过真实集成测试，见 `TEMP_WORKAROUNDS.md` |
| 2 | 开启 prefix cache 后多轮对话触发 `Paged KV trim extent would discard cache-owned pages`，之后所有请求 503，引擎不可用 | 两处修复：① `trim_pages` 遇到 cache-owned 页时收缩 trim 范围而不是抛异常；② `promote_page_at` 对已 cache-owned 的页幂等返回 |
| 3 | 怀疑 decode 慢是 attention split 策略问题，尝试 `LONG_SPLIT_KEYS=1920` | A/B 实测无效，确认瓶颈不在此，停止该方向调优 |
| 4 | decode 速度测量一直包含 prefill 时间，数字失真 | 改用 warmup + 多次迭代 + server 自报 timings 的方法分离测量 |
| 5 | 无 root 环境编译需要 ffmpeg/libcurl 头文件 | `apt-get download` + `dpkg-deb -x` 本地解包，`PKG_CONFIG_SYSROOT_DIR` 指向本地前缀；或直接 `-DNINFER_SKIP_FFMPEG=OFF` 见安装章节 |
| 6 | FP16 text runtime 下 Vision 请求返回 500 `scatter: src/dst must share BF16 or FP16 dtype`（Vision tower 输出 BF16，文本残差为 FP16） | 在 Vision 嵌入散射进文本残差前增加 BF16→激活 dtype 的转换缓冲，覆盖 TP1、TP2 与多实例路径；2×TP1 双 lane 与 TP2 单实例均实测返回正确结果 |
| 7 | `--devices` 不从 0 开始时（如 `--tp 2 --devices 4,5`）TP mailbox 崩溃：开启 P2P 时把 rank 序号当作 CUDA 设备序号 | `cudaDeviceEnablePeerAccess` 改用各 rank 的真实设备序号；`--devices 4,5` 下自检通过并选中 nvlink mailbox |

全部临时改动与复核事项见 [`TEMP_WORKAROUNDS.md`](TEMP_WORKAROUNDS.md)。

---

## 当前进度（main 分支 v0.2.5 发布状态，截至 2026-11-13）

> 本节记录的是 **main 分支的 27B v0.2.5 发布状态**，不是本分支 Flash-Next 的结论。
> 本分支 Flash-Next 的实测见本节末尾「Flash-Next 分支实测」及「速度」节。

本版本在 v0.1.0 之上完成 FP16 text runtime，修复 BFCL 多轮质量，消除 FP16 路径的
decode/prefill 回退，修复 FP16 text runtime 下 Vision 请求的 scatter dtype 失败
（TP1、TP2 与多实例均已实测），并修复 `--devices` 不从 0 开始时 TP mailbox 的
P2P 使能崩溃。v0.2.3 修复多实例下温暖 home 被 reuse-blind gross 页账单挤出，补齐
host tier 容量公式与分配失败指标；v0.2.4 消除多实例接纳被解码实例执行锁卡死——候选改为
非阻塞读取，忙实例本轮跳过，请求留在全局 pending 队列；v0.2.5 在 OpenAI/Anthropic usage
上报前缀缓存复用，并修复 FP16 文本运行时下视觉 MTP 桥接的 rmsnorm dtype 崩溃。第三方已在
8×V100-SXM2-32GB 上完成 4 卡 `2×TP2` 池与 8 卡机双池并存的端到端验证。BFCL `multi_turn_base` 全量：**no-spec 86/200 = 43.0%、
MTP-3 87/200 = 43.5%**。MTP-3 decode 四点
（262144 上下文、INT8 KV、FP16-embedding 制品）：**104.7 / 98.9 / 109.6 / 98.1 tok/s**。

**速度与质量优化已基本到达本硬件平台的上限（仅为 main 分支 27B v0.2.5 发布版的结论，不适用于本分支开发中的 Flash-Next target）。**

- decode：MTP-3 在四个上下文点为 104.7/98.9/109.6/98.1 tok/s，已达到或超过参考表
  （106.8/99.2/99.7/84.4）；MTP 接受率 52–66%，继续加 draft 或改采样策略的收益已进入平台期。
- prefill：1826 tok/s，与 BF16 基线 1825 持平；splitd、prefill GEMM、attention split 等既有
  旋钮均已逐项 A/B 到局部最优。
- 质量：no-spec 43.0%、MTP-3 43.5%（修复前基线 36.0%）；44%→43% 的差异仅为 n200 的
  4 个样本翻转（净 −2），属采样抖动而非精度退化。

剩余项（44%→43% 的边界采样、未测的 DFlash 路径）是边际优化，不是大杠杆；进一步提升需要
不同硬件或改变产品模型。详见“质量”节的“可优化点”。

### Flash-Next 分支实测（开发中，非发布产品）

本节是 `feat/qwen3-8-flash-next` 分支的新增内容，与上文 main 的 27B 发布数据分开记录。
Flash-Next 即 **Qwen3.8-Flash-Next**（Qwen4Exp 系列，亦写作 `qwen3.8-flash-next`）：
48 层（36 GDN + 12 QSA）、512 routed experts、NVFP4 权重 + FP8 draft bank、TP2、INT8 KV、
262144 上下文。artifact 为
`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`（v3 容器，108 GB，分 4 个 part）。

**实测条件**：2×V100-SXM2-32GB、TP2、`sm_70`、CUDA 12.8、greedy、eager 单请求，128-token、
2 reps、cache 15000 MiB + 专家画像（expert profile）、0 miss、window=3。最终配置为 Q4 proposal/head
默认 + draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + 专家画像 0 miss（变量名必须完整；漏掉 `NEXT`
会静默退回默认 0，见下文环境变量事故）。

| 配置 | 吞吐（tok/s） | acceptance | 状态 |
|---|---:|---:|---|
| base no-spec | **70.73–70.82** | — | 多次实测参照基线（早期单点 70.54） |
| 起点（fold commit 接线前） | 30.33 | — | 本轮起点；MTP 尚未接线 |
| MTP 最终配置（Q4 + draft=3 + min_p=0.8 + profile） | **111–113** | 0.858–0.862 | 多次实测；**1.60× base** |
| 同配置早期实测（正常波动） | 110.3–110.5 | 0.860–0.861 | 保留历史，不写死单点 |
| MTP draft=1 + profile（旧最佳，已被最终配置取代） | 51.78 | 0.881 | 历史记录：当时落后 base 约 27%，已失效 |
| MTP draft=3 + profile（min_p=0/变量名漏 `NEXT`） | 42.9 | 0.772 | 历史记录：加 min_p=0.8 与 Q4 后提升；漏字写法会使生效值静默变为 0 |
| MTP batch verify（`decode_span` 实验，已回退） | 27.0 | 0.56 | 数值不等价，已回退 |

**结论（诚实记录）**：最终配置多次实测 **111–113 tok/s**、acceptance **0.858–0.862**，相对
base **70.73–70.82** 约 **1.60×**；早期同配置 110.3–110.5 tok/s / 0.860–0.861 属正常波动，不是
另一个配置，「MTP 落后 base 约 27%」的旧结论已失效——起点 30.33 tok/s（fold commit 接线前）到
本轮约 113，提升约 **3.6×**。**120 tok/s 目标仍未达成**（2× base = 141，约 1.60×，非 2×），不得
解读为已达成 2×。acceptance 确凿根因（`debug-step0-acceptance.md`，L4）：`accepted_per_position`
是条件计数，表面「0.79→0.93→0.97 递增」是 min_p 早停的选择偏差，真实 step1 条件率 0.876；
min_p=0.8 下低置信度 draft 本身是错的——只发 1 个 draft（置信 <0.8）的轮次命中 **41/213 =
19.2%**，发 3 个 draft 的轮次命中 **1014/1051 = 96.5%**，整体 step0 1086/1366 = 79.5%；这是
经济权衡不是 bug。Q4 量化不是原因：96 步 Q4 vs FP16 argmax 一致率 **89/96 = 92.7%**，端到端
accepted_tokens Q4 3071 vs FP16 3072。已证伪的其他原因：draft QSA 漂移（正确口径 Δacc 0.0012）、
T=6 verify 数值回退（bit-exact）、head 量化（NVFP4/Q4 无增益、FP16 净负吞吐）。结论：不改 artifact
无法突破 0.79，2× 需要 artifact 层更强/多层 MTP head（artifact 契约变更）。无 profile 时 MTP 仅
**13.6–14 tok/s**（cache hit 0.85、约 1.5 万次 miss）；专家画像（从 Strata 同模型
`expert-profile.bin` 转换到 `tools/bench/expert-profile-flash-next.txt`）把 hit 提到 1.0、0 miss，
是早期关键一步。

**本轮实现（commit `b0b79c4b`）**：GDN replay fold commit 转为生产路径（Tasks 1–10，bit-exact
验证通过，取代此前「11.36 tok/s 已回退」的记录）；verify 多列权重复用 kernel
（hc_mix / linear / W8 / route / conv / gdn_proj / GDN stage+extract）；draft TP2 角色拆分；
Q4 proposal/head 默认（artifact 自带同源权重）；draft 置信度 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`
提前终止；sink 流同步。此前 `91b45ae5` 的 draft bank 量化常驻（5.03→2.52 GB）、MTP 状态事务
D2D 快照、T≤4 cached routed/shared MoE + FP16 small-T GEMM、专家画像链路
（`--expert-cache-profile`）仍是基础。

**验证**：构建通过，`git diff --check` 干净；mtp_verify / fold_commit / mtp_state /
gdn_replay_storage 通过，draft / chain / confidence / greedy_equivalence 因缺隐藏 fixture SKIP
（内核 oracle 仍运行）；历史 38/38 测试通过（`ctest -R "flash_next|mtp|moe_nvfp4"`），NLL greedy
11751、mean_nll 3.732。用户裁决接受 Q4 为默认：draft 是提案、正确性由 verify 保证；chain oracle
由 `{220,27967,813}` 改为 `{220,27967,53}`（Q4 语义），greedy 等价（MTP on/off）逐 token 一致；
FP16 对照开关 `NINFER_DIAG_FP16_DRAFT_HEAD=1` 保留，恢复 `{220,27967,813}`。

**未达成 / 负结果（标注现状；已证伪项保留但不作为下一步杠杆）**：
- **acceptance 已证伪的原因（不重复尝试）**：draft head QSA 跨轮漂移（正确口径重做 ON/OFF
  Δacc 0.0012）、T=6 verify 数值回退（bit-exact）、head 量化（NVFP4/Q4 无增益、FP16 净负吞吐）。
  draft expert 7-kernel 合并（`debug-b1-rootcause.md`）与跨轮 gating（`debug-b3-signal.md`）也已实测
  排除，不再是增量杠杆。
- **B5 悬崖已修复（commit `5286f982`）**：`kMoeMaxSmallT` 4→8，window≥5 的 MoE 10× 回退已消除，
  window=5 从 20.9 → 106.3 tok/s；但 acceptance 降到 0.79，仍低于 window=3，故 window=3 不变。
- **环境变量拼写事故（`debug-acceptance-drift.md`）**：一次实验误用 `NINFER_FLASH_DRAFT_MIN_P`
  （漏 `NEXT`）使生效 min_p=0，多个跨时段 A/B 口径失效；gate JSON 不记录生效 min_p，所有 A/B
  必须显式记录变量名 `NINFER_FLASH_NEXT_DRAFT_MIN_P` 与取值 0.8。
- **根因审计（`analysis-rootcause-audit.md` UPDATE）**：27 个已排除项中 L4=6、仍 21 项 <L4；
  「已排除」≠「找到根因」，已对最高风险项回填。
- **CUDA Graph 对 MTP 无效**（51.85 vs 51.78；整轮图 A/B −4.4%）：根因已定位并保留结论——sm_70
  pre-Ampere 上 `cudaGraphLaunch` 的 O(N) 逐节点 CPU 提交成本（3,054 节点 → ~0.6 ms/round）超过
  它省下的、已被设备执行掩盖的 host 间隙；Strata 同为 V100 却能用 graph，靠的是 kernel 内 t 权
  复用（eager 侧属性），该收益我们的多列 kernel 已在 eager 侧领取。负收益原因已查明，非 kernel 更慢。
- **GDN replay fold 链路**：此前 11.36 tok/s < 14 的回退记录已过时——fold commit 现已是生产路径
  （Tasks 1–10），bit-exact 验证通过。
- **batch verify `decode_span`**：数值不等价（acceptance 0.56），已回退；greedy 下严格 argmax
  判定要求 bit-exact，这项仍成立。
- 层分 / 并行架构未改（主模型保持 TP2）。

**下一步**：120 tok/s（2× base = 141）仍未达成；不改 artifact 无法突破 step0 0.79，2× 需要
artifact 层更强/多层 MTP head（artifact 契约变更），不是靠已排除的工程杠杆。greedy 语义下概率
rejection sampling 不可替代 bit-exact（数学上退化为严格比较）；口径澄清保持不变：120 / 99%
来自随机采样，greedy 下不可达。

**三方 MTP 对比（谨慎口径）**：下表只收录有明确来源的数字。各行硬件同为 2×V100-SXM2-32GB，
但并行方式与 verify 实现不同，跨行只能作方向性对照，不能当作严格同配置 A/B。

| 引擎 / target | 并行 | no-spec（tok/s） | MTP（tok/s） | 来源 |
|---|---|---:|---|---|
| ninfer Flash-Next（本分支，开发中） | TP2 | 70.73–70.82 | **111–113**（draft=3 + Q4 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`，acc 0.858–0.862，约 1.60×） | 本分支多次实测，cache 15000 MiB + expert profile，0 miss，window=3，greedy，128 token / 2 reps |
| ninfer 27B（main v0.2.5 发布） | TP2 | 51.0 | 104.7（draft=3，2.74 tok/round，58% accept） | README 速度表 / `docs/vs-1cat-2026-10-16.md:67` |
| 1cat-vllm | TP2 | 49.6 | 无权威数据 | `docs/vs-1cat-2026-10-16.md:40,67`（仅 no-spec）。本地 `speed_1cat_nvfp4_mtp.json` 文件名带 `mtp` 但无 spec 字段，不足以作为 1cat MTP 数字 |
| Strata（层分 PP） | 层分 PP | 未单独给出 | ~110（`--spec 4 --mtp`，层分 PP） | Strata `NEXT_SESSION_HANDOFF.md:229-236` |

**关键洞察**：同一 ninfer 引擎、同一硬件（2×V100 TP2）下，27B 的 MTP 是 **2× 正收益**
（51.0 → 104.7），Flash-Next 也曾是**负收益**（70.73–70.82 → 51.78，早期记录为 70.54）；本轮
接线与优化后，Flash-Next MTP 达到约 **1.60× 正收益**（70.73–70.82 → 111–113），已与 Strata
（~110，层分 PP）同档。差异不在引擎、硬件或并行方式，而在 Flash-Next runtime（`qwen4_exp`）的
MTP/verify 路径——现已通过 GDN replay fold commit、verify 多列权重复用 kernel、draft TP2 角色
拆分、Q4 proposal/head 与 min_p 提前终止补齐。1cat-vllm 用 TP2 + vLLM spec 框架，Strata 用层分
PP + 定制 verify kernel，二者 MTP 都 >100 tok/s，证明 batch verify 在此硬件可行。**120 tok/s
仍未达成**，但「MTP 落后 base」的旧结论已失效；acceptance 受单层 head 的 step0 能力上限约束
（低置信度 draft 命中 19.2% vs 全 draft 96.5%），2× 需 artifact 层更强/多层 MTP head。120 与
99% 均来自随机采样口径（详见下节「口径澄清」），greedy 下不可达。

**DFlash2 并行 draft（`--spec dflash2 --draft-model <companion>`）**：本轮移植
`iamwavecut/ninfer-all` 的 DFlash2 后端（Apache-2.0，commit 9ba1276）。在 2×V100 / TP2 /
greedy / 512-token prompt / 128 decode / expert cache 15000 MiB / min_p 0.8 下实测
**DFlash2-5 加 drafter GEMM 位级不变优化后约 146–149 tok/s（acceptance 0.99，约 2.07× base）**
（仅 K=5 未优化 GEMM 时为 138.4 / 1.95×），MTP-3 为 112.2 tok/s（0.859），
plain 为 71.0 tok/s；三者的 greedy token 序列逐 token 相同（无损）。五个 draft 明显优于四个
（后者 118.8）、六到七个（135.7/133.9），故 Flash-Next 建议显式 `--draft-tokens 5`。DFlash2 需
额外 `--draft-model` 指向 companion artifact，草稿窗口 2048，eager decode（不做 CUDA Graph），
TP4 拒绝；它把 acceptance 从 MTP 的 0.86 提到 0.99，已接近 2× base。

#### 口径澄清：greedy vs 随机采样（上游证据）

本轮查证上游 `Neroued/ninfer`（同模型 **Qwen3.8-Flash-Next**）的 README 与
`docs/performance/qwen3.8-flash-next.md`：此前「120 tok/s 目标」与「99% acceptance」用的是
**decode saturation 的随机采样口径**，不能当作 greedy 契约下的可达目标。要点如下：

- **99% 只在随机采样 saturation 负载**：temperature 0.6 / top-p 0.95 / top-k 20 + 130-token
  structured JSON prompt，acceptance 99.1–99.6%。同一报告的单请求 greedy 为约 34–91%
  （Structured 91%、Code 84%、Translation 72%、AIME 50%、Story 35%）。
- **head 不是 acceptance 差异来源**：上游 optimized proposal head（`proposal/head`，
  `q4_g64_fp16`）与主 head（`text/output_head`，`bf16`）来自同一个 `lm_head.weight`；本地 artifact
  conversion report 中 `proposal/head` weight/001310 与 `text/output_head` weight/000526 的 source
  均为 `base-hf-bf16:lm_head.weight`。本轮最终配置改用 artifact 自带的同源 Q4 proposal/head 为默认
  （用户裁决：draft 是提案、正确性由 verify 保证），`NINFER_DIAG_FP16_DRAFT_HEAD=1` 保留作对照。
- **两种 acceptance 的定义不同**：本 bench 为 greedy（temperature 0）+ 严格 argmax 前缀比较
  （`speculative.cpp:40-46`），greedy acceptance = P(argmax_draft == argmax_target)；随机采样
   acceptance = Σmin(p,q) = 1-TV(p,q)，q≈p 时 top-2 近邻翻转也可接受，因而接近 1。此前 draft=1
  的 0.881 与上游 greedy Structured 91% 同口径、很接近；Q4 默认后最终配置 acceptance 为
   **0.858–0.862**（早期单点 0.860），仍在同一 greedy 口径区间（相对 FP16 对照的约 0.87 略降）。
- **结论**：120 tok/s 目标来自 27B greedy MTP3 的 104.7；上游 Flash-Next 单请求 greedy 最高
  66.8（单 Spark structured）/ 107.2（双 Spark structured），>100 的 saturation 数字属于随机采样。
  greedy 契约下 99% acceptance 不可达；**120 tok/s 仍未达成**（本轮约 1.60×，非 2×）。「MTP 不超
  base / 落后 27%」的旧结论已被本轮实测推翻（70.73–70.82 → 111–113，约 1.60×），保留该历史
  数字仅为记录，不得解读为 verify 路径存在确定性缺陷。

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
| 模型（main v0.2.5 发布） | 自转换 `qwen3_8_27b_nvfp4_f16embed.ninfer`（NVFP4 + FP16 embedding，18.0 GiB；质量/速度实测用它）与标准 `qwen3_8_27b_nvfp4.ninfer`（16.8 GiB，速度相同）；源 `unsloth/Qwen3.8-27B-NVFP4` |
| Flash-Next 模型（本分支开发中） | `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`（Qwen3.8-Flash-Next / Qwen4Exp，NVFP4 + FP8 draft bank，v3 容器，108 GB / 4 part，TP2，INT8 KV，262144 上下文） |
| 服务参数 | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## 测试用例与结果

以下全部为**本项目自身实测数据**（2026-10），不与其他引擎或 fork 对比。

### 速度（TP2，NVFP4，2×V100）

> 以下 27B 数据均为 **main 分支 v0.2.5 发布状态**；本分支 Flash-Next 的实测见本节末尾
> 「Flash-Next 分支实测速度」，完整实现与验证记录见「当前进度 → Flash-Next 分支实测」。

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

#### Flash-Next 分支实测速度（开发中，非发布产品）

条件：2×V100-SXM2-32GB、TP2、`sm_70`、CUDA 12.8、greedy、eager 单请求、window=3，128-token、
2 reps、cache 15000 MiB + 专家画像（0 miss）；最终配置 = Q4 proposal/head 默认 + draft=3 +
`NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`（变量名必须完整，漏 `NEXT` 会静默退回默认 0）。
artifact：`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`。

| 配置 | 吞吐（tok/s） | acceptance | 状态 |
|---|---:|---:|---|
| base no-spec | **70.73–70.82** | — | 多次实测参照基线（早期单点 70.54） |
| 起点（fold commit 接线前） | 30.33 | — | 本轮起点 |
| MTP 最终配置（Q4 + draft=3 + min_p=0.8 + profile） | **111–113** | 0.858–0.862 | 多次实测；**1.60× base** |
| 同配置早期实测（正常波动） | 110.3–110.5 | 0.860–0.861 | 保留历史，不写死单点 |
| MTP draft=1 + profile（旧最佳，已取代） | 51.78 | 0.881 | 历史记录：当时落后 base 约 27%，已失效 |
| MTP draft=3 + profile（min_p=0/变量名漏 `NEXT`） | 42.9 | 0.772 | 历史记录；漏字写法使生效值静默变为 0 |
| MTP batch verify（`decode_span` 实验，已回退） | 27.0 | 0.56 | 数值不等价 |

**MTP 现快过 base 约 60%（70.73–70.82 → 111–113），但 120 tok/s 目标仍未达成（约 1.60×，非 2×）**
（与上表数字一致，不重复计入 main 发布口径）。起点 30.33 tok/s 到本轮约 113，提升约 **3.6×**。
背景、口径澄清与负结果现状（无 profile 时 13.6–14 tok/s、CUDA Graph 负收益根因、GDN replay fold
已转正、B5 `kMoeMaxSmallT` 4→8 悬崖修复、环境变量拼写事故、acceptance 根因与下一步）见
「当前进度 → Flash-Next 分支实测」。

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
| **ninfer（v0.2.2）** | **43.0%（86/200）** | **43.5%（87/200）** |
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
| 4 | 2×TP2 | `"0,1;2,3"` | **第三方端到端验证通过**（32GB；Vision、MTP、前缀复用、8 并发） |
| 8 | 8×TP1 | `"0;1;2;3;4;5;6;7"` | 社区/非契约；未验证 |
| 8 | 4×TP2 | `"0,1;2,3;4,5;6,7"` | 社区/非契约；未验证（单池；8 卡机双池并存已验证） |
| 8 | 2×TP4 | `"0,1,2,3;4,5,6,7"` | 社区/非契约；未验证 |

**第三方验证（2026-11-13，8×V100-SXM2-32GB）**：4 卡 `2×TP2` 池（2 实例，每实例
`--max-concurrency 4`、`--kv-capacity 1048576`、`--kv-dtype int8`、MTP-3、
`--prefix-cache --host-cache-mib 98304`）完成端到端验证：图片问答；缓存命中图片叠加前缀恢复的
视觉 MTP 流式请求；OpenAI/Anthropic usage 缓存字段；8 并发 × 256 token 聚合 341–382 tok/s
（修复池接纳停摆前 124–152）；缓存恢复 TTFT 1.1–9 s（冷预填约 114 s）。同一 8 卡机的另一组
GPU 0-3 运行冻结的 `2×2` 生产栈（端口 8000），两池并存、端口隔离。单池 8 卡 profile 仍未验证。

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

对应脚本 profile（单池 8 卡为社区/非契约）：`bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1|4xTP2|2xTP4`
（加 `--dry-run` 只打印完整命令）。8 卡 4×TP2 / 2×TP4 以及 4 卡 4×TP1 / 2×TP2 的完整参数、
路由和验证边界见 [`docs/maintainer/multi-instance.md`](docs/maintainer/multi-instance.md)。
上述单池未验证 profile 没有吞吐或显存实测数据，不得引用为验收结果。

**社区测试请求（单池 8 卡）：** 8×V100-SXM2-32GB 上的单池 8×TP1、4×TP2、2×TP4 仍未验证；
欢迎提交启动命令、binary/artifact hash、`/metrics?format=json` 的 `instances` 数组、每实例
committed decode tok/s、TTFT/TPOT、MTP acceptance、queue wait、free KV pages、host tier
实际分配量，以及 4×TP2/2×TP4 的 32GB 显存余量。4×TP1 与 8×TP1 都需要 32GB 卡；16GB V100
放不下一份完整的 27B 权重。4 卡 `4×TP1` 也未验证，欢迎一起提交。`--dry-run` 或静态检查不等于
8 卡验证；本项目不承诺修复周期或 SLA。

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

- **Vision**：当前自转换 NVFP4 制品包含 Vision；GGUF Q4_K_M 制品拒绝 `--vision`；多实例 Vision（2×TP1）与 TP2 Vision 均已通过真实集成测试。
- 多实例：2×TP1 通过真实集成测试；4 卡 `2×TP2` 已由第三方在 8×V100-SXM2-32GB 上端到端验证（Vision、MTP、前缀复用、OpenAI/Anthropic、8 并发），8 卡机上两个独立 4 卡池并存也已验证；单池 8×TP1/4×TP2/2×TP4 与 4×TP1 仍未验证，8 卡单池为社区/非契约。4×TP1 与 8×TP1 需要 32GB 卡，16GB V100 放不下一份完整的 27B 权重。
- KV 只有每实例 device KV 和进程共享的 pinned host prefix-KV 两级；没有磁盘 KV tier、KV offload 或全局活跃 KV 池。
- prefix cache 修复后未做长时间浸泡测试；超高压并发下的缓存行为仍在观察。
- **BFCL 全量 200 条：ninfer no-spec 43.0%（86/200）、MTP-3 43.5%（87/200）**。本轮修复了 tool-call 参数类型、NVFP4 QPN 除数、prefix-reuse 的 rewrite checkpoint 复用，以及 FP16 的 gating/prefill 性能与数值路径；同一 server 重跑已可复现（见“质量”节）。

---

## 鸣谢

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — NInfer 原始项目，本仓库的引擎基座。
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) — 双卡张量并行与 YaRN 扩展。
- [jason-ni/ninfer-v100-tp4](https://github.com/jason-ni/ninfer-v100-tp4) — V100 TP4 单进程并行基底（本仓库代码直接来源）。
- [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) — V100 上的 NInfer 单卡部署与跨引擎基准 fork（调研参考，未直接复用其代码）。
- [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) — Flash-Next DFlash2 并行 draft 后端的移植来源（Apache-2.0，commit 9ba1276）。
- [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) — Flash-Next DFlash2 草稿模型来源。
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

## Branch Status (read first)

This branch is **`feat/qwen3-8-flash-next`**, a Flash-Next development branch based on main's
**v0.2.5 (27B NVFP4 release)**. The main v0.2.5 27B release content is retained below and explicitly
marked as "main branch release status"; **the subject of this branch is Flash-Next—an in-development
target, not a released product**.

- **Model and terminology**: Flash-Next is **Qwen3.8-Flash-Next** (Qwen4Exp series, also written
  `qwen3.8-flash-next`): 48 layers (36 GDN + 12 QSA), 512 routed experts, NVFP4 weights + FP8 draft
  bank, TP2, INT8 KV, 262144 context.
- **Current MTP result (recorded honestly)**: Flash-Next MTP now beats base—base no-spec
  **70.73–70.82 tok/s**; the final config (Q4 proposal/head default + draft=3 +
  `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + expert profile, 0 miss, TP2 / `sm_70` / CUDA 12.8 / greedy /
  eager single request / window=3 / 128 token / 2 reps) measures **111–113 tok/s** with acceptance
  **0.858–0.862**, i.e. **1.60× base**. The earlier 110.3–110.5 tok/s / 0.860–0.861 and base 70.54
  tok/s are same-config early measurements within normal fluctuation, kept for the record rather than
  as a single point. The **120 tok/s goal is still not met** (2× base = 141, about 1.60×, not 2×).
  The acceptance root cause is that low-confidence drafts are themselves wrong under min_p=0.8:
  rounds emitting only one draft hit **41/213 = 19.2%**, rounds emitting three drafts hit
  **1014/1051 = 96.5%**, overall step0 1086/1366 = 79.5%—an economic tradeoff, not a bug; without an
  artifact change 0.79 cannot be broken, and 2× needs a stronger / multi-layer MTP head at the
  artifact layer. The old "MTP behind base / net loss" conclusion is invalid—the starting point was
  30.33 tok/s (before fold-commit wiring), and this round improves 30 → ~113, about **3.6×**. The
  variable must be spelled in full as `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`; omitting `NEXT` silently
  falls back to the default 0 and invalidates cross-time A/B comparisons. 120 / 99% are random-sampling
  caliber and unreachable under greedy (see "Caliber Clarification"). See "Current Status → Flash-Next
  Branch Measurements" and the "Speed" section.

---

## Project Statement (read first)

1. **All code and documentation in this repository are generated by AI.** The repository owner has no large-model technical background and does not understand the model algorithms or the internals of the inference engine. **Please do not ask the owner technical questions**—the owner cannot answer them and has no time to reply.
2. **Problems, reproductions, and community collaboration go through GitHub Issues/Discussions and the repository documents.** Reproducible quality reports, logits evidence, and conversion checks are welcome; include steps, logs, hardware/software versions, and environment details. No fix timeline or SLA is promised.
3. This repository is a **new project**: improved NInfer code is imported with a clean history, without upstream or other fork commit history. See “Acknowledgements” and `NOTICE` for upstream and modification history.

---

## Release

Current release: **v0.2.5 (2026-11-13)** (first public release v0.1.0, 2026-10-27). See [`CHANGELOG.md`](CHANGELOG.md) for release notes and known limitations. v0.2.5 is the **main branch's release status**; this branch, `feat/qwen3-8-flash-next`, adds the in-development Flash-Next target on top of it (see "Branch Status" at the top) and does not produce a new release.

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
| **Flash-Next target (in development, not a released product)** | Qwen3.8-Flash-Next / Qwen4Exp: 48 layers (36 GDN + 12 QSA), 512 routed experts, NVFP4 weights + FP8 draft bank, TP2, INT8 KV, 262144 context; artifact `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` (v3 container, 108 GB in 4 parts) |

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
| 1 | The Vision tower of an existing NVFP4 artifact is incompatible with this fork's contract (quantization format, no separate qkv, RoPE attention); loading crashes immediately | Do not use that artifact; the current self-converted NVFP4 artifact includes Vision, while the GGUF Q4_K_M artifact rejects Vision. Multi-instance Vision (2×TP1) and TP2 Vision have both passed real integration tests; see `TEMP_WORKAROUNDS.md` |
| 2 | With prefix cache enabled, multi-turn conversations trigger `Paged KV trim extent would discard cache-owned pages`; all subsequent requests return 503 and the engine becomes unusable | Two fixes: (1) when `trim_pages` encounters cache-owned pages, shrink the trim range instead of throwing; (2) `promote_page_at` returns idempotently for pages already owned by the cache |
| 3 | Suspected that slow decode was caused by the attention split strategy; tried `LONG_SPLIT_KEYS=1920` | A/B showed no effect; confirmed the bottleneck is elsewhere and stopped tuning this direction |
| 4 | Decode speed measurements always included prefill time, distorting the numbers | Switched to warmup + multiple iterations + server-reported timings to separate the measurements |
| 5 | Building without root requires ffmpeg/libcurl headers | Use `apt-get download` + `dpkg-deb -x` to unpack locally, and point `PKG_CONFIG_SYSROOT_DIR` at the local prefix; or use `-DNINFER_SKIP_FFMPEG=OFF` as described in the installation section |
| 6 | Under the FP16 text runtime, Vision requests return 500 `scatter: src/dst must share BF16 or FP16 dtype` (the Vision tower outputs BF16 while the text residual is FP16) | Add a BF16→activation-dtype staging buffer before scattering Vision embeddings into the text residual, covering TP1, TP2, and multi-instance paths; both 2×TP1 lanes and TP2 single-instance were measured returning correct results |
| 7 | With a device list that does not start at 0 (for example `--tp 2 --devices 4,5`), TP mailbox crashes because P2P enablement passes the rank index as the CUDA device ordinal | `cudaDeviceEnablePeerAccess` now uses each rank's real device ordinal; the self-check passes on `--devices 4,5` and selects the nvlink mailbox |

See [`TEMP_WORKAROUNDS.md`](TEMP_WORKAROUNDS.md) for all temporary changes and review items.

---

## Current Status (main branch v0.2.5 release status, as of 2026-11-13)

> This section records the **main branch's 27B v0.2.5 release status**, not a conclusion about this
> branch's Flash-Next target. Flash-Next measurements are in "Flash-Next Branch Measurements" below
> and in the "Speed" section.

This version adds the FP16 text runtime on top of v0.1.0, fixes BFCL multi-turn quality,
removes the FP16 path's decode/prefill regression, fixes the Vision `scatter` dtype
failure under the FP16 text runtime (verified at TP1, TP2, and multi-instance), and fixes
the TP mailbox P2P setup that crashed when `--devices` did not start at 0. v0.2.3 fixes
warm-home routing against the reuse-blind gross page bill and adds the host-tier capacity
formula and allocation-failure metrics; v0.2.4 removes the multi-instance admission stall
behind a decoding instance by reading each candidate non-blockingly and skipping a busy
instance for that round; v0.2.5 reports prefix-cache reuse in the OpenAI and Anthropic usage
payloads and fixes the Vision MTP bridge rmsnorm dtype crash under the FP16 text runtime.
Third-party validation completed a 4-card `2×TP2` pool and an 8-card host running two such pools
side by side. BFCL `multi_turn_base`, full 200:
**no-spec 86/200 = 43.0%, MTP-3 87/200 = 43.5%**. MTP-3 decode at four points (262144 context, INT8 KV, FP16-embedding artifact):
**104.7 / 98.9 / 109.6 / 98.1 tok/s**.

**Speed and quality optimization has essentially reached this hardware platform's limit (a conclusion about the main-branch 27B v0.2.5 release only, not about this branch's in-development Flash-Next target).**

- Decode: MTP-3 is 104.7/98.9/109.6/98.1 tok/s at the four context points, at or above the
  reference table (106.8/99.2/99.7/84.4); MTP acceptance is 52–66%, and further draft or
  sampling changes are in the plateau.
- Prefill: 1826 tok/s, on par with the BF16 baseline of 1825; the splitd, prefill GEMM, and
  attention-split knobs have each been A/B'd to a local optimum.
- Quality: no-spec 43.0%, MTP-3 43.5% (pre-fix baseline 36.0%); the 44% -> 43% delta is just
  four n200 sample flips (net -2), sampling noise rather than a precision regression.

The remaining items (the 44% -> 43% borderline sampling and the untested DFlash path) are
marginal, not large levers; further gains need different hardware or a changed product model.
See the "optimizable point" note in the Quality section.

### Flash-Next Branch Measurements (in development, not a released product)

This section is new to the `feat/qwen3-8-flash-next` branch and is kept separate from the main 27B
release data above. Flash-Next is **Qwen3.8-Flash-Next** (Qwen4Exp series, also written
`qwen3.8-flash-next`): 48 layers (36 GDN + 12 QSA), 512 routed experts, NVFP4 weights + FP8 draft
bank, TP2, INT8 KV, 262144 context. Artifact:
`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` (v3 container, 108 GB in 4 parts).

**Conditions**: 2×V100-SXM2-32GB, TP2, `sm_70`, CUDA 12.8, greedy, eager single request, window=3,
128-token, 2 reps, cache 15000 MiB + expert profile, 0 miss. Final config = Q4 proposal/head default
+ draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` + expert profile, 0 miss (the variable name must be
complete; omitting `NEXT` silently falls back to the default 0, see the environment-variable incident
below).

| Configuration | Throughput (tok/s) | acceptance | Status |
|---|---:|---:|---|
| base no-spec | **70.73–70.82** | — | multi-run reference baseline (early single point 70.54) |
| Starting point (before fold-commit wiring) | 30.33 | — | effort start; MTP not wired |
| MTP final config (Q4 + draft=3 + min_p=0.8 + profile) | **111–113** | 0.858–0.862 | multi-run; **1.60× base** |
| Same config, early measurements (normal fluctuation) | 110.3–110.5 | 0.860–0.861 | retained, not a single point |
| MTP draft=1 + profile (old best, superseded) | 51.78 | 0.881 | historical: was 27% behind base, now invalid |
| MTP draft=3 + profile (min_p=0 / variable name missing `NEXT`) | 42.9 | 0.772 | historical: improved by min_p=0.8 + Q4; the misspelling silently sets the effective value to 0 |
| MTP batch verify (`decode_span` experiment, rolled back) | 27.0 | 0.56 | numerically inequivalent, rolled back |

**Conclusion (recorded honestly)**: the final config measures **111–113 tok/s** with acceptance
**0.858–0.862**, i.e. about **1.60×** base **70.73–70.82**; earlier same-config 110.3–110.5 tok/s /
0.860–0.861 are normal fluctuation, not a different configuration, and the old "MTP about 27% behind
base" conclusion is invalid—the starting point was 30.33 tok/s (before fold-commit wiring), and this
round improves 30 → ~113, about **3.6×**. **The 120 tok/s goal is still not met** (2× base = 141,
about 1.60×, not 2×), and must not be read as having reached 2×. The acceptance root cause
(`debug-step0-acceptance.md`, L4): `accepted_per_position` is a conditional count; the apparent
"0.79→0.93→0.97 rise" is min_p early-stop selection bias, and the true step1 conditional rate is
0.876. Under min_p=0.8 low-confidence drafts are themselves wrong—rounds emitting only one draft hit
**41/213 = 19.2%**, rounds emitting three drafts hit **1014/1051 = 96.5%**, overall step0 1086/1366 =
79.5%; this is an economic tradeoff, not a bug. Q4 quantization is not the cause: 96-step Q4 vs FP16
argmax agreement **89/96 = 92.7%**, end-to-end accepted_tokens Q4 3071 vs FP16 3072. Disproven
acceptance causes: draft QSA drift (correct-caliber Δacc 0.0012), T=6 verify numerical regression
(bit-exact), head quantization (NVFP4/Q4 no gain, FP16 net-negative throughput). Conclusion: 0.79
cannot be broken without an artifact change; 2× needs a stronger / multi-layer MTP head at the
artifact layer (artifact contract change). Without a profile, MTP reaches only **13.6–14 tok/s**
(cache hit 0.85, about 15,000 misses); the expert profile (converted from Strata's `expert-profile.bin`
for the same model to `tools/bench/expert-profile-flash-next.txt`) raises the hit rate to 1.0 / 0 miss,
an early key step.

**This round's implementation (commit `b0b79c4b`)**: GDN replay fold commit moved to the production
path (Tasks 1–10, bit-exact verified, superseding the old "11.36 tok/s rolled back" record); verify
multi-column weight-reuse kernels (hc_mix / linear / W8 / route / conv / gdn_proj / GDN stage+extract);
draft TP2 role split; Q4 proposal/head default (same-source weights carried by the artifact); draft
confidence + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` early stop; sink stream sync. The earlier `91b45ae5`
work—quantized resident draft bank (5.03→2.52 GB), D2D snapshot for the MTP state transaction, T≤4
cached routed/shared MoE + FP16 small-T GEMM, and the expert-profile path
(`--expert-cache-profile`)—remains the foundation.

**Verification**: build OK, `git diff --check` clean; mtp_verify / fold_commit / mtp_state /
gdn_replay_storage pass; draft / chain / confidence / greedy_equivalence SKIP (missing hidden fixture,
kernel oracle still runs); historically 38/38 tests pass (`ctest -R "flash_next|mtp|moe_nvfp4"`),
greedy NLL 11751, mean_nll 3.732. User ruling: accept Q4 as default—draft is a proposal, correctness
is guaranteed by verify; the chain oracle changes from `{220,27967,813}` to `{220,27967,53}` (Q4
semantics), greedy equivalence (MTP on/off) is token-by-token identical; the FP16 control switch
`NINFER_DIAG_FP16_DRAFT_HEAD=1` is retained and restores `{220,27967,813}`.

**Unmet / negative results (status annotated; disproven items retained but no longer levers)**:
- **Disproven acceptance causes (do not retry)**: draft-head QSA cross-round drift (correct-caliber
  rerun ON/OFF Δacc 0.0012), T=6 verify numerical regression (bit-exact), head quantization
  (NVFP4/Q4 no gain, FP16 net-negative throughput). Draft expert 7-kernel merging
  (`debug-b1-rootcause.md`) and cross-round gating (`debug-b3-signal.md`) are also measured and
  excluded, and are no longer incremental levers.
- **B5 cliff fixed (commit `5286f982`)**: `kMoeMaxSmallT` 4→8 removed the window≥5 MoE 10× fallback;
  window=5 goes from 20.9 → 106.3 tok/s, but acceptance drops to 0.79, still below window=3, so
  window=3 stays.
- **Environment-variable spelling incident (`debug-acceptance-drift.md`)**: one experiment used
  `NINFER_FLASH_DRAFT_MIN_P` (missing `NEXT`), silently setting the effective min_p to 0 and
  invalidating several cross-time A/B comparisons; gate JSON does not record the effective min_p, so
  every A/B must explicitly record the variable name `NINFER_FLASH_NEXT_DRAFT_MIN_P` and its value 0.8.
- **Root-cause audit (`analysis-rootcause-audit.md` UPDATE)**: of 27 excluded items, L4 = 6 and 21
  remain <L4; "excluded" ≠ "root cause found", and the highest-risk items have been backfilled.
- **CUDA Graph gives no benefit for MTP** (51.85 vs 51.78; whole-round graph A/B −4.4%): root cause
  located and conclusion retained—on sm_70 pre-Ampere `cudaGraphLaunch`'s O(N) per-node CPU submission
  cost (3,054 nodes → ~0.6 ms/round) exceeds the host gaps it removes, which are already hidden by
  device execution; Strata uses graph on the same V100 by relying on kernel-internal t-weight reuse (an
  eager-side property), a benefit our multi-column kernels already capture in eager mode. The
  negative result is explained, not caused by slower kernels.
- **GDN replay fold path**: the old 11.36 tok/s < 14 rollback record is outdated—fold commit is now
  the production path (Tasks 1–10), bit-exact verified.
- **batch verify `decode_span`**: numerically inequivalent (0.56 acceptance), rolled back; strict
  argmax under greedy still requires bit-exact comparison.
- Layer-split / parallel architecture is unchanged (the main model stays at TP2).

**Next steps**: 120 tok/s (2× base = 141) is still not met; without an artifact change step0 0.79
cannot be broken, and 2× needs a stronger / multi-layer MTP head at the artifact layer (artifact
contract change), not the already-excluded engineering levers. Probabilistic rejection sampling cannot
replace bit-exact comparison under greedy (mathematically it degenerates to strict comparison); the
caliber clarification stands: 120 / 99% come from random sampling and are unreachable under greedy.

**Three-way MTP comparison (strict sourcing)**: the table includes only figures with an explicit
source. All rows use the same 2×V100-SXM2-32GB hardware, but parallelism and verify implementations
differ, so cross-row numbers are directional rather than a strict same-configuration A/B.

| Engine / target | Parallelism | no-spec (tok/s) | MTP (tok/s) | Source |
|---|---|---:|---|---|
| ninfer Flash-Next (this branch, in development) | TP2 | 70.73–70.82 | **111–113** (draft=3 + Q4 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`, acc 0.858–0.862, about 1.60×) | this branch, multi-run: cache 15000 MiB + expert profile, 0 miss, window=3, greedy, 128 token / 2 reps |
| ninfer 27B (main v0.2.5 release) | TP2 | 51.0 | 104.7 (draft=3, 2.74 tok/round, 58% accept) | README speed table / `docs/vs-1cat-2026-10-16.md:67` |
| 1cat-vllm | TP2 | 49.6 | no authoritative data | `docs/vs-1cat-2026-10-16.md:40,67` (no-spec only). Local `speed_1cat_nvfp4_mtp.json` has `mtp` in its name but no spec field, so it cannot back a 1cat MTP figure |
| Strata (layer-split PP) | layer-split PP | not given separately | ~110 (`--spec 4 --mtp`, layer-split PP) | Strata `NEXT_SESSION_HANDOFF.md:229-236` |

**Key insight**: with the same ninfer engine and the same hardware (2×V100 TP2), 27B MTP is a **2×
win** (51.0 -> 104.7), and Flash-Next was once a **net loss** (70.73–70.82 -> 51.78, early record
70.54); after this round's wiring and optimization, Flash-Next MTP reaches about a **1.60× win**
(70.73–70.82 -> 111–113), on par with Strata (~110, layer-split PP). The difference is not the
engine, hardware, or parallelism — it is Flash-Next's runtime (`qwen4_exp`) MTP/verify path, now
completed via GDN replay fold commit, verify multi-column weight-reuse kernels, draft TP2 role split,
Q4 proposal/head, and min_p early stop. 1cat-vllm uses TP2 + the vLLM spec framework and Strata uses
layer-split PP + a custom verify kernel; both exceed 100 tok/s with MTP, showing batch verify is
feasible on this hardware. **120 tok/s is still not met**, but the "MTP behind base" conclusion is
invalid; acceptance is bounded by the single-layer head's step0 capability (low-confidence drafts hit
19.2% vs full drafts 96.5%), and 2× needs a stronger / multi-layer MTP head at the artifact layer.
Both 120 and 99% come from the random-sampling caliber (see the "Caliber Clarification" subsection
below) and are unreachable under greedy.

**DFlash2 parallel drafting (`--spec dflash2 --draft-model <companion>`)**: this round ports the
DFlash2 backend from `iamwavecut/ninfer-all` (Apache-2.0, commit 9ba1276). On 2×V100 / TP2 / greedy
/ 512-token prompt / 128 decode / 15000 MiB expert cache / min_p 0.8 it measures **DFlash2-5 with
the bit-identical drafter-GEMM tuning at about 146–149 tok/s (acceptance 0.99, ~2.07× base)**
(138.4 / 1.95× with K=5 alone), against MTP-3 at 112.2 tok/s (0.859) and plain at 71.0 tok/s; all
three emit token-identical greedy sequences (lossless). Five drafts clearly beat four (118.8) and
six/seven (135.7/133.9), so Flash-Next should pass `--draft-tokens 5` explicitly. DFlash2
additionally needs `--draft-model` pointing at the companion artifact, uses a 2048-token draft
window, decodes eagerly (no CUDA Graph), and rejects TP4; it lifts acceptance from MTP's 0.86 to
0.99 and now exceeds 2× base.

#### Caliber Clarification: Greedy vs Random Sampling (Upstream Evidence)

This round checked upstream `Neroued/ninfer` (same model, **Qwen3.8-Flash-Next**): its README and
`docs/performance/qwen3.8-flash-next.md` show that the earlier "120 tok/s goal" and "99% acceptance"
were calibrated on **random sampling in decode saturation** and are not reachable targets under the
greedy contract.

- **99% only with random-sampling saturation load**: temperature 0.6 / top-p 0.95 / top-k 20 plus a
  130-token structured JSON prompt yields 99.1–99.6% acceptance. Single-request greedy figures in the
  same report are about 34–91% (Structured 91%, Code 84%, Translation 72%, AIME 50%, Story 35%).
- **The head is not the source of the acceptance gap**: upstream's optimized proposal head
  (`proposal/head`, `q4_g64_fp16`) and main head (`text/output_head`, `bf16`) come from the same
  `lm_head.weight`; the local artifact conversion report lists both `proposal/head` weight/001310 and
  `text/output_head` weight/000526 with source `base-hf-bf16:lm_head.weight`. The final config makes
  the artifact-native same-source Q4 proposal/head the default (user ruling: draft is a proposal,
  correctness is guaranteed by verify), with `NINFER_DIAG_FP16_DRAFT_HEAD=1` kept as a control.
- **The two acceptance numbers are different quantities**: this bench uses greedy (`temperature 0`)
  with strict argmax-prefix comparison (`speculative.cpp:40-46`), so greedy acceptance is
  P(argmax_draft == argmax_target); random-sampling acceptance is Σmin(p,q) = 1-TV(p,q), where a
  top-2 neighbor flip is still accepted when q≈p, so the value approaches 1. The earlier draft=1
  0.881 was the same caliber as upstream's greedy Structured 91% and very close to it; with Q4 as
  default the final config's acceptance is **0.858–0.862** (early single point 0.860), still in the
  same greedy caliber (slightly below the ~0.87 FP16 control).
- **Conclusion**: the 120 tok/s goal came from the 27B greedy MTP3 figure of 104.7; upstream
  Flash-Next single-request greedy tops out at 66.8 (single Spark structured) / 107.2 (dual Spark
  structured), while the >100 saturation figures use random sampling. Under the greedy contract 99%
  acceptance is unreachable; **120 tok/s is still not met** (about 1.60× this round, not 2×). The old
  "MTP not beating base / 27% behind" conclusion has been overturned by this round's measurements
  (70.73–70.82 -> 111–113, about 1.60×); the historical figures are retained for the record only and
  must not be read as a deterministic defect in the verify path.

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
| Model (main v0.2.5 release) | Self-converted `qwen3_8_27b_nvfp4_f16embed.ninfer` (NVFP4 + FP16 embedding, 18.0 GiB; used for the quality/speed measurements) and the standard `qwen3_8_27b_nvfp4.ninfer` (16.8 GiB, same speed); source `unsloth/Qwen3.8-27B-NVFP4` |
| Flash-Next model (in development on this branch) | `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` (Qwen3.8-Flash-Next / Qwen4Exp, NVFP4 + FP8 draft bank, v3 container, 108 GB / 4 parts, TP2, INT8 KV, 262144 context) |
| Server flags | `--tp 2 --devices 0,1 --kv-dtype int8 --max-context 262144` |

## Test Cases and Results

All data below are **measured on this project** (October 2026) and are not comparisons against other engines or forks.

### Speed (TP2, NVFP4, 2×V100)

> The 27B data below are all **main branch v0.2.5 release status**; Flash-Next measurements for this
> branch are at the end of this section ("Flash-Next Branch Measurements Speed"), with the full
> implementation and verification record under "Current Status → Flash-Next Branch Measurements".

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

#### Flash-Next Branch Measurements Speed (in development, not a released product)

Conditions: 2×V100-SXM2-32GB, TP2, `sm_70`, CUDA 12.8, greedy, eager single request, window=3,
128-token, 2 reps, cache 15000 MiB + expert profile (0 miss); final config = Q4 proposal/head default
+ draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` (the variable name must be complete; omitting `NEXT`
silently falls back to the default 0). Artifact:
`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`.

| Configuration | Throughput (tok/s) | acceptance | Status |
|---|---:|---:|---|
| base no-spec | **70.73–70.82** | — | multi-run reference baseline (early single point 70.54) |
| Starting point (before fold-commit wiring) | 30.33 | — | effort start |
| MTP final config (Q4 + draft=3 + min_p=0.8 + profile) | **111–113** | 0.858–0.862 | multi-run; **1.60× base** |
| Same config, early measurements (normal fluctuation) | 110.3–110.5 | 0.860–0.861 | retained, not a single point |
| MTP draft=1 + profile (old best, superseded) | 51.78 | 0.881 | historical: was 27% behind base, now invalid |
| MTP draft=3 + profile (min_p=0 / variable name missing `NEXT`) | 42.9 | 0.772 | historical; the misspelling silently sets the effective value to 0 |
| MTP batch verify (`decode_span` experiment, rolled back) | 27.0 | 0.56 | numerically inequivalent |

**MTP now beats base by about 60% (70.73–70.82 → 111–113), but the 120 tok/s goal is still not met
(about 1.60×, not 2×)** (same numbers as the CN table above, not counted in the main release figures).
The starting point was 30.33 tok/s and this round reaches ~113, about **3.6×**. For background, the
caliber clarification, and the status of negative results (13.6–14 tok/s without a profile, CUDA
Graph negative-result root cause, GDN replay fold now in production, B5 `kMoeMaxSmallT` 4→8 cliff fix,
environment-variable spelling incident, acceptance root cause and next steps), see "Current Status →
Flash-Next Branch Measurements".

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
| **ninfer (v0.2.2)** | **43.0% (86/200)** | **43.5% (87/200)** |
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
| 4 | 2×TP2 | `"0,1;2,3"` | **third-party end-to-end validation passed** (32GB; Vision, MTP, prefix reuse, 8 concurrent) |
| 8 | 8×TP1 | `"0;1;2;3;4;5;6;7"` | community/out-of-contract; unverified |
| 8 | 4×TP2 | `"0,1;2,3;4,5;6,7"` | community/out-of-contract; unverified (single pool; the 8-card dual-pool host is validated) |
| 8 | 2×TP4 | `"0,1,2,3;4,5,6,7"` | community/out-of-contract; unverified |

**Third-party validation (2026-11-13, 8×V100-SXM2-32GB):** a 4-card `2×TP2` pool (two instances,
each `--max-concurrency 4`, `--kv-capacity 1048576`, `--kv-dtype int8`, MTP-3,
`--prefix-cache --host-cache-mib 98304`) passed end-to-end: image question answering; a streaming
vision MTP request whose cached images ride a prefix restore; OpenAI/Anthropic usage cache fields;
8 concurrent × 256 tokens at 341–382 tok/s aggregate (124–152 before the pool-admission stall fix);
cache-restore TTFT 1.1–9 s (cold prefill about 114 s). The other GPU pair on the same 8-card host
ran a frozen `2×2` production stack (port 8000), and the two pools coexisted with port isolation.
Single-pool 8-card profiles remain unverified.

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

Script profiles (single-pool 8-card is community/out-of-contract): `bash tools/v100/bench_multi_instance_8gpu.sh 8xTP1|4xTP2|2xTP4`
(add `--dry-run` to print the full command). See
[`docs/maintainer/multi-instance.md`](docs/maintainer/multi-instance.md) for the 8xTP1, 4xTP2,
2xTP4, 4xTP1, and 2xTP2 parameters, routing, and validation boundaries. These unverified
single-pool profiles have no measured throughput or memory result and are not acceptance evidence.

**Community request (single-pool 8 cards):** on an 8×V100-SXM2-32GB host, single-pool 8xTP1, 4xTP2,
and 2xTP4 are still unverified. Please report the launch command, binary/artifact hashes, the
`/metrics?format=json` `instances` array, per-instance committed decode tok/s, TTFT/TPOT, MTP
acceptance, queue wait, free KV pages, actual host-tier allocation, and 32GB memory headroom for
4xTP2/2xTP4. 4xTP1 and 8xTP1 need 32GB cards; a 16GB V100 cannot hold one full 27B copy. The 4-card
4xTP1 profile is also unverified and welcome. `--dry-run` or static checks are not 8-card
validation; no fix timeline or SLA is promised.

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

- **Vision**: the current self-converted NVFP4 artifact includes Vision; the GGUF Q4_K_M artifact rejects `--vision`; multi-instance Vision (2×TP1) and TP2 Vision have both passed real integration tests.
- Multi-instance: 2×TP1 has passed a real integration test; the 4-card 2×TP2 profile passed third-party end-to-end validation on an 8×V100-SXM2-32GB host (Vision, MTP, prefix reuse, OpenAI/Anthropic, 8 concurrent), and the same host ran two independent 4-card pools side by side. Single-pool 8xTP1/4xTP2/2xTP4 and 4xTP1 remain unverified; the single-pool 8-card profiles are community/out-of-contract. 4×TP1 and 8×TP1 require 32GB cards; a 16GB V100 cannot hold one full 27B copy.
- KV has two tiers only: per-instance device KV and the process-shared pinned host prefix-KV tier. There is no disk KV tier, KV offload, or global active-KV pool.
- After the prefix-cache fix, no long soak test has been run; cache behavior under very high concurrency is still being observed.
- **BFCL full 200: ninfer no-spec 43.0% (86/200), MTP-3 43.5% (87/200)**. This round fixed tool-call argument typing, the NVFP4 QPN divisor, the rewrite-checkpoint prefix reuse, and the FP16 gating/prefill performance and numerical paths; same-server reruns are now reproducible (see the “Quality” section).

---

## Acknowledgements

- [Neroued/ninfer](https://github.com/Neroued/ninfer) — the original NInfer project and the engine base of this repository.
- [wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m) — dual-GPU tensor parallelism and YaRN extension.
- [jason-ni/ninfer-v100-tp4](https://github.com/jason-ni/ninfer-v100-tp4) — V100 TP4 single-process parallel base (direct source of this repository's code).
- [dollarwong/ninfer-v100](https://github.com/dollarwong/ninfer-v100) — NInfer single-GPU deployment and cross-engine benchmark fork on V100 (research reference; its code was not directly reused).
- [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) — source of the ported Flash-Next DFlash2 parallel-drafting backend (Apache-2.0, commit 9ba1276).
- [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) — source of the Flash-Next DFlash2 drafter model.
- [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) — NVFP4 quantized weight source for the default model.
- [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) — reference for the `.ninfer` artifact format (incompatible with this fork's contract; format reference only).
- All contributors to the Qwen / NInfer ecosystem.

## License

This repository's code is distributed under the applicable license terms and **inherits the Apache License 2.0**. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for the full text. Model artifacts are licensed under their publishing repositories (Hugging Face).

---

*This README and all code and documentation in this repository are generated and maintained by AI (see “Project Statement”).*
