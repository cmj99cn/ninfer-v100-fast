# Qwen3.8-Flash-Next (Qwen4Exp) NVFP4 on 2×V100-SXM2-32GB — 设计文档

日期：2026-11-11。状态：待批准。

## 1. 意图与背景

让本仓支持 `Qwen3.8-Flash-Next`（Qwen4Exp 架构，NVFP4 混合精度工件），在
2×Tesla V100-SXM2-32GB（sm_70、CUDA 12.8）上以 TP1/TP2 出 token，
**解码吞吐必须明显超过 Strata 在同款 V100 上的实测数字**。上游
`Neroued/ninfer` 已支持该模型，但仅面向 `sm_120a`（RTX 5090 + CPU expert
offload）与 `sm_121a`（DGX Spark）；本仓是 Volta 路线，不能拷贝，需要按本仓
架构重新落地。

Strata 对标基线（同款卡、来源 `Niko1221/Strata` 文档与社区实测）：

| 配置 | 解码 |
|---|---:|
| V100-PCIE-32GB + IQ3_XXS + MTP（校准后） | 22–33.5 tok/s |
| V100-SXM2-32GB + UD-IQ4_XS + MTP3 | 38–51 tok/s |
| V100-SXM2 + 其 Volta 专用 kernel（STRATA_SM70_TABLE） | 再 +10–15% |

**成功线**：在 2×V100-SXM2-32GB 上稳定超过 60 tok/s（约为 Strata 同卡最好
数字的 1.2–1.5 倍），MTP 开启、262144 容量、长会话不掉速。低于 Strata 即失败。

质量约束（用户明确）：**不使用低比特量化（Q2/IQ2/IQ3 级别）**；只接受
NVFP4 档位（约等于 Q4 质量）。

## 2. 目标 / 非目标 / 反目标

**目标**
- 注册身份 `qwen3.8-flash-next/nvfp4`，走公共 `.ninfer` Engine 路线（CLI、
  OpenAI/Anthropic serving、bench、prefill/decode/MTP）。
- 直接消费上游官方工件 `neroued/Qwen3.8-Flash-Next-NInfer`
  （`qwen3_8_flash_next.ninfer` + 3 个 part，共 101 GiB）。
- TP1 与 TP2（2×32GB）。TP2 是验收形态。
- MTP（1..5 草稿，验收用 MTP-3）、INT8 group-64 主 KV、262144 容量。
- 专家缓存：设备驻留窗口 + host RAM 溢出支撑 + 逐层静态热度优先级。

**非目标（本期不做）**
- Vision（多模态）：M5 发布后作为 M6 增量补上（用户 2026-11-11 确认）。
  工件内含 Vision 资源，补做无需返工。
- GLM-5.3-Flash、Qwen3.6-35B-A3B 的 DFlash 变更、其他新 checkpoint。
- CPU 计算专家补集（upstream 5090 profile 的 hybrid 路径）。
- FP8 KV、NF4 KV、CUDA-Graph 捕获 MoE 解码（沿用 eager decode，与 27B TP2
  现状一致）。
- 4×16GB legacy 路径上的 Flash-Next。

**反目标（技术上可行也算失败）**
- 以 IQ2/IQ3 低比特包作为产品身份。
- 运行时家族选择、模型图框架、通用插件机制。
- 把 Qwen4Exp 塞进 Qwen3.6 家族当 delta（见 §5.1，按契约建第二个家族）。
- 明显低于 Strata 的吞吐。

## 3. 环境与资源约束

- GPU：2×V100-SXM2-32GB，`sm_70`，CUDA 12.8，NVLink P2P 可用（现有
  `CollectiveComms` 走 NCCL/NVLink）。
- Host：94GB RAM。跑 Flash-Next 前停掉常驻 27B 推理进程，可用约 90GB。
  预算：专家 host 支撑 ~14GiB + PLE 磁盘行缓存 + pinned 槽 + 前端/状态，
  余量充足。
- 磁盘：≥300GB 可用。工件 101 GiB 一次性下载，足够。
- 主机 CPU 占用上限 ~85%（构建并发据此配额）。
- 客观事实（来自上游源码/清单，已被两路侦察核实）：
  - 48 层 = 36 GDN + 12 QSA（`l % 4 == 3` 为 QSA，即 `indexed_attention`）；
  - 每层 MoE：512 路由专家、top-10、`norm_topk_prob=true`、共享专家 1 个；
    expert intermediate 640；
  - hidden 2560，HC（hyper-connection）4 流残差、lowrank 320，
    残差宽 10240；词表 248320；
  - PLE 仅第 1 层：128 shard × [2500012, 160]，NF4（`nf4_g16_e4m3fn`），
    约 28.8GB；行地址 = 2–3 原始 token ID 窗口的 host hash（16 头）；
  - MTP：1 层 QSA 私有层 + 块 FP8 专家库 + 全词表 Q4 草头（优化版），
    草稿窗口 1..5；
  - 上下文 262144；GDN 循环态 FP32；QSA 主 K/V 完整保留，indexer 做 4-token
    压缩块打分（top ≤512 块，全 query 头共享选择）。

**工件获取（决定）**：官方 4 文件经 `HF_ENDPOINT=https://hf-mirror.com`
后台下载到 `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/`（约 101 GiB，
镜像速率预计 ~1 小时/数小时）。M1 的实物对表以该目录为准。

## 4. 权重与容器方案（决定）

**决定：扩展本仓 Reader 支持完整的 v3 多分片容器，直接消费官方工件，不写
转换管线，不自造容器。**

理由：磁盘上唯一可再生的形态就是官方 4 分片工件；任何"重新转换"都要
200GB+ 工作区并引入一次有损风险；本仓 Reader 已有 27B 的窄 v3 适配
（单文件投影），把"窄适配"升级为"通用 v3（入口 + 续卷逻辑地址空间）"
是同一条演进线，且 `src/artifact` 的职责本来就是通用 framing。

需要扩展的 Reader/格式面（全部落在 `src/artifact` + `src/core`，并同步更新
`docs/maintainer/artifact-container.md`）：

| 项 | 现状 | 需要 |
|---|---|---|
| v3 多分片 | 仅单文件窄适配 | 入口 + `NINPRT` 续卷的逻辑 payload 地址空间；按 16B artifact_id 校验 |
| `nvfp4` batched 块缩放 | 有单矩阵 `blockscale-k16-m128x4-v1` | `batched_block_scale_k16_m128x4_v1`（专家库：代码/高半字节/缩放三平面 + 专家步长） |
| `q8_g32_fp16` | 无（有 `W8G32_F16S`，语义待对表） | Q8 A16 组 32、FP16 组缩放；若与 W8G32 逐位同构则复用，否则新增格式 |
| `nf4_g16_e4m3fn` | 无 | NF4 固定 codebook + E4M3 组缩放（PLE 表） |
| `fp8_e4m3fn_block128x128_bf16` | 无（只有行缩放 FP8） | 块 128×128 FP8、BF16 块缩放（MTP 专家库） |
| `bf16`/`fp32`/`q4_g64_fp16` | 已有 | 复用 |

身份合成：上游无 `model_id/weights_id`（按架构分派）。Reader 的 v3 投影层
为该工件合成 `model_id="qwen3.8-flash-next"`、`weights_id="nvfp4"`，注册进
`src/targets/registry.cpp` 分派表与容器文档的身份注册表。

## 5. 运行时架构（决定）

### 5.1 第二个家族运行时

按 AGENTS.md 契约，Qwen4Exp 不是 Qwen3.6 家族成员（4 流 HC 残差、QSA
mixer、PLE、块 FP8 MTP 库都与 3.6 家族的固定结构冲突）。因此：

- 新家族 `src/targets/qwen4_exp`：identity-free 的 Qwen4Exp 家族运行时，
  拥有 `SequencePlan<Variant>` / `RequestPlan<Variant>` / `Program<Variant>`
  算法、Text/MTP 调度、状态事务、workspace、graph frontier 约定。结构上
  模仿 `qwen3_6` 家族的模板形态（PIMPL + `instantiate.h`），但独立成树，
  与 3.6 家族零共享可变状态。
- 新包 `src/targets/qwen3_8_flash_next`：身份、binder/分片图、`LoadedModel`、
  三个执行叶家族（见 §5.2）、`detail::Variant`、graph profiles。
- registry：`ActiveTarget` variant 增加第三个备选，`construct_target` 增加
  `qwen3.8-flash-next` 分支；`construct_registered<Target,Loaded,Instance>`
  模板原样复用。
- 前端：Flash-Next 用 Qwen 前端（tokenizer/模板/MRoPE，六资源内嵌于工件）。
  `qwen4_exp` 家族 alias 复用 `qwen3_6` 的前端类型（同属 Qwen 前端语义，
  资源随工件走）；不复制代码。

### 5.2 三个执行叶家族（包侧）

1. **attention 投影（QSA + HC）**：输入 HC 混合流 → `q8_g32` 投影 →
   `qsa_prepare`（逐头 RMSNorm+RoPE[11,11,10]、θ=1e7；主 K/V 入页式缓存；
   index key 4-token 块聚合）→ `qsa_indexer`（top ≤512 块打分，BF16 索引页）
   → `qsa_sparse_gqa`（选中块 + 可见尾块；(24,2) 头制；sigmoid 门控在
   attention 之后）→ 输出投影。MTP 层复用同一家族的 MTP 变体。
2. **GDN 投影/控制**：复用既有算子家族
   （`gated_delta_net`、`causal_conv1d_silu_split`、FP32 状态池），控制投影
   改为 `gdn_control_proj`（对 HC 混合流，而非 3.6 家族的残差折叠式）。
3. **post-mixer（MoE）**：router BF16 [E+1,H] → top-10 softmax 归一 →
   路由专家 NVFP4 batched 库（gate_up (512,1280,2560)、down (512,2560,640)）
   + 共享专家 BF16/既有 SwiGLU；输出按路由系数合并。

### 5.3 新 Op 清单（`src/ops` + `include/ninfer/ops` 契约）

| Op | 说明 |
|---|---|
| `hc_*`（mix/norm/read/up） | 4 流残差的门控混合、范数、读写投影；贯穿主模型与 MTP |
| `qsa_prepare` / `qsa_indexer` / `qsa_sparse_gqa` / `qsa_tail_extract` | QSA 四件套；sm_70 解码/验证窗口内核参考自有 GQA 解码 kernel + mma.m8n8k4 路线 |
| `moe_routed_nvfp4` | 分组 NVFP4 路由专家（解码/验证窗口、TP K 切分形式）；路由 top-10 归一 |
| `moe_block_fp8`（MTP 库） | 块 128×128 FP8 专家，仅 MTP 层 |
| `ple_lookup` / `ple_gate` / `ple_conv` | NF4 行查表（host hash → 行缓存/盘读 → pinned 上传）、门控、第 1 层膨胀卷积 |
| 复用 | `gated_delta_net` 家族、Q4 草头（现有 `q4_g64` head 机制）、KV int8 g64 编解码、采样、allreduce/mailbox |

### 5.4 专家缓存与 offload（核心性能机制）

- TP2 下每卡预算（估）：稠密 ~3GB + 专家缓存 ~27GiB + KV/工作区/graphs
  ~2-3GB ≈ 32GB 上限内；专家总 68GiB，显存驻留 ~54GiB（约 80%），
  host RAM 支撑 ~14GiB。
- TP 形态：路由专家按 expert-TP（intermediate K 切分，TP2 每专家 640→320，
  落在 group-16 边界上）；router/共享专家/嵌入复制，头部按词表切；
  down 输出汇合点做一次 allreduce（复用 `linear_add_row_parallel` 形态）。
  EP（整专家）形式本期不做。
- 逐层静态热度优先级：移植上游 `kFlashNextExpertPriority`（48×512 u32 表，
  决定缓存副本集合，不改变路由），启动时按优先级填充设备窗口。
- miss 流水线：MTP 窗口路由在轮首已知 → 下一轮 miss 集预取（双窗口交替，
  pinned 槽有界，事件同步复用上游 upload 语义）。每轮 miss 流量估
  ~160-800MB（13.1GB/s 实测 PCIe gen3），由轮间计算遮蔽。
- PLE：表留盘（NVMe），行缓存 + 按 hash 批量取行；每 token 仅 16 行 ×160
  NF4 值 ≈ 2.5KB/行，第 1 层专用，延迟可接受。

## 6. 数值正确性与验证

- **独立 oracle**：`tools/reference/qwen3_8_flash_next/`（torch 朴素 FP32
  参考模型，覆盖 Text/MTP 全路径；NVFP4/NF4/块FP8 按存储码本 + 精确缩放
  解码），仿 `tools/reference/qwen3_6_27b` 模式。
- **Op 级**：QSA 四件套（vs FP64 参考：prepare RoPE 精确、indexer 选择一致、
  sparse GQA 容差）、MoE NVFP4 分组 GEMM（容差准则 + KL 对 FP16 路径）、
  HC/PLE（PLE lookup 要求对 BF16 lookup 精确舍入）、GDN 复用既有测试。
- **TP2 parity**：沿用 `tools/tp2/parity` 模式（逐位 argmax 一致性 + logit
  余弦）。
- **端到端**：真实工件上 greedy 抽查 + 教师强制 logit parity 对参考模型；
  MTP accept/commit 状态与 27B 同套语义测试。
- **容器**：v3 多分片 reader 契约测试（分片边界、artifact_id 校验、越界）。

## 7. 性能验证

- 公共 bench：262144 容量、INT8 KV、MTP-3、eager decode；
  报 500 token / ~59K 占用两档的 committed tok/s + 接受率 + prefill tok/s。
- 同机 Strata 对比：Strata 已装（`~/Strata-data`），用其 V100 CUDA 12
  engine 跑。**基线分两步**：先以本机已装的 IQ2_XS 包实测同 corpus；
  UD-IQ4_XS（Strata V100 最优档位，约比 IQ2_XS 慢 ~30%，需另行下载）
  留到后续精确比较阶段再补测。结论以"本仓 NVFP4 vs Strata 同卡实测
  IQ2_XS /（后续）UD-IQ4_XS"表述，记录双方版本与设置。
- 归因顺序：端到端 → 阶段（轮内 transfer/compute 遮蔽）→ 必要时 kernel
  级（QSA、MoE）。不做超出决策需要的profiling。

## 8. 里程碑

1. **M1 容器**：v3 多分片 Reader + 新格式 + 绑定契约测试；用真实工件
   `--validate` 通过，binder 全量绑定（含 TP2 分片图）。
2. **M2 文本 TP1**：`qwen4_exp` 家族 + 包骨架，单卡 greedy 文本出 token，
   对参考模型 parity 通过；PLE/HC/QSA/GDN/MoE 全链路。
3. **M3 MTP**：MTP 层 + Q4 草头 + accept/commit；验收 1..5 窗口。
4. **M4 专家缓存 + TP2**：设备窗口 + host 支撑 + 优先级 + 预取流水线；
   TP2 parity。
5. **M5 发布面**：registry/CLI/serving/bench 接线、README/文档更新、
   性能 campaign（§7）与 Strata 对比结论。

## 9. 风险

| 风险 | 应对 |
|---|---|
| QSA sm_70 kernel 性能不达线 | 参考 Strata 已验证的 mma.m8n8k4 与少 shuffle 路线；M2 末做 kernel 归因，必要时只优化 indexer/sparse_gqa 两个热点 |
| 每 rank 显存超限 | 启动期容量决议已存在；缓存窗口大小进入 `SequenceCapacityCurve`/preflight 链路，超出即降缓存 + host 支撑 |
| `q8_g32_fp16` 与 `W8G32_F16S` 语义差异超预期 | M1 首项对表；不同构即新增格式，不影响后续里程碑 |
| PLE 盘读抖动 | 行缓存 + 轮首批量预取；PLE 仅第 1 层，最坏退化为固定 +2-4ms/轮 |
| 94GB RAM 被其他进程挤压 | 文档明示 Flash-Next 需要 ~65-70GB 空闲 RAM；启动时校验并给出诊断 |
| 多分片读放大 | 按对象顺序读；不实现随机跨卷寻址之外的东西 |

## 10. 已定 vs 留给建造者

**已定（用户/本设计）**：NVFP4 官方工件直读；不写转换管线；第二家族 +
新包；TP2 为验收形态；MTP-3 验收；INT8 KV；本期无 Vision、无 CPU 专家
补集、无 CUDA Graph MoE 解码；成功线 >60 tok/s 且胜过 Strata 同卡数字。

**留给建造者**：Reader v3 的代码组织（沿用窄适配函数 vs 新增通用 v3 读取
器内部结构）；`q8_g32` 复用或新增格式的最终判定；HC/QSA/PLE/MoE 各 Op 的
kernel 分解与 tile 选择；家族运行时文件切分；workspace/graph frontier 的
具体字节布局；`expert_priority` 表的存放位置与加载方式。
