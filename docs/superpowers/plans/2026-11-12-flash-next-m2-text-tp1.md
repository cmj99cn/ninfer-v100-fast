# Qwen3.8-Flash-Next 文本路径（M2）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在本仓落地 Qwen4Exp 家族运行时与 `qwen3.8-flash-next` 包，实现 Flash-Next 的**文本**前向（四路 HC 残差、GDN、QSA、MoE、PLE），并用独立 oracle 证明数值正确。

**架构：** 新建 identity-free 家族 `src/targets/qwen4_exp`（拥有 SequencePlan/RequestPlan/Program 算法、Text 调度、状态事务、workspace）与新包 `src/targets/qwen3_8_flash_next`（身份、绑定、LoadedModel、三个执行叶家族、graph frontier）。前端（tokenizer/模板/MRoPE/输出语义）直接 alias 复用 `qwen3_6` 家族，不复制实现。缺位的算子（HC 四路、QSA 四件套、NVFP4 路由专家、PLE）在 `src/ops` 新写语义闭合的实现，各配一个独立 FP32/FP64 oracle。

**技术栈：** C++20、CUDA 12.8、`sm_70`（Volta）、NCCL（TP2 留 M4）、CUTLASS 3.x（现有 FetchContent）、仓库自有的无框架 CTest 风格、PyTorch 参考模型。

**规格：** `docs/superpowers/specs/2026-11-11-qwen4exp-flash-next-nvfp4-design.md`（§2 非目标、§3 客观事实、§5 运行时架构、§6 数值正确性、§8 里程碑）。上游语义权威见 `docs/maintainer/qwen3.8-flash-next-artifact.md`（本计划任务 1 建立）。

## 全局约束

- 只用上游官方 4 文件 v3 工件 `qwen3_8_flash_next.ninfer`（+ `.part-0001..0003`，108,445,200,388 B），不写转换流水线、不改工件字节。
- 目标形态：TP1 单卡（V100-SXM2 32GB）；TP2 在 M4。**本计划不做**：MTP、TP2、专家缓存/host 支撑、Vision（���6）、性能 campaign、CUDA Graph 捕获。
- 家族与包必须遵守 `AGENTS.md` 的归属边界：`src/artifact` 只做容器与绑定；dtype 在 `src/core`；解码语义归目标包；算子实现归 `src/ops`；家族不持有任何目标身份字符串。
- 每条数值主张必须有独立 oracle（朴素 FP32/FP64，按存储码本 + 精确缩放解码），不得用另一条实现或参考模型充当第二 oracle。
- 新代码不写注释；C++20；提交用 Conventional Commit 小写类型；每个任务结束一次提交。
- 测试沿用仓库风格：自建 main + 返回码，CUDA/工件缺失时返回 77（CTest `SKIP_RETURN_CODE 77`）。
- 禁止低比特量化（Q2/IQ3 档位）作为任何产品路径；本里程碑只走 NVFP4/Q8/NF4/BF16/FP16。

## 审查重点

1. **PLE 行地址与 hash**：PLE 表行号由 raw token 的二/三元组 hash 决定（16 个头，vocab/offset 来自工件 `resource/text/ple_hash_controls`）。任何 hash 顺序、乘数、offset 或行映射的偏差都会让第 1 层输出静默错误 —— 必须有逐 token 的 oracle 比对。
2. **QSA 块选择**：indexer 对 4-token 压缩块打分后选 top-512 块（`indexer_budget 2048 / compress 4`），所有 query 头共享同一块集合；选择的块集合与 tail 可见块的拼接规则必须与参考实现逐元素一致，否则长上下文会静默丢块。
3. **NVFP4 专家库几何**：512 个专家的 gate_up `[512,1280,2560]` / down `[512,2560,640]` 是 batched blockscale 布局（码面 + 缩放面 + 每矩阵 FP32 除数），按专家切分时缩放 tile 与除数必须跟着专家走，否则单个专家的数值会错而总量看似正常。
4. **HC 四路流的注入系数**：`hc_mix` 产出 FP32 injection ∈ [0,2]（`2*sigmoid(b/4)`），直接乘到残差上；把它错写成 sigmoid 或漏掉 /4 会让所有层输出整体偏移，且不会触发任何形状断言。
5. **GDN 的 Z 与卷积位置**：Z 绕过因果卷积、conv 的 SiLU 与输出 gate 的 sigmoid 分属不同位置；把 SiLU 施加到 Z 上会让门控在 [0,1] 之外，数值仍在合法区间但质量明显下降。

## 文件结构

**新建 `src/targets/qwen4_exp`（家族，identity-free）**

| 文件 | 职责 |
|---|---|
| `export/ninfer/targets/qwen4_exp/runtime.h` | `SequencePlan`/`SequencePlanner`/`RequestBasePlan`/`RequestPlan`/`Program` 模板出口 + `make_sequence_planner`/`create_program` |
| `export/ninfer/targets/qwen4_exp/frontend.h` | alias 复用 `qwen3_6` 前端类型（`Frontend`/`PreparedPrompt`/`OutputSession`） |
| `export/ninfer/targets/qwen4_exp/model_view.h` | 家族侧语义权重视图 schema（`ModelView<Variant>`） |
| `impl/runtime/instance.h` | `Variant` 别名与家族常量（attention/gdn 缩放、prefill 对齐、上下文上限） |
| `impl/runtime/instantiate.h` | 共享源码 include 清单（包 TU 定义宏后 include 一次） |
| `impl/runtime/api_impl.h` | 五个模板类的显式特化出口 |
| `impl/runtime/program.h` / `program_impl.h` | Program 状态机：准入、prefill、decode 轮、发布 |
| `impl/runtime/schedule.h` | Text 调度接口（prefill chunk / decode batch / 采样发布） |
| `impl/runtime/text_context.h` / `text_context_impl.h` | 层循环主体（HC 主干 → 层 → 主头） |
| `impl/runtime/workspace_recipe.h` | workspace 组成与容量查询 |

**新建 `src/targets/qwen3_8_flash_next`（包，身份与叶子）**

| 文件 | 职责 |
|---|---|
| `export/ninfer/targets/qwen3_8_flash_next/package.h` | `Package` 身份、`WeightsProfile`、`LoadPlan`/`LoadedModel` PIMPL |
| `impl/config.h` | Flash-Next 文本常量（48 层/36 GDN/12 QSA、hidden 2560、HC 4×320、残差 10240、专家 512/top-10/640、QSA 24×2×256、indexer 4×1×128/2048、词表 248320） |
| `impl/variant.h` / `impl/variant.cpp` | `detail::Variant`：三个执行叶家族 + 常量 + graph profile |
| `impl/load/bindings.h` / `bindings.cpp` | 工件绑定计划（参数名 → 对象 → 角色 → 分片）、`LoadedModelData` |
| `impl/package.cpp` | `Package::` 静态成员实现 |

**新建 `include/ninfer/ops/` 契约 + `src/ops/` 实现**：`linear_split.h`、`gated_residual.h`（HC 六入口）、`qsa.h`（五入口）、`ple.h`（hash/lookup/gate/conv）、`moe_nvfp4.h`（topk 路由 + grouped NVFP4）。

**新建测试**：`tests/targets/qwen3_8_flash_next/test_{bindings,load_plan,text_forward}.cpp`、`tests/ops/{gated_residual,qsa,ple,moe_nvfp4,linear_split}_test.cpp`、`tools/reference/qwen3_8_flash_next/`（参考模型）、`tests/targets/qwen3_8_flash_next/test_reference_parity.py`。

---

### 任务 1：v3 目录元数据与工件绑定清单（容器能力）

**文件：**
- 修改：`src/artifact/reader.h` / `src/artifact/reader.cpp`（暴露 v3 目录的 `components`/`bindings`/`uses`）
- 测试：`tests/test_v3_artifact_reader.cpp`

- [ ] **步骤 1：编写失败的测试**

在 `tests/test_v3_artifact_reader.cpp` 增加 `test_v3_directory_metadata`：用现有 fixture writer 写一个含 `components.text.config`（`hidden_size`、`num_hidden_layers`）、`bindings`（`{"text/layers/0/attention_hc/down": {"object": "weight/000001"}}`）、`uses`（`{"parameter": "text/layers/0/attention_hc/down", "input": "text/layers/0/attention_hc/normalized_streams", "activation_policy": "A16Only"}`）的 v3 目录，断言 reader 能返回：
- `reader.v3_directory().components` 按组件名索引，text 组件含 config 键值对（`hidden_size == 2560`）；
- `reader.v3_directory().bindings` 把参数名映射到对象 id；
- `reader.v3_directory().uses` 给出参数 → 输入名与 activation policy。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：编译失败（`Reader::v3_directory` 不存在）。

- [ ] **步骤 3：实现目录元数据读取**

在 `src/artifact/reader.h` 增加（不放 `nlohmann::json` 到公开头，避免把 JSON 依赖泄到公共接口）：

```cpp
struct V3Use { std::string parameter; std::string input; std::string activation_policy; };
struct V3Directory {
    // component name -> config key -> value (numbers as double, strings as string)
    std::map<std::string, std::map<std::string, std::string>> component_config;
    std::map<std::string, std::string> bindings;   // parameter name -> object id
    std::vector<V3Use> uses;
};
```

`Reader::v3_directory()` 返回 `const V3Directory&`；v2 工件返回空目录（`component_config`/`bindings` 为空、`uses` 为空），且单文件 v3 窄适配路径同样返回其目录内容。实现放在 `src/artifact/reader.cpp` 的 v3 解析处：解析 `components[*].config` 的标量为字符串（数值用最短往返格式），`bindings` 的 `object` 字段，`uses` 的三个字段。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：真实工件核对（手工，一次性证据）**

运行：
```bash
python3 - <<'EOF'
import json,struct
p='/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer'
with open(p,'rb') as f:
    h=f.read(32); jl=struct.unpack_from('<Q',h,8)[0]; d=json.loads(f.read(jl))
print(len(d['bindings']), len(d['uses']), sorted(d['components']))
EOF
```
预期：`1786 1205 ['mtp','text','vision']`。把这三个数字写进任务 2 的绑定完整性断言。

- [ ] **步骤 6：Commit**

```bash
git add src/artifact/reader.h src/artifact/reader.cpp tests/test_v3_artifact_reader.cpp
git commit -m "feat(artifact): expose the v3 directory config, bindings and uses"
```

---

### 任务 2：Flash-Next 权重角色映射与维护者文档

**文件：**
- 创建：`src/targets/qwen3_8_flash_next/impl/load/roles.h`（角色枚举与参数名常量）
- 创建：`src/targets/qwen3_8_flash_next/impl/load/roles.cpp`（角色 → 参数名的映射表）
- 创建：`docs/maintainer/qwen3.8-flash-next-artifact.md`（语义与几何权威）
- 测试：`tests/targets/qwen3_8_flash_next/test_roles.cpp`

- [ ] **步骤 1：确定角色命名空间（读工件）**

运行（一次性取证，输出进文档）：
```bash
python3 - <<'EOF'
import json,struct,collections,re
p='/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer'
with open(p,'rb') as f:
    h=f.read(32); jl=struct.unpack_from('<Q',h,8)[0]; d=json.loads(f.read(jl))
names=sorted(d['bindings'])
print(len(names))
pat=collections.Counter(re.sub(r'\d+','N',n) for n in names)
for k,v in sorted(pat.items()): print(v,k)
EOF
```
预期：打印按数字归一化后的参数名模式与计数（形如 `text/layers/N/attention_hc/down`、`text/layers/N/attention/query` 等）。把这张表作为角色映射表的**唯一事实来源**写进 `roles.h` 的常量区。

- [ ] **步骤 2：编写失败的测试**

`tests/targets/qwen3_8_flash_next/test_roles.cpp`：断言
- 角色枚举覆盖第 1 步模式表的全部模式（模式集合相等，缺一个即失败）；
- `roles::parameter_name(role, layer)` 对 layer 0/3/47 产生该模式下的正确名字（例如 `text/layers/3/attention/...`）；
- `roles::layer_of("text/layers/17/...") == 17`，非层参数返回 `std::nullopt`。

- [ ] **步骤 3：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_flash_next_roles_test && ./build/tests/ninfer_flash_next_roles_test`
预期：编译失败（`roles.h` 不存在）。

- [ ] **步骤 4：实现角色映射**

`roles.h` 定义 `enum class Role { Embedding, OutputHead, FinalHcNorm, FinalHcDown, FinalHcUp, LayerHcNorm, LayerHcDown, LayerHcUp, AttentionHcNorm, AttentionHcDown, AttentionHcUp, MoeHcNorm, MoeHcDown, MoeHcUp, GdnQuery, GdnKey, GdnValue, GdnZ, GdnA, GdnB, GdnALog, GdnDtBias, GdnConvolution, GdnNorm, GdnOutput, QsaQuery, QsaGate, QsaKey, QsaValue, QsaIndexQuery, QsaIndexKey, QsaQueryNorm, QsaIndexQueryNorm, QsaKeyNorm, QsaIndexKeyNorm, QsaOutput, MoeRouter, MoeSharedScore, MoeSharedGateUp, MoeSharedDown, MoeRoutedGateUp, MoeRoutedDown, PleKey, PleValue, PleNormKey, PleNormQuery, PleNormConv, PleConvolution, ... }`（以步骤 1 的模式表为准，不多不少），以及 `parameter_name(Role, std::uint32_t layer)`、`layer_of(std::string_view)`。

- [ ] **步骤 5：写维护者文档**

`docs/maintainer/qwen3.8-flash-next-artifact.md` 必须包含（上游权威引用写仓库路径 + 行号）：
- 目标清单与格式普查（bf16 916、nf4 128、q8_g32 96、nvfp4 96、fp32 73、block128 2、q4 1、resource 7；工件目录 fp32=73 与 `.conversion.json` 的 72 相差 1，以目录为准）；
- 文本 config 全量（hidden 2560、48 层、layer_types 36 GDN + 12 QSA（`l%4==3`）、HC streams 4 / lowrank 320、residual 10240、QSA 24 query heads / 2 kv heads / head_dim 256、indexer 4 heads / 1 kv head / dim 128 / compress 4 / budget 2048 → block_budget 512、GDN 16 key heads × 48 value heads × dim 128、conv kernel 4、MoE 512 专家 / top-10 / intermediate 640 / shared 640 / norm_topk_prob、PLE embed 2560 / ngram 3 / heads_per_ngram 8 / split_ngram_parts 128 / conv kernel 4 / dilation 3 / layer_ids [1] / shard [2500012,160]、vocab 248320、rope mrope_section [11,11,10] / partial_rotary_factor 0.25 → rotary_dim 64 / theta 1e7 / interleaved、rms_norm_eps 1e-6、mamba_ssm_dtype float32）；
- 派生维度公式：`query_width = query_heads*head_dim = 6144`、`kv_width = kv_heads*head_dim = 512`、`index_query_width = 4*128 = 512`、`index_key_width = 1*128 = 128`、`residual_width = hidden*streams = 10240`；
- 四路 HC 数学（`hc_expand_norm` / `hc_grouped_rmsnorm` / `hc_combine` / `hc_combine_norm` / `hc_mix` / `hc_final_mix`，逐条抄上游 `include/ninfer/ops/gated_residual.h` 的公式）；
- 文本主干顺序（照上游 `src/models/qwen4_exp/execution/text.cpp`）：stem 的 `hc_expand_norm` → 每层两次 `hc_mix`（attention HC、MoE HC）与两次 combine（`hc_combine_norm` 用下一层的 `attention_hc.norm`；`i + 1 == ple.layer` 时改用不带归一化的 `hc_combine` 并随即执行 PLE）→ finish 的 `hc_final_mix` → 主头；
- QSA 数学（`qsa_prepare` / `qsa_prepare_kv` / `qsa_indexer` / `qsa_sparse_gqa` / `qsa_tail_extract`，逐条抄上游 `include/ninfer/ops/qsa.h` 的公式与索引约定）；
- PLE 数学（hash 公式、`ple_gate`、`ple_conv_add_norm`、行查表契约，逐条抄上游 `include/ninfer/ops/ple.h` 与 `src/models/qwen4_exp/ple/hash.cpp`）；
- MoE 路由（top-10 + `norm_topk_prob=true`）与 batched NVFP4 专家几何；
- 解码边界：哪些格式本仓已能解码（bf16/fp32/q4/q8 行拆），哪些必须等目标包解码（NVFP4 batched、NF4、块 FP8）。

- [ ] **步骤 6：运行测试验证通过**

运行：同步骤 3。预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add src/targets/qwen3_8_flash_next/impl/load/roles.h src/targets/qwen3_8_flash_next/impl/load/roles.cpp docs/maintainer/qwen3.8-flash-next-artifact.md tests/targets/qwen3_8_flash_next/test_roles.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): map artifact parameter names to weight roles"
```

---

### 任务 3：包骨架、registry 接线与装载计划（不执行前向）

**文件：**
- 创建：`src/targets/qwen3_8_flash_next/export/ninfer/targets/qwen3_8_flash_next/package.h`
- 创建：`src/targets/qwen3_8_flash_next/impl/config.h`、`impl/package.cpp`
- 修改：`src/targets/registry.h` / `src/targets/registry.cpp`（第三个 `ActiveTarget` 备选 + `construct_target` 分支）
- 修改：`src/CMakeLists.txt`（`add_subdirectory(targets/qwen3_8_flash_next)`）
- 测试：`tests/targets/qwen3_8_flash_next/test_load_plan.cpp`

- [ ] **步骤 1：编写失败的测试**

`test_load_plan.cpp`：
- `Package::model_id == "qwen3.8-flash-next"`、`Package::weights_id == "nvfp4"`、`Package::resolve_weights("qwen3.8-flash-next", "nvfp4")` 命中本包；`resolve_weights("qwen3.8-27b", "nvfp4")` 不命中本包。
- 用任务 1 的 fixture 风格写一个合成 v3 工件（`components.text.config` 含上表常量、`bindings` 覆盖 48 层的必要角色），断言 `bind_artifact` 产出的 `BindingPlan`：
  - 每个被绑对象在工件 `bindings` 中有对应参数名；
  - 绑定覆盖 `text/embedding`、`text/output_head`、`text/final_hc/*`、48 层各自的 HC/GDN 或 QSA/MoE 角色；
  - 未被任何角色认领的 `bindings` 条目为 0（真实工件上 1786 条绑定必须被 100% 认领，允许一个对象承载多个角色）。
- 断言 TP1 下每层的 routed 专家库是整对象复制，TP2 计划在 M4（本任务只断言 tp=1）。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_flash_next_load_plan_test && ./build/tests/ninfer_flash_next_load_plan_test`
预期：编译失败（包尚不存在）。

- [ ] **步骤 3：实现包骨架与 registry 接线**

- `impl/config.h`：`namespace ninfer::targets::qwen3_8_flash_next::detail { struct TextConfig { ... } }`，常量全部取自任务 2 文档（hidden 2560、layers 48、qsa_layers = `l%4==3`、hc_streams 4、hc_lowrank 320、residual_width 10240、query_heads 24、kv_heads 2、head_dim 256、index_heads 4、index_kv_heads 1、index_head_dim 128、index_compress 4、index_budget 2048 → block_budget 512、gdn_key_heads 16、gdn_value_heads 48、gdn_head_dim 128、conv_kernel 4、experts 512、experts_per_tok 10、expert_intermediate 640、shared_intermediate 640、vocab 248320、rope_theta 1e7、rope_sections {11,11,10}、partial_rotary 0.25 → rotary_dim 64、norm_eps 1e-6、ple_layer_ids {1}、ple_ngram 3、ple_heads_per_ngram 8、ple_split_parts 128、ple_conv_kernel 4、ple_conv_dilation 3）。
- `export/.../package.h`：`struct Package { static constexpr std::string_view model_id = "qwen3.8-flash-next"; static constexpr std::string_view weights_id = "nvfp4"; static constexpr std::string_view target_key = "qwen3_8_flash_next"; static WeightsProfile resolve_weights(std::string_view, std::string_view); static ModelSamplingDefaults sampling_defaults(); static detail::LoadPlan plan_load(artifact::Binder&, WeightsProfile, int tp); ... };`
- `impl/load/bindings.{h,cpp}`：`bind_artifact(Binder&, WeightsProfile, int tp)` 按角色遍历 48 层，构造 `BindingPlan`（含每层 HC/GDN 或 QSA/MoE 的对象句柄与形状），并把工件 `bindings` 中未被认领的参数名收集为诊断。
- `src/targets/registry.h`：新增 `LoadedQwen3_8_FlashNext` / `Qwen3_8_FlashNextInstance`，`ActiveTarget` 增加 `std::unique_ptr<Qwen3_8_FlashNextInstance>` 备选；`src/targets/registry.cpp` 的 `construct_target` 增加 `identity.model_id == Qwen3_8_FlashNext::model_id` 分支，复用现有 `construct_registered<Target, Loaded, Instance>` 模板。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen3_8_flash_next src/targets/registry.h src/targets/registry.cpp src/CMakeLists.txt tests/CMakeLists.txt tests/targets/qwen3_8_flash_next/test_load_plan.cpp
git commit -m "feat(flash-next): add the package skeleton and registry wiring"
```

---

### 任务 4：HC 四路残差算子（六个入口 + oracle）

**文件：**
- 创建：`include/ninfer/ops/gated_residual.h`
- 创建：`src/ops/gated_residual/{hc_mix.cu,hc_norm.cu,hc_combine.cu}`
- 修改：`src/CMakeLists.txt`（注册实现）
- 测试：`tests/ops/gated_residual_test.cpp`

- [ ] **步骤 1：编写失败的测试**

`tests/ops/gated_residual_test.cpp`（照 `tests/ops/` 既有 op 测试风格，FP32/FP64 oracle）：
- `hc_expand_norm`：embedding `[H,M]` BF16、norm_weight `[H,S]` BF16 → 断言 `normalized[d,s,m] == embedding[d,m]*(1+w[d,s])/sqrt(sum_d embedding[d,m]^2/H + eps)`（FP64 计算后与 BF16 输出比，容差按 BF16 舍入）；
- `hc_grouped_rmsnorm`：逐流独立统计；
- `hc_combine`：`residual + injection[s,m]*block_output[d,m]`，injection 为 FP32 ∈[0,2]；
- `hc_combine_norm`：先用实际 BF16 更新字再归一化；
- `hc_mix`：`[a,b] = down_injection * flatten(normalized)`（a:320、b:4，down_injection `[324,10240]`，行 [320,324) 是 injection）、`u = SiLU(a/4)`、`g = up*u`（up `[10240,320]`）、`block_input[d,m] = sum_s sigmoid(g[s*H+d,m]) * normalized[d,s,m] / 4`、`injection[s,m] = 2*sigmoid(b[s,m]/4)`；
- `hc_final_mix`：down-only `[320,10240]` 变体，产出 sample_hidden。

测试几何用 H=2560、S=4、lowrank=320 的真实形状，M 取 1 与 7 两种；oracle 用 FP64 手写公式。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_gated_residual_test && ./build/tests/ninfer_gated_residual_test`
预期：编译失败（契约头不存在）。

- [ ] **步骤 3：实现契约头与实现**

`include/ninfer/ops/gated_residual.h` 按仓库 Op 契约格式（`Op:` / `Math / indexing:` / `Logical shapes:` / `Supported domain:` / `Numeric:` / `Effects:` / `Workspace:` / `Execution:`）写六个入口的精确数学（照任务 2 文档第 4 节，逐字一致），签名：

```cpp
void hc_expand_norm(const Tensor& embedding, const Tensor& norm_weight, float epsilon,
                    Tensor& residual, Tensor& normalized, cudaStream_t stream);
void hc_grouped_rmsnorm(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                        Tensor& normalized, cudaStream_t stream);
void hc_combine(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                Tensor& updated_residual, cudaStream_t stream);
void hc_combine_norm(const Tensor& residual, const Tensor& block_output, const Tensor& injection,
                     const Tensor& next_norm_weight, float epsilon, Tensor& updated_residual,
                     Tensor& normalized, cudaStream_t stream);
void hc_mix(const Tensor& normalized, const HcMixWeights& weights, WorkspaceArena& workspace,
            Tensor& block_input, Tensor& injection, cudaStream_t stream);
void hc_final_mix(const Tensor& normalized, const HcFinalMixWeights& weights,
                  WorkspaceArena& workspace, Tensor& sample_hidden, cudaStream_t stream);
```

`HcMixWeights{ down_injection (BF16 [324,10240], 行 [320,324) 为 injection), up (BF16 [10240,320]) }`、`HcFinalMixWeights{ down (BF16 [320,10240]), up (BF16 [10240,320]) }`（BF16 行连续，行序 = `s*2560+d`）。

实现：`hc_norm` 用 FP32 统计 + BF16 输出（`1+w` 不先舍入到 BF16）；`hc_combine` 输出 BF16；`hc_mix` 的 down_injection 与 up 用 `ops::linear` 的 BF16 路径（必要时新增 `linear_split` 在任务 6 一并提供）。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add include/ninfer/ops/gated_residual.h src/ops/gated_residual src/CMakeLists.txt tests/ops/gated_residual_test.cpp tests/CMakeLists.txt
git commit -m "feat(ops): add the four-stream gated residual ops"
```

---

### 任务 5：QSA 四件套 + `linear_split`（五入口 + oracle）

**文件：**
- 创建：`include/ninfer/ops/qsa.h`、`include/ninfer/ops/linear_split.h`
- 创建：`src/ops/qsa/{qsa_prepare.cu,qsa_indexer.cu,qsa_sparse_gqa.cu,qsa_tail_extract.cu}`、`src/ops/linear/split/linear_split.cu`
- 测试：`tests/ops/qsa_test.cpp`、`tests/ops/linear_split_test.cpp`

- [ ] **步骤 1：编写失败的测试**

- `linear_split`：一个 packed 父权重（row-split K128 布局）按角色行区间切出 N 个逻辑投影，断言每个输出的数值等于独立 `ops::linear`（同一权重的等价行区间）结果；
- `qsa_prepare`：断言 query 归一化 + RoPE（partial rotary 64 维，section [11,11,10] interleaved）、key 归一化 + RoPE、index_query 归一化 + RoPE、index_key 4-token 块聚合 `u = BF16(mean(raw_index_key[4g:4g+4]))`、`index.K[g] = R(N_128(u, index_key_norm), rho)`，以及原始 index key 与坐标记录被写入；
- `qsa_indexer`：`score[g] = sum_{h=0..3} max(0, sum_{d=0..127} index_query[d,h,t,b]*index.K[d,g])`，取 top-512 块；
- `qsa_sparse_gqa`：选中块 + 可见 tail 的注意力 `out[d,h] = BF16(attention[d,h]*sigmoid(FP32(output_gate[d,h])))`，indexer 分数不进入 logits；
- `qsa_tail_extract`：tail 状态按记录的 raw key / rope 位置推进。

几何用真实形状：query 24 heads × 256、kv 2 heads × 256、index 4 heads × 128、index kv 1 × 128、compress 4、block_budget 512。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_qsa_test ninfer_linear_split_test && ./build/tests/ninfer_qsa_test && ./build/tests/ninfer_linear_split_test`
预期：编译失败。

- [ ] **步骤 3：实现契约与实现**

`include/ninfer/ops/qsa.h` 的数学逐条抄上游 `include/ninfer/ops/qsa.h`（prepare/indexer/sparse_gqa/tail_extract 四节），签名：

```cpp
void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const QsaTailStateView& before, QsaCacheView cache,
                 WorkspaceArena& workspace, QsaRawRecordWriteView& records,
                 Tensor& query, Tensor& index_query, cudaStream_t stream);
void qsa_indexer(const Tensor& index_query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, std::uint32_t max_visible_tokens,
                 WorkspaceArena& workspace, QsaBlockSelection& selected, cudaStream_t stream);
void qsa_sparse_gqa(const Tensor& query, const Tensor& output_gate, const QsaCacheView& cache,
                    const QsaBatchView& batch, const QsaBlockSelection& selected,
                    std::uint32_t max_visible_tokens, WorkspaceArena& workspace, Tensor& out,
                    cudaStream_t stream);
void qsa_tail_extract(const ConstQsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& base_positions, const Tensor& valid_columns,
                      QsaTailStateView& after, cudaStream_t stream);
```

`QsaBlockSelection{ Tensor selection; Tensor counts; }`（selection `[512,T,B]`、counts `[T,B]`）。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add include/ninfer/ops/qsa.h include/ninfer/ops/linear_split.h src/ops/qsa src/ops/linear/split tests/ops/qsa_test.cpp tests/ops/linear_split_test.cpp src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(ops): add the QSA indexed attention ops and packed linear split"
```

---

### 任务 6：NVFP4 路由专家（top-10 归一化 + grouped GEMM + oracle）

**文件：**
- 创建：`include/ninfer/ops/moe_nvfp4.h`
- 创建：`src/ops/moe_nvfp4/{route.cu,grouped.cu,shared.cu}`
- 测试：`tests/ops/moe_nvfp4_test.cpp`

- [ ] **步骤 1：编写失败的测试**

- 路由：router 权重 BF16 `[513,2560]`（512 专家 + 1 shared 行），对每个 token 取 top-10，`norm_topk_prob=true` → 概率 = softmax(top-10 logits) 归一化，共享专家行给 sigmoid 分数；断言 top-10 的选择与 tie 规则（并列时专家 id 小者先）、归一化后和为 1；
- grouped gate_up/down：真实几何 512 专家、hidden 2560、intermediate 640、gate_up `[512,1280,2560]`、down `[512,2560,640]`（batched blockscale 布局），取 4 个专家做 FP32 oracle 比对：oracle 按存储码本解 NVFP4（4-bit code + 16 元素组 E4M3 scale + 每矩阵 FP32 除数）后做 FP32 GEMM；
- 共享专家：BF16 SwiGLU 路径贡献一次。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_moe_nvfp4_test && ./build/tests/ninfer_moe_nvfp4_test`
预期：编译失败。

- [ ] **步骤 3：实现契约与实现**

`include/ninfer/ops/moe_nvfp4.h`：

```cpp
struct MoeNvfp4Weights {
    Weight router;          // BF16 contiguous [513,2560]
    Weight routed_gate_up;  // batched blockscale [512,1280,2560]
    Weight routed_down;     // batched blockscale [512,2560,640]
    Weight shared_gate_up;  // BF16 [1280,2560]
    Weight shared_down;     // BF16 [640,1280]
};
void moe_routed_nvfp4(const Tensor& x, const MoeNvfp4Weights& weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream);
```

数值契约：路由 top-10 + `norm_topk_prob`（softmax 在 top-10 内归一化）、每专家贡献按路由系数缩放求和、共享专家贡献一次（不加路由系数）、输出累加到 `destination`。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add include/ninfer/ops/moe_nvfp4.h src/ops/moe_nvfp4 tests/ops/moe_nvfp4_test.cpp src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(ops): add NVFP4 routed experts with top-10 normalization"
```

---

### 任务 7：PLE（hash、行查表、gate、conv + oracle）

**文件：**
- 创建：`include/ninfer/ops/ple.h`
- 创建：`src/ops/ple/{hash.cpp,lookup.cu,gate.cu,conv.cu}`
- 测试：`tests/ops/ple_test.cpp`

- [ ] **步骤 1：编写失败的测试**

- hash：给定 token 序列与尾部两 token，按上游 `hash_tokens` 公式（pair = id*mul0 ^ tail1*mul1；triple = pair ^ older*mul2；16 个头：头 0..7 用 pair、8..15 用 triple；`index = word % vocab_sizes[h]`（负数取正余数）再 `+ offsets[h]`）计算期望行号；乘数/vocab/offset 取自工件 `resource/text/ple_hash_controls`（280 B，解析为 16 组控制值）。
- 行查表：NF4 分页行（`row_paged_g16_4k_v1`）按 `row_paged_nf4_row_offset` 取 160 个码 + 10 个 E4M3 组缩放，按 NF4 codebook 解码为 BF16（查表出口精确舍入），断言与独立解码一致。
- gate：`k_s = Norm(projected.key[:,s], norm_key[:,s])`、`q_s = Norm(residual[:,s], norm_query[:,s])`、`a_s = sum_d k_s*q_s/sqrt(H)`、`v_s = sign(a_s)*sqrt(max(|a_s|,1e-6))`、`g_s = sigmoid(v_s)`、`ideal_U[d,s] = g_s*projected.value[d]`、`ideal_N[:,s] = Norm(U[:,s], norm_conv[:,s])`；
- conv：`v[c,t] = sum_{j=0..3} kernel[c,j]*N[c,t-9+3j]`（dilation 3）、`ideal_Y[d,s] = residual[d,s] + gated_value[d,s] + SiLU(v[t])`（广播到流）、`ideal_Z[:,s] = Norm(Y[:,s], next_hc_norm_weight[:,s], eps)`。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_ple_test && ./build/tests/ninfer_ple_test`
预期：编译失败。

- [ ] **步骤 3：实现契约与实现**

`include/ninfer/ops/ple.h`：`ple_hash_tokens`（host 侧，签名含 ids、tail、eos、controls、输出 `std::span<std::uint32_t>`）、`ple_lookup_rows`（按行号从 NF4 表取行并解码为 BF16，host 或 device 路径由包决定）、`ple_gate`、`ple_conv_add_norm`、`ple_state_extract`。数学逐条抄上游 `include/ninfer/ops/ple.h`。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add include/ninfer/ops/ple.h src/ops/ple tests/ops/ple_test.cpp src/CMakeLists.txt tests/CMakeLists.txt
git commit -m "feat(ops): add the PLE hash, lookup, gate and convolution ops"
```

---

### 任务 8：家族运行时骨架（准入 → prefill → decode → 发布）

**文件：**
- 创建：`src/targets/qwen4_exp/export/ninfer/targets/qwen4_exp/runtime.h`、`export/ninfer/targets/qwen4_exp/frontend.h`
- 创建：`src/targets/qwen4_exp/impl/runtime/{instance.h,instantiate.h,api_impl.h,program.h,program_impl.h,schedule.h,layouts.h,workspace_recipe.h}`
- 测试：`tests/targets/qwen3_8_flash_next/test_program_lifecycle.cpp`

- [ ] **步骤 1：编写失败的测试**

- `SequencePlanner` → `finalize(main_page_groups)` → `SequencePlan`：容量、KV 容量、并发上限、设备预留字节、workspace 字节均为非零且与请求的 `max_context` 单调；
- `Program::plan_request_base` 拒绝超过容量的请求与非法 token id；
- `Program::start_prefill_lane` → `advance_prefill_lane` 走完一个短提示；`decode_batch` 产出一轮结果；`abort_lane` 释放槽位；
- 家族不持有身份：断言 `sizeof` 与暴露接口里没有任何 `model_id` 字段（编译期无法断言的部分改为文档约定）。

- [ ] **步骤 2：运行测试验证失败**

运行：`/home/cmj/strata/.venv/bin/ninja -C build ninfer_flash_next_program_lifecycle_test && ./build/tests/ninfer_flash_next_program_lifecycle_test`
预期：编译失败（家族不存在）。

- [ ] **步骤 3：实现家族骨架**

- `export/.../runtime.h`：照 `qwen3_6/export/ninfer/targets/qwen3_6/runtime.h` 的五个 PIMPL 模板形态（`SequencePlan`/`SequencePlanner`/`RequestBasePlan`/`RequestPlan`/`Program`），只保留文本路径需要的方法（不做 MTP/DFlash/Vision/混合批/前缀缓存/CUDA Graph）。
- `impl/runtime/instance.h`：`using Variant = NINFER_QWEN4EXP_VARIANT;` + 别名（`WeightsProfile`/`TextConfig`/`ModelView`/三个叶权重类型/`GraphExecutionProfile`）+ 家族常量（`attention_scale`、`gdn_scale`、`prefill_chunk_alignment`、`maximum_context = 262144`）。
- `impl/runtime/instantiate.h`：共享源码 include 清单（`layouts.h` → `program.h` → `schedule.h` → `text_context.h` → `api_impl.h` → 对应 `*_impl.h`），供包 TU 定义宏后 include 一次。
- `impl/runtime/api_impl.h`：五个模板类的显式特化出口。
- `impl/runtime/program_impl.h`：准入（容量/页数/采样校验）、prefill 分块推进、decode 轮、token 发布（采样 → 事务 → 输出）。
- `impl/runtime/schedule.h`：`prefill_text_chunk`、`ordinary_decode_batch`、`sample_from_hidden`。
- `impl/runtime/workspace_recipe.h`：workspace 根与容量（隐藏 2560、残差 10240、QSA 缓冲区、MoE scratch、PLE 缓冲）。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/targets/qwen4_exp tests/targets/qwen3_8_flash_next/test_program_lifecycle.cpp tests/CMakeLists.txt
git commit -m "feat(qwen4-exp): add the identity-free family runtime skeleton"
```

---

### 任务 9：文本主干（HC → 48 层 → 主头）与缩减配置 greedy

**文件：**
- 创建：`src/targets/qwen4_exp/impl/runtime/text_context.h`、`text_context_impl.h`
- 创建：`src/targets/qwen3_8_flash_next/impl/variant.h`、`impl/variant.cpp`（三个执行叶）
- 修改：`src/targets/qwen3_8_flash_next/impl/load/bindings.{h,cpp}`（装载时按角色解码权重视图；M2 只需要 BF16/Q8 行拆与 NVFP4 batched 的**几何视图**，不解码 NVFP4 数值）
- 测试：`tests/targets/qwen3_8_flash_next/test_text_forward.cpp`

- [ ] **步骤 1：确定 M2 的执行配置（写入文档）**

M2 的执行配置（写入 `docs/maintainer/qwen3.8-flash-next-artifact.md` 的“验收配置”一节）：
- 层：前 4 层（layer 0 GDN、layer 1 GDN + PLE、layer 2 GDN、layer 3 QSA）→ 覆盖 GDN、PLE、QSA 三条路径；
- 专家：全部 512 个（4 层 × 2 个 batched 对象 ≈ 5.7 GB，单卡可容）；
- 头与嵌入：完整（各 ≈1.27 GB BF16）；PLE 表留盘，按行查表；
- 该配置仅用于 M2 数值验收，公共 Engine 路径始终按完整 48 层装载（装载计划不因此裁剪）。

- [ ] **步骤 2：编写失败的测试**

`test_text_forward.cpp`（env 门控 `NINFER_FLASH_NEXT_ARTIFACT`，无工件返回 77）：
- 用真实工件装载前 4 层，喂 8 个 token 的提示，断言：
  - 产出 1 个 greedy token，token id 在 `[0, 248320)`；
  - 与任务 10 的参考模型在同配置下的 greedy token **完全一致**（参考模型未就绪前该断言用占位参考值，写成读环境变量 `NINFER_FLASH_NEXT_REFERENCE_TOKEN` 的比较；任务 10 落地后改为直接调用参考模型）；
  - 重复同一提示两次得到相同 token（确定性）；
  - GDN 状态在两轮 decode 后被更新（FP32 状态池的条数随层数增长）。

- [ ] **步骤 3：运行测试验证失败**

运行：`NINFER_FLASH_NEXT_ARTIFACT=... ./build/tests/ninfer_flash_next_text_forward_test`
预期：编译失败（家族/包未接线）。

- [ ] **步骤 4：实现文本主干**

- `qwen4_exp` 的 `text_context_impl.h` 层循环（严格照上游 `src/models/qwen4_exp/execution/text.cpp` 的主干顺序）：
  1. **stem**：`ops::embedding` → `hc_expand_norm(input, layers[0].attention_hc.norm, eps)` → 四路 `residual` 与 `normalized`；
  2. **每层 i 的 local_mixer**：`hc_mix(normalized, layers[i].attention_hc.mix)` → `block_input` + `injection` → mixer（GDN 或 QSA）消费 `block_input` → `block_output`；
  3. **每层 i 的 finish_mixer**：`hc_combine_norm(residual, block_output, injection, layers[i].moe_hc.norm, eps)` → 更新 `residual` 与 `normalized`；随后 `hc_mix(normalized, layers[i].moe_hc.mix)` → 新的 `block_input` + `injection`；
  4. **每层 i 的 MoE**：router → top-10（`norm_topk_prob`）→ routed NVFP4 专家 + 共享专家 → `block_output`；
  5. **每层 i 的 combine**：`hc_combine_norm(residual, block_output, injection, next_norm, eps)`，其中 `next_norm = layers[i+1].attention_hc.norm`；**若 `i + 1 == ple.layer`（本工件 ple.layer = 1，即第 0 层之后）则改为不带归一化的 `hc_combine`**，紧接着执行 PLE（`ple_lookup_rows` → `ple_gate` → `ple_conv_add_norm`，其 next HC 取 `layers[ple.layer].attention_hc`），PLE 之后再进入第 1 层的 attention HC；
  6. **finish**：`hc_final_mix(normalized, final_hc.mix)` → `sample_hidden` → 主头（BF16 `[248320,2560]`）→ logits → `ops::sample`（greedy）。MTP conditioning 取 finish 之前的四路 `residual`（M3 用）。
- GDN 叶（包侧 `variant.cpp`）：`gdn_control_proj`（对 HC 混合流）→ `linear_split`（QKVZ）→ `causal_conv1d_silu_split`（Q/K/V 过卷积，Z 绕过）→ `gated_delta_net`（FP32 状态，scale = 1/sqrt(key_dim)）→ `sigmoid_gated_rmsnorm` → 输出投影。
- QSA 叶（包侧 `variant.cpp`）：`linear_split`（query+gate 一父、K/V/index_key 一父）→ `qsa_prepare` → `qsa_indexer` → `qsa_sparse_gqa` → 输出投影。
- 状态：GDN FP32 循环态（`LinearAttentionStatePool`，按层 × 槽位）、QSA tail/index cache、KV（INT8 group-64）、PLE 卷积历史。

- [ ] **步骤 5：运行测试验证通过**

运行：同步骤 3。预期：PASS（greedy token 与占位参考一致）。

- [ ] **步骤 6：Commit**

```bash
git add src/targets/qwen4_exp src/targets/qwen3_8_flash_next docs/maintainer/qwen3.8-flash-next-artifact.md tests/targets/qwen3_8_flash_next/test_text_forward.cpp tests/CMakeLists.txt
git commit -m "feat(flash-next): run the four-stream text trunk through the QSA and GDN layers"
```

---

### 任务 10：参考模型与端到端 parity

**文件：**
- 创建：`tools/reference/qwen3_8_flash_next/{__init__.py,config.py,bindings.py,weights.py,ops.py,model.py,state.py,text.py,cli.py,requirements.txt}`
- 创建：`tests/targets/qwen3_8_flash_next/test_reference_parity.py`

- [ ] **步骤 1：编写失败的测试**

`test_reference_parity.py`（pytest，env 门控 `NINFER_FLASH_NEXT_ARTIFACT`）：
- 参考模型对同一提示的 greedy token 与 C++ 实现在前 4 层配置下一致；
- 参考模型的 PLE hash 行号与 C++ `ple_hash_tokens` 对同一 token 序列完全一致（逐头比对）；
- 参考模型的 QSA 块选择集合与 C++ `qsa_indexer` 的选择一致（逐 token 比对选中的块 id 集合）。

- [ ] **步骤 2：运行测试验证失败**

运行：`python3 -m pytest tests/targets/qwen3_8_flash_next/test_reference_parity.py -x`
预期：FAIL（参考模型不存在）。

- [ ] **步骤 3：实现参考模型**

照 `tools/reference/qwen3_6_27b/` 的结构实现 Flash-Next 参考（复用 `tools/reference/qwen3_6/common/` 的前端与采样）：
- `config.py`：M2 验收配置常量（层数 4、hidden 2560、HC 4×320、QSA 几何、专家 512/top-10/640、词表 248320、rope [11,11,10]/theta 1e7/partial 0.25）；
- `bindings.py`：从工件目录的 `bindings` 解析参数名 → 对象，按角色绑定（复用 `tools.artifact` 的读取器）；
- `weights.py`：预算感知的物化（NVFP4 batched 专家按行解码、NF4 PLE 表按行读取、Q8 行拆按既有 `dequantize_row_split`）；
- `ops.py`：`hc_expand_norm`/`hc_grouped_rmsnorm`/`hc_combine`/`hc_combine_norm`/`hc_mix`/`hc_final_mix`/`qsa_*`/`ple_*`/`topk_softmax_route` 的 **FP32** 参考实现（不复制生产 kernel 的中间精度）；
- `model.py`/`state.py`/`text.py`：4 层前向 + 采样循环。

- [ ] **步骤 4：运行测试验证通过**

运行：同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add tools/reference/qwen3_8_flash_next tests/targets/qwen3_8_flash_next/test_reference_parity.py
git commit -m "feat(tools): add the qwen3.8-flash-next reference model and parity test"
```

---

## 自检记录

- **规格覆盖度**：M2 的目标是"家族 + 包骨架 + 单卡 greedy 出 token + 参考 parity + PLE/HC/QSA/GDN/MoE 全链路"。任务 1/2 提供容器与语义基线，任务 3 提供包与 registry，任务 4–7 提供四个缺位算子族，任务 8 提供家族运行时，任务 9 提供文本主干与 greedy，任务 10 提供参考与 parity。规格 §8 的 M2 条目逐项有对应任务；MTP（M3）、TP2 与专家缓存（M4）、发布面与性能（M5）、Vision（M6）按里程碑留在后续计划，不在本计划内。
- **步骤扫描**：每个任务都有"写失败测试 → 跑失败 → 实现 → 跑通过 → 提交"的可执行步骤；数学与索引约定全部指向任务 2 建立的维护者文档（上游权威），不在计划里复述实现。
- **类型一致性**：`Role`/`roles::parameter_name`（任务 2）、`HcMixWeights`/`HcFinalMixWeights`（任务 4）、`QsaProjectionView`/`QsaBlockSelection`（任务 5）、`MoeNvfp4Weights`（任务 6）、`V3Directory`/`V3Use`（任务 1）在后续任务中同名同指；`Variant`/`WeightsProfile`/`TextConfig` 与 `qwen3_6` 家族同形。
- **审查重点**：五条审查重点分别由任务 7（PLE hash/行地址）、任务 5（QSA 块选择）、任务 6（NVFP4 专家几何）、任务 4（HC 注入系数）、任务 9/任务 4（GDN 的 Z 与卷积位置）中的具体断言钉住。
- **比例**：计划 10 个任务、约 560 行，对应规格 §5/§6/§8 的一个里程碑；公式细节落在维护者文档而非计划内。

## 计划未尽事项（留给后续里程碑）

- 完整 48 层 + 512 专家的单卡执行需要专家 host 支撑与按需上传（规格 §5.4），属 M4；M2 的执行配置因此限定为前 4 层。
- TP2 分片图（专家 intermediate K 切分、头按词表切分、down allreduce）属 M4。
- MTP 私有 QSA 层、块 FP8 专家库、Q4 草头属 M3。
- CUDA Graph 捕获、并发准入扩展、前缀缓存属 M5 前的家族完善。
