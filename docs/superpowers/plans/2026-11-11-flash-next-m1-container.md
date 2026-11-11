# Flash-Next M1（v3 多分片容器）实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 subagent-driven-development（推荐）或 executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 本仓 Reader 直接读取上游 v3 多分片 `.ninfer` 工件（入口 + NINPRT 续卷），支持 Qwen3.8-Flash-Next 工件用到的全部新格式/布局，并在真实工件上通过对象级验收。

**架构：** `src/artifact/reader` 增加一条通用 v3 读取路径（多分片打开、目录解析、逻辑地址空间→分片文件映射、身份合成），与现有 27B 窄适配并存；`src/artifact/storage_layouts` 增加三个新几何；`src/core` 只在编译器要求的范围内扩展 dtype 映射。对象名直接暴露上游 v3 object id，不做 27B 式改名投影——Flash-Next 包绑定（M2）按上游 id 绑定。

**技术栈：** C++20 / CUDA 12.8 / sm_70；测试为本仓 `ninfer_add_test`（Catch 风格，见既有 `tests/test_ninfer_artifact_reader.cpp`）。

**规格：** `docs/superpowers/specs/2026-11-11-qwen4exp-flash-next-nvfp4-design.md`（本计划实现其里程碑 M1，并把其中"binder 全量绑定"收窄为对象级验收 + 单对象 TP 切分冒烟——包级绑定图属于 M2 的包 binder）。

## 全局约束

- 容器事实以两处为准：上游 `src/artifact/framing.h`（magic/头/对齐常量）与真实工件 `qwen3_8_flash_next.ninfer`（下载中：`/home/cmj/models/Qwen3.8-Flash-Next-NInfer/`）。
- v3 头：入口 magic `NINFER\0\x03`，续卷 magic `NINPRT\0\x03`，头 32B（+8 u64 字段、+16 起 16B artifact_id），payload 对齐 4096；分卷 payload 从文件内偏移 4096 开始；逻辑 payload = 入口 payload + 各续卷 payload 顺序拼接。
- 目录 JSON（上游 v3 schema）：`metadata.name`、`files[{path,payload_bytes}]`（`files[0].path == null`）、`objects[{id,kind,format,layout,shape,offset,bytes,...}]`、`bindings`、`uses`、`components`。
- 本仓 v2 契约不变：v2 工件、27B/coder390 窄 v3 适配的行为一字不改。
- 新格式拼写（上游 `formats.cpp`）：`bf16`/`fp32`/`int32`/`q4_g64_fp16`/`q8_g32_fp16`/`nvfp4`/`nf4_g16_e4m3fn`/`fp8_e4m3fn_block128x128_bf16`；新布局拼写：`row_paged_g16_4k_v1`、`block_scale_m128_k128_v1`、`batched_block_scale_k16_m128x4_v1`。
- 几何公式（上游 `src/core/weight_view.cpp`）：
  - NF4 分页行：`row_bytes = k/2 + k/16`，`rows_per_page = 4096/row_bytes`，`page_count = ceil(n/rows_per_page)`，`matrix_scale_offset = page_count*4096`，`bytes = matrix_scale_offset + 4`；
  - Q8_G32 row-split：group 32，码 1B/值、缩放 2B/组，列对齐 128，平面序 code(256 对齐)→high(0B)→scale(256 对齐)，无矩阵缩放；
  - 块 FP8 128×128：n%128=k%128=0，码 1B/值，缩放 2B/块，scale_offset=align(code,256)，`scale_bytes = (n/128)*(k/128)*2`；
  - batched NVFP4：n%128=0、k%64=0，group 16，码 k/2/行、缩放 k/16/行，逐矩阵平面拼接，每矩阵尾随 4B FP32 除数。
- 主机 CPU 占用上限 ~85%（构建并发据此配额）。

## 审查重点（Review Focus）

1. **对象跨分卷边界**：真实工件按字节预算切卷，**确有对象跨卷**（`weight/000028`、`weight/000074`、`weight/000171`）——上游用 `segments()` 把跨界对象拆成多段读。跨卷对象必须被支持：不得静默错读，也不得整体拒绝；`payload()` 的单段语义对跨界对象显式报错，`read_direct`/materializer 必须分段取数。→ 任务 1 的 `crossing_object_rejected` 改为 `crossing_object_read_in_segments`，任务 5 用真实工件验证这 3 个对象的分段读字节正确。
2. **续卷归属校验**：part index / artifact_id 不匹配的续卷文件（或两卷互换）必须报错，不得拼接。→ 任务 1 的 `wrong_part_index`、`swapped_parts` 用例。
3. **NF4 不完整末页**：`rows_per_page=45`（2500012%45=7），末页不满时的 page_count 与 matrix_scale_offset 必须按 ceil 计算，错一位会让 PLE 全表错位。→ 任务 3 的精确字节数断言（227,553,284）。
4. **q8_g32 与 W8G32 的语义差异**：两者同构是本计划假设（复用 `W8G32_F16S`），必须用真实工件采样字节序验证平面序与缩放 dtype，不等则在任务 2 改为新增枚举并回归任务 3。→ 任务 5 的对表步骤。
5. **offset/bytes 与几何自洽**：每个对象 `bytes == tensor_encoded_size(layout,format,shape)` 且 `offset % alignment == 0`（上游 `describe_tensor` 同款校验），不满足即工件被误读。→ 任务 5 全量扫描。

---

### 任务 1：v3 多分片读取（framing + 目录 + 地址映射）

**文件：**
- 修改：`src/artifact/reader.cpp`（新增通用 v3 路径；现有 `Qwen38Nvfp4V3Adapter` 与 v2 路径不动）
- 修改：`src/artifact/reader.h`（`Reader::Impl` 私有结构按需调整；公共接口不变）
- 测试：`tests/test_v3_artifact_reader.cpp`（新建，`tests/CMakeLists.txt` 用 `ninfer_add_test(ninfer_v3_artifact_reader_test SOURCES test_v3_artifact_reader.cpp)` 注册）

- [ ] **步骤 1：编写失败的测试**

测试内自建合成 v3 工件（直接写字节，不依赖 Python fixture）：入口（magic `NINFER\0\x03`、头 32B、`metadata.name="qwen3.8-flash-next"` 的目录 JSON、两个对象）+ 两个续卷（magic `NINPRT\0\x03`、index 1/2、同一 16B id）。用例：
- `identity_synthesized`：`identity()` 返回 `{model_id="qwen3.8-flash-next", weights_id="nvfp4"}`；
- `objects_and_payloads_across_parts`：入口对象与两个续卷各一对象的 `payload()` 内容逐字节等于写入值；
- `wrong_part_index_rejected`：index 写 3 → 打开抛 `ArtifactError`；
- `swapped_parts_rejected`：两续卷文件互换 → 抛；
- `foreign_part_id_rejected`：续卷 artifact_id 改一字节 → 抛；
- `crossing_object_rejected`：一个对象逻辑区间跨两卷 → 改为 `crossing_object_read_in_segments`：跨卷对象被正确拆段读出（分段内容逐字节等于写入值），`payload()` 对该对象显式报错。

- [ ] **步骤 2：运行测试验证失败**

运行：`cmake --build build --target ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：编译失败或 `identity` 返回错误值（v3 多分片当前不被支持）。

- [ ] **步骤 3：实现通用 v3 路径**

`reader.cpp` 中新增（与现有 `Qwen38Nvfp4V3Adapter` 平行的内部类，命名 `V3MultiPartReader`）：
- 打开入口：校验 `kV3Magic`（现 39-42 行常量复用），读 32B 头（+8 的目录长度 u64、+16 起 16B artifact id），目录 JSON 解析上游 v3 schema；`metadata.name → (model_id, weights_id)` 用 `constexpr` 映射表（本期一项：`"qwen3.8-flash-next" → ("qwen3.8-flash-next","nvfp4")`，未知 name 抛错并提示）；
- 续卷：按 `files[]` 顺序在入口同目录打开 `files[i].path`，逐个校验 magic、`read_u64_le(header+8)==i`、artifact id 逐字节相等、文件大小 == `4096 + payload_bytes[i]`（沿用上游 `open_file` 的校验集，`src/artifact/reader.cpp` 现有 `MappedFile` 复用为每卷一个映射）；
- 逻辑地址：逻辑原点 = 入口 payload 首字节（`entry_payload_start = align_up(32 + json_bytes, 4096)` 是它所在的文件内偏移）；第 i 卷逻辑基址 = `Σ_{j<i} payload_bytes[j]`，卷内偏移 = `4096 + (logical - base)`；对象 `offset+bytes` **允许跨卷**（上游按字节预算切卷，真实工件有 3 个跨界对象），跨卷对象按段拆解；
- 跨卷支持（上游 `segments()` 语义）：每个对象可解析为 1..N 段 `(volume, file_offset, length)`；`read_direct` 对跨卷区间分段 `pread` 进同一目标缓冲（每段起点 4096 对齐）；`payload()` 的单段语义对跨界对象显式报错（"object spans volumes"），不得算出越界 span；materializer 按段发起读取（见 `src/artifact/materializer.cpp` 的对齐/合并逻辑：读区间在卷界处切开，不得因对齐尾越过卷尾而失败）；
- 对象表：`objects[]` 的 `id/format/layout/shape/offset/bytes` 直接构造 `TensorDescriptor`/`ResourceDescriptor`（kind 分派；format/layout 字符串经任务 2 的解析函数——本任务先用桩映射 `bf16→BF16` 等已知项，任务 2 完成全量）；
- `payload(object)` 返回所在卷映射内的 span（须校验 `bytes` 不越出所在卷）；`read_direct(absolute_offset, dst)` 以逻辑偏移解析卷后 `pread`（4096 对齐约束保持）。
- [ ] **步骤 4：运行测试验证通过**

运行：`cmake --build build --target ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：PASS（6 用例全绿）。

- [ ] **步骤 5：Commit**

```bash
git add src/artifact/reader.cpp src/artifact/reader.h tests/test_v3_artifact_reader.cpp tests/CMakeLists.txt
git commit -m "feat(artifact): read v3 multi-part .ninfer artifacts"
```

### 任务 2：v3 格式/布局枚举与拼写映射

**文件：**
- 修改：`src/artifact/reader.h:21-41`（枚举扩展）
- 修改：`src/artifact/reader.cpp:98-120`（`parse_format`/`parse_layout` v3 拼写映射）、`format_name`/`layout_name`
- 测试：`tests/test_v3_artifact_reader.cpp`（追加用例）

- [ ] **步骤 1：编写失败的测试**

追加用例 `v3_spelling_round_trip`：对上游 v3 拼写（`"bf16","fp32","int32","q4_g64_fp16","q8_g32_fp16","nvfp4","nf4_g16_e4m3fn","fp8_e4m3fn_block128x128_bf16"` 与布局 `"contiguous_le_v1","row_split_k128_v1","row_paged_g16_4k_v1","block_scale_k16_m128x4_v1","batched_block_scale_k16_m128x4_v1","block_scale_m128_k128_v1"`）断言 `parse → format_name/layout_name → 拼写` 往返相等；未知拼写抛 `ArtifactError`。

- [ ] **步骤 2：运行测试验证失败**

运行：`cmake --build build --target ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：FAIL（枚举不存在）。

- [ ] **步骤 3：扩展枚举与映射**

`NumericFormat` 增加 `NF4_G16_E4M3S`、`FP8_E4M3FN_BLOCK128_BF16S`；`q8_g32_fp16` 映射到既有 `W8G32_F16S`（任务 5 对表兜底）；`StorageLayout` 增加 `RowPagedG16x4KV1`、`BlockScaleM128K128V1`、`BatchedBlockScaleK16M128x4V1`；`parse_format`/`parse_layout` 增加上述 v3 拼写（v2 大写拼写不动，两条命名并存）；`format_name`/`layout_name` 补齐新枚举（v3 小写拼写）。

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/artifact/reader.h src/artifact/reader.cpp tests/test_v3_artifact_reader.cpp
git commit -m "feat(artifact): accept v3 formats and layouts for Flash-Next"
```

### 任务 3：三个新布局的几何与切分

**文件：**
- 修改：`src/artifact/storage_layouts.cpp`（新几何 + 切分分派）
- 修改：`src/artifact/reader.h`（新几何结构体与函数声明，放在 `RowScaleGeometry` 之后）
- 测试：`tests/test_v3_artifact_reader.cpp`（追加几何用例）

- [ ] **步骤 1：编写失败的测试**

用例（全部硬编码期望值，按全局约束公式）：
- `nf4_paged_geometry`：`row_paged_nf4_geometry(NF4_G16_E4M3S, [2500012,160])` → `row_bytes=90, rows_per_page=45, page_count=55556, matrix_scale_offset=227553280, encoded_bytes=227553284`；`k%16!=0` 或格式不符抛错。
- `block128_geometry`：`block_scale_m128_k128_geometry(FP8_E4M3FN_BLOCK128_BF16S, [2560,640])` → `code_plane_bytes=1638400, scale_plane_offset=1638400, scale_plane_bytes=200, encoded_bytes=1638600`；`n%128||k%128` 抛错。
- `batched_blockscale_geometry`：`batched_block_scale_geometry(NVFP4, [512,1280,2560])` → 单矩阵 `code_plane_bytes=1638400, scale_plane_bytes=204800, matrix_stride=1843200, matrices=512, encoded_bytes=943720448`（512×1843200 + 512×4）；`[512,2560,640]` → `matrix_stride=921600, encoded_bytes=471859200+2048`；`n%128||k%64` 抛错。
- `batched_slices`：对 `[512,1280,2560]` 断言 `tensor_row_slice`（专家区间 `[0,256)`）与 `tensor_column_slice`（K 区间 `[0,1280)`，TP2 每专家半宽）产生的 `encoded_bytes` 与拷贝数符合"逐矩阵应用现有 BlockScale 切分"的构造（256 个专家块的拷贝区间数与单矩阵规则一致）。

- [ ] **步骤 2：运行测试验证失败**

运行：`cmake --build build --target ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：FAIL（函数不存在）。

- [ ] **步骤 3：实现几何与切分**

`storage_layouts.cpp`：
- `RowPagedNf4Geometry row_paged_nf4_geometry(NumericFormat, span<const u64> shape)`：按公式实现；只接受 `NF4_G16_E4M3S` 与二维 `[rows, k]`、`k%16==0`；
- `Block128Geometry block_scale_m128_k128_geometry(NumericFormat, shape)`：复用 `row_scale_geometry` 的平面骨架（码面 + 256 对齐 + 缩放面），块 128×128、BF16 缩放 2B；
- `BatchedBlockScaleGeometry batched_block_scale_geometry(NumericFormat, shape)`：`matrices = shape[0]`，逐矩阵复用现有 `block_scale_geometry(NVFP4, [shape[1], shape[2]])`，矩阵间紧凑拼接，每矩阵尾随 4B FP32 除数；`tensor_encoded_size` 分派补齐；
- 切分：`tensor_row_slice`/`tensor_column_slice` 对 `BatchedBlockScaleK16M128x4V1` 按"逐矩阵应用现有 BlockScale 规则"实现（行切分=整专家区间、列切分=单 K 区间且 `k%16==0`）；`RowPagedG16x4KV1` 本期不实现切分（PLE 按整 shard 复制/分派），调用即抛。

- [ ] **步骤 4：运行测试验证通过**

运行同步骤 2。预期：PASS。

- [ ] **步骤 5：Commit**

```bash
git add src/artifact/storage_layouts.cpp src/artifact/reader.h tests/test_v3_artifact_reader.cpp
git commit -m "feat(artifact): geometries for NF4-paged, block128 FP8, batched NVFP4"
```

### 任务 4：dtype 映射补口（仅编译面）

**文件：**
- 修改：`src/core/tensor.h`（`TensorDType`/量化枚举按需增加 NF4、块 FP8 两项）
- 修改：`src/artifact/typed_binding.cpp`（`NumericFormat → TensorDType` 映射补两行；`validate_only`/`materialize` 路径对新格式先抛"解码在 M2 实现"）

- [ ] **步骤 1：编写失败的测试**

`tests/test_v3_artifact_reader.cpp` 追加 `validate_only_accepts_v3_objects`：对合成工件里一个 `nf4` 对象和一个块 FP8 对象调用 `Binder::require_tensor(...).validate_only().finish()`，断言不抛且 `MaterializationPlan` 只含 validate 项。

- [ ] **步骤 2：运行测试验证失败**

运行：`cmake --build build --target ninfer_v3_artifact_reader_test && ./build/tests/ninfer_v3_artifact_reader_test`
预期：FAIL（dtype 映射缺失抛错）。

- [ ] **步骤 3：补映射**

`tensor.h` 枚举加 `NF4_G16_E4M3S`、`FP8_E4M3FN_BLOCK128_BF16S`（值序追加尾部）；`typed_binding.cpp` 映射补两行；materialize 对这两个 dtype 抛 `ArtifactError("decode lands in the qwen3.8-flash-next package")`——显式边界，不是占位符。

- [ ] **步骤 4：运行测试验证通过**；同时跑既有 `ninfer_artifact_reader_test`、`ninfer_artifact_materialization_test`、`ninfer_tensor_slice_test` 确认零回归。

- [ ] **步骤 5：Commit**

```bash
git add src/core/tensor.h src/artifact/typed_binding.cpp tests/test_v3_artifact_reader.cpp
git commit -m "feat(core): dtype slots for NF4 and block-FP8 (decode in qwen4_exp package)"
```

### 任务 5：真实工件验收（下载完成后执行）

**文件：**
- 测试：`tests/targets/test_flash_next_reader_real.cpp`（新建；env 门控 `NINFER_FLASH_NEXT_ARTIFACT`，未设时 skip）
- 注册：`tests/CMakeLists.txt`（`ninfer_add_test`，与其它 target 测试同节）

- [ ] **步骤 1：编写（先跳过的）测试**

门控为真时断言：
- `identity == {qwen3.8-flash-next, nvfp4}`；
- 全量对象扫描：按 format 计数 `{nvfp4:96, q8_g32_fp16:96, nf4:128, fp8_block128:2, bf16:916, fp32:72, q4_g64_fp16:1}`（以工件旁 `artifact-manifest.json` 为准，若与清单不符，以清单修正断言并记录）；
- 每个对象 `bytes == tensor_encoded_size(...)` 且 `offset % tensor_alignment(layout) == 0`（审查重点 5）；
- 3 个跨界对象（`weight/000028`、`weight/000074`、`weight/000171`）被正确分段读出：分段拼接后的字节与卷内原始字节一致（上游 `segments()` 语义）。

- [ ] **步骤 2：运行验证**

```bash
NINFER_FLASH_NEXT_ARTIFACT=/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer \
  ./build/tests/ninfer_flash_next_reader_real_test
```
预期：PASS；下载未完成前该测试 skip。

- [ ] **步骤 3：q8_g32 ↔ W8G32 对表**

对 4 个抽样 `q8_g32_fp16` 对象：按上游 row-split 公式手算几何与平面偏移，断言与 `row_split_geometry(W8G32_F16S, shape)` 完全一致；再用 `payload()` 前 64B 抽查缩放面 FP16 值的量级合理（|scale| ∈ (1e-3, 1e3)）。不一致时：新增 `Q8G32_F16S` 枚举替代映射，回归任务 2/3。

- [ ] **步骤 4：Binder TP 切分冒烟**

同一测试文件：对一条 nvfp4 `gate_up` 库对象（shape `[512,1280,2560]`）执行 `require_tensor(...).device(...)` 的 TP2 `ShardPlacement`（专家轴各 256）+ K 轴半宽切分，断言两 rank 的分片 encoded_bytes 等于任务 3 公式的半量、无越界拷贝。

- [ ] **步骤 5：Commit**

```bash
git add tests/targets/test_flash_next_reader_real.cpp tests/CMakeLists.txt
git commit -m "test(artifact): accept the real qwen3.8-flash-next v3 artifact"
```

## 自检记录

- 规格覆盖度：M1 的 reader/格式/几何/真实工件验收四要素各有任务；规格 M1 中"binder 全量绑定（含 TP2 分片图）"已在计划头部声明收窄到包级 binder（M2），并保留任务 5 步骤 4 的单对象切分冒烟作为桥。
- 步骤扫描：每个代码步骤给出签名/文件/精确数值或"复用现有规则"的单一做法；函数体留给实现者的只有机械实现。
- 类型一致性：`NF4_G16_E4M3S`、`FP8_E4M3FN_BLOCK128_BF16S`、`RowPagedG16x4KV1`、`BlockScaleM128K128V1`、`BatchedBlockScaleK16M128x4V1` 在任务 2/3/4/5 中同名同指。
- 审查重点：五条均落到具体用例（任务 1 ×3、任务 3 ×1、任务 5 ×2）。
- 比例：计划 ≤ 规格，无代码抄写。
