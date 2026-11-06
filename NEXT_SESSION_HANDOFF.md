# NInfer FP16 文本 runtime 交接文档

> 最后更新：2026-11-01（第二次会话）
> 仓库：`/home/cmj/ninfer-v100-fast`
> 分支：`fp16-text-runtime`
> HEAD：`4e07a7dd`（`wip(fp16): checkpoint FP16 op layer and conv state layout fix`）
> 工作区：本会话新增的 FP16 attention/append 与 NVFP4 全 T 改动，**尚未 commit**。
> 产品状态：BF16 文本路径可用；FP16 文本 runtime 仍在 ops 层推进，尚未接入 Engine。

---

## 0. 一句话结论

FP16 的 ops 层正在一块块补齐，且每块都有独立 oracle。当前**已原生支持**：
FP16 activation + INT8 group-64 KV 的 GQA append/attention（decode、cached、prompt/chunked），
以及 FP16 的 NVFP4 linear 在所有 T（1…4096+）。
**尚未支持**：FP8 output head、W8/Q4 MTP、NVFP4 融合叶子、GDN 融合叶子、以及 runtime 的
编译期 dtype 接线。在这些完成前，Engine 不能切到 FP16。

---

## 1. 关键事实（本会话新确认）

1. **V100 没有原生 BF16 tensor core。** 现有 BF16 kernel 在 MMA 前已经把 BF16 转成 FP16
   （见 `fp16_volta_mma_gemm.cuh`、`nvfp4_volta_qpn_gemm.cuh` 的 `half` 路径）。
   因此 BF16→FP16 的收益主要是 activation/residual 多 3 位尾数（精度），不是 MMA 速度。
   FP16 与 BF16 都是 2 字节，显存占用相同。
2. **QPN 系列 kernel 早已模板化 `Activation`。** `launch_nvfp4_volta_qpn_with_fp16_activation`
   已存在；缺的只是输出策略和 dispatch。
3. **FP16 + INT8 KV 必须自己实现。** 1cat 的 `fp16_attention_paged_kernel` 只接受 FP16 KV
   cache；release profile 用 INT8 group-64 KV，显存上也不能改成 FP16 KV。
4. 现有 kernel 的输出绝大多数已是模板参数（`class Output` / `OutputPolicy` / `Epilogue`），
   所以输出 FP16 通常只需新增一个 output 结构体 + 实例化；输入 activation 才是主要改动面。

---

## 2. 当前目标（用户明确要求）

- 真正全 FP16 文本推理：FP16 activation/residual/workspace；
- FP16 attention、FP16 GDN、FP16 conv；
- 层间不允许 `FP16→BF16→FP16` 边界转换；
- 用 BF16 原版和 1cat-vLLM 做端到端对比；
- 只有端到端证据支持时才声称完成。

---

## 3. 机器、artifact 与关键路径

| 项 | 值 |
|---|---|
| GPU | 2× Tesla V100-SXM2 32GB，NVLink |
| 构建工具 | `/home/cmj/sglang-v100-venv/bin/ninja -C build` |
| ctest | `/home/cmj/sglang-v100-venv/bin/ctest`（系统无 ctest） |
| CUDA | 12.8，`sm_70` |
| 产品 artifact | `/home/cmj/models/ninfer-artifacts-nvfp4attn/qwen3_8_27b_nvfp4.ninfer` |
| 量化源 | `/home/cmj/models/Qwen3.8-27B-QUASAR-NVFP4` |
| legacy Q4_K_M | `/Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer` |
| BFCL 结果根目录 | `/home/cmj/eval-runs/bfcl/` |
| 诊断/临时证据 | `/home/cmj/tmp_layer_diag/` |

产品契约见 `AGENTS.md`：release target 是 2×32GB、NVFP4、262144 context、
INT8 group-64 KV、MTP-3、eager decode；4×16GB / Q4_K_M / 180000 / TP4 是 legacy regression。

---

## 4. 本会话完成的改动（未提交）

### 4.1 FP16 + INT8 group-64 KV 的 GQA（修复了唯一失败的已提交测试）

新增 `src/ops/kernel/gqa_activation.cuh`：
- `GqaActivation<T>`：`load` / `as_half` / `store` / `pack2`，特化 `__nv_bfloat16` 与 `half`；
- `GqaAppendInputOf<T>`（原 `GqaAppendInput` 保留为 bf16 别名）。

模板化 activation 的 kernel（保持 BF16 路径不变）：
- `gqa_attention_decode_i8.cuh`（SIMT tiled）
- `gqa_attention_decode_i8_tc_volta.cuh`（phase-1）
- `gqa_attention_decode_i8_tc_volta_v2.cuh`（key-split）
- `gqa_attention_prefill_i8.cuh`（prompt attention + i8/i8_page/i4nl fill）
- `gqa_attention_small_t_reduce_output_kernel`（新增 `Out` 模板参数）

host 侧：
- `src/ops/launcher/gqa_attention_decode.cu`：small-T / cached small-T 按 `q.dtype` 分派；
- `src/ops/launcher/gqa_attention_prefill.cu`：append fill、prompt attention、A2 分派；
- `src/ops/wrapper/gqa_attention.cpp`：允许 FP16 q/k/v/out，但**仅当 KV cache 是 I8/I4NL**；
  FP16 + 宽 T 的 VoltaFlash 暂时改走 ChunkedSmallT（正确但非最优，性能后续再做）。

验证：`ninfer_gqa_attention_f16_i8_test` 从失败变为通过；
`ninfer_gqa_attention_test`、`f16_dense/paged/cache`、`long_context`、`kv_append_f16` 全部通过。

### 4.2 FP16 NVFP4 linear 全 T

- `src/ops/linear/nvfp4/nvfp4_output.cuh`：新增 `Nvfp4Fp16ContiguousOutput`；
- `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu`：新增 `launch_nvfp4_volta_qpn_f16`；
- `src/ops/linear/nvfp4/nvfp4_dispatch.cpp`：`nvfp4_f16_dispatch` 现在
  - T ≥ 128 且传入 workspace：CUTLASS FP16（权重 dequant 到 FP16、FP32 累加、FP16 输出）；
  - T ≤ 32：一次 QPN FP16；
  - 33…127：按 32 行切块多次 QPN；
  - `LinearPolicy` 视为许可：FP16 一律走 A16（A4/A8 尚无 FP16 形式）。
- `tests/ops/linear/test_nvfp4_f16.cpp`：新增 t = 1 / 6 / 32 / 40 / 127 / 4096，
  相对 L2 均 < 3e-4；同时修了 t=1 时采样列越界的测试 bug。

验证：`ninfer_linear_nvfp4_f16_test` 通过；nvfp4/linear_add/linear_pair/linear_swiglu/
fp8 相关测试全部通过。

### 4.4 FP16 W8 / Q4 linear（MTP 投影与 draft head）

- 新增 `src/ops/common/activation.cuh`（`ActivationTraits<T>`，GQA 与 linear 共用）；
  `gqa_activation.cuh` 改为别名。
- W8：`w8_rowsplit_output.cuh` 的 tile/contiguous output 模板化；
  `w8_rowsplit_gemm_simt.cuh/.cu` 增加 `Act`（Volta 上 `launch_w8_small_t` 实际走 SIMT）；
  新增 `launch_w8_simt_r8_c4_f16/_c8_f16` 与 `w8_f16_dispatch`。
- Q4：`q4_rowsplit_gemm_simt.cuh/.cu` 增加 `Act`（含 Volta 的 shared stage）；
  新增 `launch_q4_simt_r8_c4_f16/_c8_f16` 与 `q4_f16_dispatch`（draft head）。
- `linear.cpp`：FP16 + W8/Q4 分派到对应 f16 入口。
- 新测试：
  - `test_w8_f16.cpp`：MTP 四个投影形状，T=1/4/8，rel L2 < 3.3e-4；
  - `test_q4_f16.cpp`：draft head 131072x5120，T=1/4/8，rel L2 < 3.0e-4。
- 回归：W8/Q4 的 BF16 linear、swiglu、attn_input_proj、GDN input/snapshot/record 全过。

### 4.3 FP16 FP8 linear（output head / vocabulary）

- `src/ops/linear/fp8/fp8_output.cuh`：新增 `Fp8Fp16ContiguousOutput`；
- `src/ops/linear/fp8/fp8_volta_qpn_gemm.cuh`：QPN kernel 与 shared launcher 增加 `Act` 模板参数
  （`Fp8QpnActivation<T>` 负责 bf16/half → half 的 A 操作数）；
- `src/ops/linear/fp8/fp8_volta_qpn_gemm.cu`：新增 `launch_fp8_volta_qpn_f16`；
- `src/ops/linear/fp8/fp8_dispatch.cpp`：新增 `fp8_f16_dispatch`，按 32 行切 QPN，覆盖 T=1…宽 prefill；
  T=1 也用 QPN（BF16 下 t=1 的 SIMT GEMV 略快，这里为保持 dtype 边界单一而接受几个百分点）；
- `src/ops/linear/linear.cpp`：FP16 + FP8 权重分派到 `fp8_f16_dispatch`，并放宽
  `validate_linear_semantics`。
- 新增 `tests/ops/linear/test_fp8_f16.cpp`：14336x5120 T=1/8/33；248320x5120 T=1/9/128，
  相对 L2 均 < 3.1e-4。

注意：FP8 的 CUTLASS sm70 路线仍是 BF16；FP16 的 FP8 走 QPN 切块（正确，prefill 后续可优化）。

---

## 5. 仍缺少的 FP16 ops（按对可运行性的重要性排序）

1. ~~FP8 linear 的 FP16~~ 已完成（见 4.3）。embedding 的 FP16 输出已支持（仅 FP8 table）。
   FP8 的 CUTLASS prefill 仍是 BF16，FP16 走 QPN 切块。
2. **NVFP4 融合叶子**：`attn_input_proj`、`gdn_input_proj`（含 conv snapshot/record）、
   `linear_add`、`linear_swiglu`、`linear_pair`。
   **可选策略**：FP16 首版在 `Variant` 里把融合叶子拆成 `ops::linear` + 已支持 FP16 的
   elementwise（`silu_mul`/`residual_add`），先保证正确，再加融合 kernel 的 FP16 输出。
   注意 `split_launch` / column_parallel / row_parallel 也要一起处理。
3. **MTP**：W8 plain linear 与 Q4 draft head 的 FP16 已完成（见 4.4）。
   仍缺：
   - `linear_pair`（MTP k/v）：**计划在 `Variant` 层拆成两次 `ops::linear`**，不新增 pair FP16 kernel；
   - `mtp_split_attn_in` / `mtp_pack_fc_input` 的 FP16 已完成（kernel + wrapper + 测试）；
   - MTP 的 sampling/draft-token 产出边界需要确认 FP16→logits 无层间 cast。
4. **GDN 剩余叶子**：norm+gating 融合、conv snapshot/record、recurrent 的 FP16 入口已在
   `gdn_gating_proj_f16` / `causal_conv1d_f16` / `gdn_f16`，但融合 wrapper 仍只接受 BF16。
5. **runtime 接线**：已完成基础层，剩下 leaf / schedule：
   - 已加 `src/targets/qwen3_6/impl/runtime/text_dtype.h` + CMake `NINFER_TEXT_DTYPE=bf16|f16`；
   - 输出头边界已定：FP8 output head 与 Q4 draft head 支持 FP16 activation → **BF16 logits**
     （sampling 契约要求 BF16），因此 `ops::linear` 现允许 FP16-in/BF16-out 的 head 调用；
   - `TextConfig`/`DFlashConfig` 已有 `activation_dtype`；
   - `workspace_recipe.h` 全部文本 workspace 已改用 `Config::activation_dtype`；
   - **仍待**：`layouts_impl.h`（41 处，尺寸同为 2 字节，语义上仍应对齐）、
     `text_context_impl.h`（133 处，实际 Tensor dtype 与 op 调用）、
     `variant.cpp`（15 处，融合叶子/MTP workspace/`linear_pair` 拆分）。
6. **sampling / logits**：目前 logits 是 FP32，argmax/sampling 不需要 FP16；但要确认
   output head 的 FP16 输出到 logits 的边界。

---

## 6. 建议的下一步顺序

1. FP8 linear FP16（output head 是每 token 必经路径）。
2. NVFP4 融合叶子：先按“拆成 linear + elementwise”接入，保证端到端正确；
   再以基准数据决定是否给融合 kernel 加 FP16 输出（性能优化）。
3. W8/Q4 linear + `linear_pair` 的 FP16（MTP）。
4. runtime 编译期 dtype 接线，先 workspace_recipe/layouts，再 text_context/Variant。
5. 端到端：BF16 smoke → FP16 smoke → 与 BF16 同 prompt 数值/输出对比 → BFCL 200。
6. 只有端到端证据支持时才声称完成；失败明确记为负结果。

---

## 7. 已知坑

- conv state 形状是 `[3,C,S]`；kernel 按 `[3,C]` 访问，不要改回 `[C,3]`。
- `gdn_input_proj` 的 snapshot/record 形状必须与 state pool 一致。
- `gqa_attention_workspace_capacity_bytes` 仍按 BF16 cache 预计算；VoltaFlash staging 在
  FP16 下会多算但无害（当前 FP16 走 ChunkedSmallT）。
- 非 Volta 构建里 `gqa_attention_prompt_attention_launch` 的 BF16 分支会用 BF16 cache；
  FP16 activation 只配 I8/I4NL cache，wrapper 已拒绝其他组合。
- `KvCacheStorage::Float16` 枚举存在但未暴露 CLI、未接模型。
- `ops::linear_add_row_parallel` 尚无 FP16；全 FP16 时先拆开或补 split 形式。
- 当前 2×32GB 目标不要用 4×16GB legacy 的 180000 capacity 做验收。
- 构建 CPU 并发不要超过用户约 85% 上限。
- **不要在用户未明确要求时 commit。**
- `tests/ops/linear/test_nvfp4_f16.cpp` 的采样列在 t=1 时要用 `t>1?1:0`，否则越界。

---

## 8. 可直接粘贴给新会话的提示词

```text
你接手 /home/cmj/ninfer-v100-fast，先读 NEXT_SESSION_HANDOFF.md、AGENTS.md。
不要读取旧对话，不要重复已做过的 BFCL 诊断。

当前事实：
- 分支 fp16-text-runtime，HEAD 4e07a7dd；本会话的 FP16 attention/append 与 NVFP4 全 T
  改动仍在工作区，未 commit。
- 已原生支持：FP16 activation + INT8 group-64 KV 的 GQA append/attention；FP16 NVFP4
  linear 在 T=1…4096。
- 未支持：FP8 output head 的 FP16、W8/Q4 MTP、NVFP4 融合叶子、GDN 融合叶子、runtime
  编译期 dtype 接线。
- V100 的 BF16 kernel 内部已经用 FP16 MMA；FP16 的收益是精度，不是 MMA 速度。
- 禁止层间 FP16→BF16→FP16 边界转换。
- 目标硬件：2×V100-SXM2-32GB，NVFP4 artifact，262144 context，INT8 group-64 KV，
  MTP-3，eager decode；不要用 4×16GB legacy 验收。
- 不要用卡片提问；自行选择技术上最优的方案，直接执行；不要主动 commit。

任务：
继续完成全 FP16 文本 runtime。按以下顺序推进，每步都用独立 oracle/BF16 对比：
1. FP8 linear 的 FP16（含 vocabulary head）。
2. NVFP4 融合叶子的 FP16（可先拆成 linear + elementwise 保正确）。
3. W8/Q4 linear + linear_pair 的 FP16（MTP）。
4. TextConfig/workspace_recipe/layouts/text_context/Variant 的编译期 dtype 接线。
5. BF16 smoke、FP16 smoke、BF16 同 prompt 对比、BFCL 200。

先报告已确认状态，然后开始改代码。
```
