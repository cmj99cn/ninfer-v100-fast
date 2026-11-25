# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The first public release was `v0.1.0` (2026-10-27). `v0.2.0` (2026-11-07) adds
the FP16 text runtime, closes the BFCL multi-turn quality gap, and removes the
FP16 path's decode/prefill regression. `v0.2.1` (2026-11-09) fixes Vision under
the FP16 text runtime. `v0.2.2` (2026-11-11) fixes TP peer-access setup for
device lists that do not start at device 0. `v0.2.3` (2026-11-12) fixes
multi-instance routing against a warm home and surfaces host-cache diagnostics.
`v0.2.4` (2026-11-13) removes the multi-instance admission stall behind a
decoding instance. `v0.2.5` (2026-11-13) reports prefix-cache reuse in every
serving usage payload and fixes the Vision MTP bridge under the FP16 text
runtime.

## [Unreleased] - feat/qwen3-8-flash-next（开发中）

分支开发条目，基于 v0.2.5；未 bump 版本号、不是 release。Flash-Next 是开发中 target。
MTP 现已快过 base **1.56×**（70.54 → 110.3–110.5 tok/s）；120 tok/s 目标仍未达成（非 2×），
起点 30.33 tok/s（fold commit 接线前）到本轮 110，约 3.6×。

### 本轮 MTP 接线与优化（2026-11-25，commit `b0b79c4b`）

新增条目：`perf(flash-next): wire MTP fold commit and optimize verify with batched small-T kernels`。

- GDN replay fold commit 转为生产路径（Tasks 1–10，bit-exact 验证通过）。
- verify 多列权重复用 kernel（hc_mix / linear / W8 / route / conv / gdn_proj / GDN stage+extract）。
- draft TP2 角色拆分；sink 流同步。
- Q4 proposal/head 设为默认（artifact 自带同源权重）；`NINFER_DIAG_FP16_DRAFT_HEAD=1` 保留作对照。
- draft 置信度 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8` 提前终止。
- 用户裁决：接受 Q4 为默认——draft 是提案、正确性由 verify 保证；chain oracle 由
  `{220,27967,813}` 改为 `{220,27967,53}`（Q4 语义），greedy 等价（MTP on/off）逐 token 一致。

### Added（早期基础，commit `91b45ae5`）

- 新增 Flash-Next target（开发中）。`perf(flash-next): optimize MTP with quantized draft, D2D
  snapshot and expert profile`。
- draft bank FP8 量化常驻，占用从 5.03GB 降到 2.52GB。
- 专家画像 `--expert-cache-profile` 接入 bench（画像由 Strata 同模型 `expert-profile.bin` 转换）。
- MoE/shared T≤4 oracle 测试。

### Changed

- MTP 状态事务从 pageable host 往返改为 device-to-device 异步快照。
- cached routed/shared MoE 与 FP16 linear 扩展到 T≤4；新增 GDN qkvz T≤4 的 bit-exact
  decode kernel；注册 36 层 GDN replay fold 几何（本轮已转为生产路径）。

### Performance

评测条件：2×V100 TP2，cache 15000 + expert profile 0 miss，128-token、2 reps、greedy；
最终配置 = Q4 proposal/head 默认 + draft=3 + `NINFER_FLASH_NEXT_DRAFT_MIN_P=0.8`。

- base 无 spec：70.54 tok/s；起点（fold commit 接线前）：30.33 tok/s。
- **最终配置：110.3–110.5 tok/s**（3 次实测 110.39 / 110.53 / 110.28），acceptance
  **0.860–0.861**，**1.56× base**，相对起点约 **3.6×**。
- 历史值（已被取代，保留记录）：draft=1 51.78 tok/s / 0.881（当时落后 base 约 27%）、
  draft=3 42.9 tok/s / 0.772（min_p=0）、batch verify 27.0 tok/s / 0.56（已回退）、
  无 profile 13.6–14 tok/s。
- 120 tok/s 目标仍未达成（1.56×，非 2×）；「MTP 落后 base / 负收益」的旧结论已失效。
- 口径澄清：120 / 99% 来自随机采样饱和口径，greedy 下不可达；上述 1.56× 为 greedy
  committed tok/s，不与随机采样 saturation 数字混用。

### Known limitations

- Flash-Next 与 MTP 优化均为开发中，未作为新版本发布；120 tok/s 目标仍未达成。
- 并行架构未改：主模型保持 TP2。1cat-vllm（TP2）与 Strata（层分 PP）的 MTP 均 >100 tok/s，
  本分支现已达到 ~110 tok/s，与 Strata 同档。
- CUDA Graph 负收益结论保留，根因已定位：sm_70 上 `cudaGraphLaunch` 的 O(N) 逐节点提交成本
  （3,054 节点 → ~0.6 ms/round）超过可削的、已被设备执行掩盖的 host 间隙；Strata 同为 V100
  能用 graph 靠 kernel 内 t 维权复用（eager 侧属性），该收益我们已在 eager 侧领取。
- 负结果历史保留：batch verify `decode_span` 数值不等价（PLE/GDN 状态漂移，已回退）；
  层分 / 去 allreduce、staged routed GEMV、FP8 MTP 等已回退。GDN replay fold 此前
  11.36 tok/s 的回退记录已过时——fold commit 现已是生产路径，bit-exact 验证通过。

### Verification

- 构建 OK，`git diff --check` 干净；mtp_verify / fold_commit / mtp_state / gdn_replay_storage
  通过；draft / chain / confidence / greedy_equivalence 因缺隐藏 fixture `NINFER_MTP_HIDDEN`
  SKIP（内核 oracle 仍运行）。
- 历史：38/38 测试通过；NLL greedy 11751。

## [0.2.5] - 2026-11-13

### Added

- Chat Completions and Anthropic Messages report prefix-cache reuse. OpenAI responses
  (non-streaming, tool-call, and the streaming usage chunk) carry
  `usage.prompt_tokens_details.cached_tokens`; Anthropic messages carry
  `cache_read_input_tokens` and `cache_creation_input_tokens` (`message_start` reports zero cache
  counters and the final `message_delta` carries the real read count). Responses already reported
  `input_tokens_details.cached_tokens`.

### Fixed

- The Vision MTP bridge crashed under the FP16 text runtime with
  `rmsnorm: tensors must share BF16 ... [x=0 w=0 out=6]`. The bridge handed the Vision session's
  BF16 embeddings to an MTP stem whose workspace is allocated at the activation dtype. Both the
  tensor-parallel and single-device bridges now cast to the activation dtype first, the stem
  validators accept the activation dtype, and the multimodal prefill MTP path allocates at the
  activation dtype and casts the visual scatter source when it differs.

### Docs

- Record the third-party 8×V100-SXM2-32GB deployment validation: a 4-card 2×TP2 pool passed
  end-to-end (Vision, MTP, prefix reuse, OpenAI/Anthropic serving, 8 concurrent requests), and the
  same host ran two independent 4-card pools side by side. Single-pool 8-card profiles remain
  unverified community paths.

## [0.2.4] - 2026-11-13

### Fixed

- Multi-instance admission stalled behind a decoding instance. `EnginePool::admit`
  probed every instance while holding the pool lock, and that probe waited on the
  executor's execution lock, which the worker holds across a decode/prefill unit.
  New requests waited for the instance to go fully idle, and `/metrics` scrapes
  serialized behind the same pool lock. The router now reads each candidate's
  prefix probe and capacity with one non-blocking `try_lock`; an instance that is
  mid-unit reports no readable view and is skipped for that dispatch round, so the
  request waits in the global pending queue instead of inside `admit()`. The
  aged-head and home-affinity rules apply to the readable candidates; if a queue
  head's only fitting instance is busy it stays pending and a newer request may be
  dispatched first. Admission remains the only queue: no request is preempted and
  the permit still reserves the candidate's net page bill.

## [0.2.3] - 2026-11-12

### Fixed

- Multi-instance admission billed every request at its reuse-blind gross page
  count (`ceil((prompt + max + margin) / 64)`), so a warm home whose free pages
  covered only the net bill was filtered out and a cold instance could evict it.
  The router now derives the per-candidate net requirement (gross minus the
  candidate's device-resident prefix; a host-manifest restore still allocates
  device pages and does not reduce it), and the permit reserves exactly those
  net pages instead of the gross upper bound.

### Added

- `--host-cache-mib` capacity documentation: the 80/20 block/snapshot budget
  split, the per-slab sizing formula (`tp × per-rank page bytes`), and worked
  examples for the TP2 NVFP4 profile (`1024` MiB ≈ 373 block slabs ≈ 24K host
  prefix tokens; `12288` MiB ≈ 4485 slabs ≈ 287K).
- Host-tier allocation-failure counters (`block_alloc_failures` /
  `snapshot_alloc_failures` in `HostCacheMetrics`, exposed as Prometheus
  `ninfer_prefix_cache_host_alloc_failures_total{kind="block|snapshot"}` and
  JSON fields). Only caller-visible failures count; an allocation recovered by
  evicting an older slab or manifest does not. Sustained growth means the host
  budget is smaller than the working set and write-throughs are being skipped.

## [0.2.2] - 2026-11-11

### Fixed

- TP mailbox P2P enablement used the rank index as the CUDA device ordinal, so a
  device list that does not start at 0 (for example `--tp 2 --devices 4,5`)
  enabled peer access against unrelated GPUs, never enabled the real pair, and
  then aborted in the mailbox self-check with `cudaErrorIllegalAddress`. Peer
  access now uses each rank's real device ordinal, restoring NVLink mailbox
  selection on non-zero device lists.

## [0.2.1] - 2026-11-09

### Fixed

- Vision requests under the FP16 text runtime no longer fail with
  `scatter: src/dst must share BF16 or FP16 dtype`. The Vision tower emits BF16
  while the text residual is FP16, so both prefill paths now stage the tower
  output through an activation-dtype buffer with a BF16->FP16 cast before
  scattering into the residual. Covers TP1, TP2, and multi-instance (2xTP1)
  serving.

## [0.2.0] - 2026-11-07

### Added

- FP16 text-activation runtime for the NVFP4 text stream, selected at configure
  time with `NINFER_TEXT_DTYPE=f16`: FP16 hidden/residual/activation
  workspaces plus FP16 embedding, RMSNorm, RoPE, GDN recurrent/conv/gating,
  attention staging, and the FP8/Q4/W8 control heads.
- FP16 wide-T CUTLASS routes for the fused GDN input projection and the GDN
  conv snapshot/record split projection.
- FP16 activation support on the Volta flash (split-D) prefill attention route:
  the Q stage, out unstage, KV append (i8/i4nl/bf16 cache), and the vendored
  f32 Q/out conversions.

### Changed

- The 2x32GB release profile is MTP-3 (`--spec mtp --draft-tokens 3`) on the
  FP16-embedding NVFP4 artifact at 262144-token capacity, INT8 group-64 KV,
  eager decode; no-spec is retained as a fallback.
- The README now reports v0.2.0 speed/quality and documents the no-spec
  44.0% -> 43.0% four-sample flip as an optimizable evaluation point.

### Fixed

- Replaced the FP16 GDN gating projection's naive per-(head,token) kernel
  (serial K, uncoalesced weight reads) with a CTA-per-head, shared-weight,
  warp-per-token reduction.
- The FP16 fused GDN input projection no longer chunks QPN at prefill width;
  it uses the FP16 CUTLASS wide-T route for T >= 33.
- FP16 prefill attention no longer falls back to the chunked small-T path; it
  runs the Volta flash split-D kernel used by the BF16 build.
- BFCL multi-turn quality: no-spec 43.0% (86/200) and MTP-3 43.5% (87/200), up
  from the 36.0% pre-fix baseline. The NVFP4 QPN divisor, tool-call schema
  typing, and the prefix-reuse rewrite checkpoint are fixed, and the rewrite
  checkpoint is disabled for no-spec and MTP.

### Performance

- MTP-3 decode (262144 context, INT8 KV, f16embed artifact): 104.7 / 98.9 /
  109.6 / 98.1 tok/s at 502 / 3.7K / 15K / 59K occupied context; the standard
  artifact scores 103.1 / 102.4 / 112.6 / 97.7.
- Cold prefill (2048-token CLI): 1826 tok/s, matching the BF16 baseline of 1825
  (1522 before the FP16 fix).

## [0.1.0] - 2026-10-27

### Added

- First public release of the V100-focused NInfer inference engine.
- Single-process TP2 execution on 2×Tesla V100-SXM2-32GB with NVLink; TP4 remains a
  4×16GB legacy regression path.
- Split-D long-context prefill kernel and cuBLAS prefill GEMM route.
- MTP speculative decoding with up to three qualified drafts on V100X2.
- Content-addressed multi-turn prefix cache, including shared host tier and
  cross-instance manifest reuse for 2×TP1.
- Multi-instance serving documentation, including community/out-of-contract
  8-GPU profiles for 8×TP1, 4×TP2, and 2×TP4, plus 4-card 4×TP1/2×TP2. Only
  2×TP1 has passed a real integration test; community validation is requested
  for the rest.
- OpenAI-compatible and Anthropic-compatible serving surfaces, CLI, and metrics.
- INT8 group-64 KV cache, NVFP4 text execution, and the self-converted
  Qwen3.8-27B NVFP4 `.ninfer` artifact route.
- Release and reproduction documentation for the unresolved BFCL quality gap.

### Changed

- The default model path in the documentation is the self-converted NVFP4
  artifact on 2×V100-SXM2-32GB; the 4×16GB/Q4_K_M path is retained as a legacy
  regression target, not the current release profile.
- SM70 test coverage now gates NVFP4 A4 and FP8 A8 routes by compute capability
  instead of executing unsupported kernels.
- Q4/Q5 split-storage `gdn_input_proj_column_parallel` requires an explicit
  workspace; production and split tests use the same contract.

### Fixed

- cuBLAS BF16 GEMM output no longer stages through FP16; the route now uses
  FP32 staging, eliminating overflow on real activation magnitudes.
- Zero-capacity `DeviceArena` construction in split tests.
- Output-head embedding device selection after staging retirement.
- Volta FP8 workspace-interval assertions and Q4/Q5 grouped-MMA capability
  gating.
- Mixed-batch MTP identity test is pinned to the CUTLASS prefill route because
  cuBLAS reduction order depends on batch shape.
- Full SM70 test suite passes: **135/135**, with 15 hardware/artifact-gated
  skips.

### Known Issues

- **BFCL-v3 multi-turn tool calling is resolved for the current release profile.**
  Latest full 200: **82/200 = 41.0%** vs 1cat-vLLM **73/200 = 36.5%** (paired
  net +9). The fixes are tool-call schema typing / Python-literal normalization,
  the NVFP4 QPN divisor, and disabling decode-state turn-checkpoint reuse on the
  non-speculative prefix path. See the README quality section.
- The current self-converted NVFP4 artifact includes Vision; the GGUF Q4_K_M
  artifact rejects Vision. Multi-instance Vision is unverified.
- Long-duration prefix-cache soak testing under very high concurrency has not
  been completed.
- Only 2×TP1 multi-instance has passed a real integration test. The 4-card
  (4×TP1/2×TP2) and 8-card (8×TP1/4×TP2/2×TP4) profiles are unverified; the
  8-card profiles are community/out-of-contract. No throughput or memory claims
  are made for them. 4×TP1 and 8×TP1 require 32GB cards.
- KV caching has two tiers only: per-instance device KV and the process-shared
  pinned host prefix-KV tier. There is no disk KV tier, KV offload, or global
  active-KV pool.

### Verification

- `ctest --test-dir build --output-on-failure -j 1`: **135/135 passed**,
  15 skipped, 1616 s, with `NINFER_VOLTA_SPLITD=1`,
  `NINFER_SM70_PREFILL_GEMM=1`, and the documented model/artifact environment.
- Full BFCL-v3 `multi_turn_base` 200-sample run: 54/200 = 27.0%; 2404 requests,
  5959 s; average latency 2.471 s, TTFT 804.9 ms, TPOT 13.8 ms,
  38.91 tok/s.
- Quick before/after quality check: cold-start text, 8K prefix reuse, and MTP
  token IDs matched; no regression observed.
- 8-GPU profile script: `bash -n`, `py_compile`, and `--dry-run` for 8xTP1,
  4xTP2, and 2xTP4 passed. These are command-construction checks only; no
  4-card or 8-card execution was performed.

[0.1.0]: https://github.com/cmj99cn/ninfer-v100-fast/releases/tag/v0.1.0
