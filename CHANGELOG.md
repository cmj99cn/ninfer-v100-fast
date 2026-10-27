# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The first public release is `v0.1.0`: the engine is usable on the documented
V100 profiles, but the BFCL quality gap and the limitations below are still open.

## [0.1.0] - 2026-10-27

### Added

- First public release of the V100-focused NInfer inference engine.
- Single-process TP2/TP4 execution on 2×Tesla V100-SXM2 with NVLink.
- Split-D long-context prefill kernel and cuBLAS prefill GEMM route.
- MTP speculative decoding with up to three drafts.
- Content-addressed multi-turn prefix cache, including shared host tier and
  cross-instance manifest reuse for 2×TP1.
- Multi-instance serving documentation and 8-GPU profiles for 8×TP1, 4×TP2,
  and 2×TP4, plus 4-card 4×TP1/2×TP2. Only 2×TP1 has passed a real integration
  test; community validation is requested for the rest.
- OpenAI-compatible and Anthropic-compatible serving surfaces, CLI, and metrics.
- INT8 group-64 KV cache, NVFP4 text execution, and the self-converted
  Qwen3.8-27B NVFP4 `.ninfer` artifact route.
- Release and reproduction documentation for the unresolved BFCL quality gap.

### Changed

- The default model path in the documentation is the self-converted NVFP4
  artifact; the old Q4_K_M/V100X2 assumptions are not used by this release.
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

- **BFCL-v3 multi-turn tool calling remains below the vLLM baseline.**
  Latest full 200: **54/200 = 27.0%** vs 1cat-vLLM **77/200 = 38.5%**.
  This is the highest ninfer full-200 result so far (previous best 25.0%),
  but the **11.5pp** gap is unresolved. A single seed cannot support 6pp
  attribution. Active blind fixing is paused; community help is requested.
  See [`BFCL_QUALITY_ISSUE.md`](BFCL_QUALITY_ISSUE.md) and
  [`BFCL_REPRODUCTION.md`](BFCL_REPRODUCTION.md).
- Multimodal is unavailable on the current artifact; text-only is supported.
- Long-duration prefix-cache soak testing under very high concurrency has not
  been completed.
- Only 2×TP1 multi-instance has passed a real integration test. The 4-card
  (4×TP1/2×TP2) and 8-card (8×TP1/4×TP2/2×TP4) profiles are unverified; no
  throughput or memory claims are made for them. 8×TP1 requires 32GB cards.
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
