#pragma once

#include "artifact/reader.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

inline constexpr std::uint32_t kPleHidden        = 2560;
inline constexpr std::uint32_t kPleStreams       = 4;
inline constexpr std::uint32_t kPleChannels      = 10240;
inline constexpr std::uint32_t kPleFeatures      = 160;
inline constexpr std::uint32_t kPleHeads         = 16;
inline constexpr std::uint32_t kPleShards        = 128;
inline constexpr std::uint32_t kPleShardRows     = 2500012;
inline constexpr std::uint32_t kPleConvWindow    = 9;
inline constexpr std::uint32_t kPleConvTaps      = 4;
inline constexpr std::uint32_t kPleConvDilation  = 3;
inline constexpr std::size_t kPleHashControlsBytes = 280;
inline constexpr float kPleGateFloor             = 1.0e-6F;

/**
 * The PLE n-gram hash controls of the artifact.
 *
 * The 280-byte `resource/text/ple_hash_controls` payload is 35 little-endian U64 words:
 * multipliers [0, 3), vocab_sizes [3, 19) and offsets [19, 35). The payload is the only source
 * of these values; nothing in this repository restates them.
 */
struct PleHashControls {
    std::array<std::uint64_t, 3> multipliers;
    std::array<std::uint64_t, 16> vocab_sizes;
    std::array<std::uint64_t, 16> offsets;
};

/**
 * Op: exact decode of the hash controls payload.
 *
 * Math / indexing: word i of the payload is the little-endian U64 at byte offset 8 * i;
 * words [0, 3) are the multipliers, words [3, 19) the vocabulary sizes and words [19, 35) the
 * offsets.
 *
 * Supported domain: a payload of exactly `kPleHashControlsBytes` bytes. Any other length throws.
 *
 * Numeric: exact byte transform, no rounding.
 *
 * Effects: returns a value; mutates nothing.
 *
 * Workspace: none.
 *
 * Execution: host side.
 */
[[nodiscard]] PleHashControls parse_ple_hash_controls(std::span<const std::byte> payload);

/**
 * Op: the n-gram row addresses of a token sequence.
 *
 * Math / indexing: for token i with the two preceding ids `tail = {older, previous}` (oldest
 * first) and the end-of-sequence id `eos`:
 *
 *   older  = eos when tail[1] == eos else tail[0]
 *   pair   = U64(x) * multipliers[0] XOR U64(tail[1]) * multipliers[1]
 *   triple = pair XOR U64(older) * multipliers[2]
 *
 *   for head in [0, 16):
 *     word  = int64_bits(pair) for head < 8, else int64_bits(triple)
 *     index = positive modulo of word by vocab_sizes[head] (a negative remainder adds the
 *             vocabulary size)
 *     rows[head + 16 * i] = index + offsets[head]
 *
 * The tail advances to `{tail[1], x}` after every token, so one token addresses 16 rows. PLE
 * consumes the raw token ids: vision features never replace the ids that feed the hash.
 *
 * Logical shapes:
 *   T  token extent, positive
 *   ids   U64 [T]      host, contiguous
 *   tail  U64 [2]      host, oldest first
 *   rows  U32 [16 * T] host, contiguous, head order within each token
 *
 * Supported domain: the registered controls above. `rows` must hold 16 * T elements.
 *
 * Numeric: exact 64-bit integer transform. The modulo is the positive modulo of the signed word;
 * a negative word is not converted to an unsigned value before the remainder.
 *
 * Effects: writes all of rows. `rows` is disjoint from `ids`. `ids`, `tail` and `controls` are
 * read-only.
 *
 * Workspace: none.
 *
 * Execution: host side, no device work and no persistent state. The caller advances its own tail
 * between calls.
 */
void ple_hash_tokens(std::span<const std::uint64_t> ids, std::array<std::uint64_t, 2> tail,
                     std::uint64_t eos, const PleHashControls& controls,
                     std::span<std::uint32_t> rows);

/**
 * The paged NF4 PLE table.
 *
 * `geometry` is the whole-shard `row_paged_g16_4k_v1` geometry of `[2500012, 160]`, and
 * `divisors` is the device table of the 128 per-shard FP32 weight divisors, in shard order. The
 * table is not resident as a whole: the caller keeps it on disk and stages only what a call
 * addresses.
 */
struct PleNf4Table {
    ninfer::artifact::RowPagedNf4Geometry geometry;
    std::uint32_t shard_count;
    const float* divisors;
};

/**
 * The shard and the row inside that shard that carry one global PLE row.
 *
 * Math / indexing:
 *   shard = global_row / shard_rows
 *   row   = global_row % shard_rows
 *
 * The hash partitions the global row space among the heads by `offsets`, and the shards hold the
 * global rows in order, so this is the assignment the lookup must use. It is a derivation from
 * the ordered shard list and the hash controls, not an artifact fact: the qualification test pins
 * it against the conversion report's `shard_i` sources and the per-shard objects, and covers row
 * numbers that cross a shard boundary.
 *
 * Supported domain: `global_row` inside `[0, shard_count * shard_rows)`. A row outside the table
 * has no address and throws; the lookup still answers such a row with exact `+0` words.
 *
 * Numeric: exact integer division.
 *
 * Effects: returns a value; mutates nothing.
 *
 * Workspace: none.
 *
 * Execution: host side.
 */
struct PleRowAddress {
    std::uint32_t shard;
    std::uint32_t row;
};

[[nodiscard]] PleRowAddress ple_row_address(const PleNf4Table& table, std::uint32_t global_row);

/**
 * Op: decode of the addressed PLE rows into the concatenated n-gram embedding.
 *
 * Math / indexing: entry `i = head + 16 * (t + T * b)` of `rows` is the global row that head
 * `head` addresses for the live column `(t, b)`. For that entry, with the address
 * `(shard, row) = ple_row_address(table, rows[i])`:
 *
 *   code[d]        = nibble of `staged_rows[i]` holding element d, low nibble first
 *   group_scale[g] = exact E4M3FN value of the scale byte of group g = d / 16
 *   embedding[160 * head + d, t, b] = round_bf16(code_value[d] * group_scale[g] / divisor[shard])
 *
 * The codebook is the registered E2M1 codebook shared with the NVFP4 path, and the group scale is
 * the exact stored E4M3FN byte. `staged_rows[i]` must be the `row_bytes` bytes of `rows[i]` at
 * `row_paged_nf4_row_offset(geometry, row)` inside the payload of shard `shard`; that fetch
 * belongs to the caller, because the table stays on disk. A row outside the table's row space
 * contributes exact `+0` words.
 *
 * Logical shapes:
 *   rows          U32 [16 * T * B] device, contiguous, entry order (t, b, head)
 *   staged_rows   bytes [16 * T * B * row_bytes] device, one encoded row per entry, same order
 *   embedding     BF16 [160 * 16, T, B] device, contiguous, dimension 0 fastest
 *   Element (f, t, b) of the embedding is stored at f + 2560 * (t + T * b); head `head` owns
 *   features `[160 * head, 160 * head + 160)`.
 *
 * Supported domain: the registered `[2500012, 160]` geometry and 128 shards. The device route is
 * the registered route: the table is never resident as a whole, only the addressed row bytes and
 * the 128 divisor words are. The caller stages the host rows produced by `ple_hash_tokens` with
 * the staged block, and `rows`, `staged_rows` and the embedding bases must be 16-byte aligned with
 * a positive entry count.
 *
 * Numeric: the codebook value times the exact stored group scale is exactly representable in
 * BF16, so the codebook exit is an exact rounding boundary and signed zero is preserved. The
 * per-shard divisor is the only rounding point of the Op. The oracle decodes each stored code
 * with the exact stored scale and divisor in FP64; the output is exact when the FP32 division of
 * that product is exact, and is otherwise within one BF16 ULP of the oracle.
 *
 * Effects: writes all of embedding. `table`, `rows` and `staged_rows` are read-only.
 *
 * Workspace: none.
 *
 * Execution: enqueues on the given stream, allocates no device memory, keeps no persistent state
 * and supports CUDA Graph capture and replay at bound addresses.
 */
void ple_lookup_rows(const PleNf4Table& table, const std::uint32_t* rows, std::size_t entries,
                     const std::uint8_t* staged_rows, Tensor& embedding, cudaStream_t stream);

/**
 * Op: the PLE gate and the convolution norm it feeds.
 *
 * Math / indexing: for each live column `(t, b)` with `0 <= t < n[b]` and each stream `s`, with
 * `eps = epsilon` and `H = 2560`:
 *
 *   k_s[d] = Norm(projected.key[d + H * s, t, b], norm_key[d + H * s], eps)
 *   q_s[d] = Norm(residual[d, s, t, b], norm_query[d + H * s], eps)
 *   a_s    = sum_d k_s[d] * q_s[d] / sqrt(H)
 *   v_s    = sign(a_s) * sqrt(max(|a_s|, 1e-6))
 *   g_s    = 1 / (1 + exp(-v_s))
 *   gated_value[d + H * s, t, b]    = round_bf16(g_s * projected.value[d, t, b])
 *   normalized_conv[d + H * s, t, b] = Norm(stored gated_value, norm_conv[d + H * s], eps)
 *
 * `sign(0) = 0`, so the gate is 0.5 at zero, and the `1e-6` floor is independent of `eps`. The
 * convolution norm consumes the stored BF16 words of `gated_value`, which is an observable output
 * of this Op. Every gain `1 + w` is evaluated and applied in FP32 and is never rounded to BF16
 * before the multiplication.
 *
 * Logical shapes:
 *   H  hidden width, 2560
 *   S  stream count, 4
 *   C  channels, H * S = 10240
 *   T  token extent, positive
 *   B  batch rows, positive
 *   projected.key    BF16 [C, T, B]  contiguous, dimension 0 fastest
 *   projected.value  BF16 [H, T, B]  contiguous, dimension 0 fastest, shared by every stream
 *   residual         BF16 [H, S, T, B] contiguous, dimension 0 fastest
 *   norm_key         BF16 [C]         contiguous, stream-major
 *   norm_query       BF16 [C]         contiguous, stream-major
 *   norm_conv        BF16 [C]         contiguous, stream-major
 *   gated_value      BF16 [C, T, B]   contiguous, dimension 0 fastest
 *   normalized_conv  BF16 [C, T, B]   contiguous, dimension 0 fastest
 *   Element (d, s, t, b) of a four-dimensional operand is stored at
 *   d + H * s + H * S * (t + T * b); element (f, t, b) of a two-dimensional operand is stored at
 *   f + C * (t + T * b).
 *
 * Supported domain: the registered geometry above, 16-byte aligned bases, epsilon positive and
 * finite.
 *
 * Numeric: the statistics and the dot product are accumulated in FP32; that accumulator precision
 * is a required implementation profile. The oracle evaluates the complete formula in FP64 from
 * the represented inputs and reproduces the BF16 storage rounding of `gated_value`, which is an
 * observable output; both outputs are promoted and compared against that result, so their own
 * storage rounding belongs to the numerical criterion and not to the oracle.
 *
 * Effects: writes all of `gated_value` and all of `normalized_conv`. The two outputs are mutually
 * disjoint and disjoint from every input. Read-only inputs may share backing and are preserved.
 *
 * Workspace: none.
 *
 * Execution: enqueues on the given stream, allocates no device memory, keeps no persistent state
 * and supports CUDA Graph capture and replay at bound addresses.
 */
void ple_gate(const Tensor& projected_key, const Tensor& projected_value, const Tensor& residual,
              const Tensor& norm_key, const Tensor& norm_query, const Tensor& norm_conv,
              float epsilon, Tensor& gated_value, Tensor& normalized_conv, cudaStream_t stream);

/**
 * Op: the dilated PLE convolution, its SiLU, and the residual it updates.
 *
 * Math / indexing: for row `b`, live token `t` with `0 <= t < n[b]`, and channel `c = d + H * s`,
 * the convolution input is
 *
 *   N[c, k, b] = history[c, 9 + k, b]                    for -9 <= k < 0
 *              = normalized_conv[c, offsets[b] + k, b]   for 0 <= k < n[b]
 *
 *   v[c, t, b] = sum_{j = 0}^{3} kernel[c, j] * N[c, t - 9 + 3 * j, b]
 *   updated_residual[d, s, t, b] = round_bf16(residual[d, s, t, b] + gated_value[c, t, b] + SiLU(v[c, t, b]))
 *
 * The dilation is 3, tap 0 reads `t - 9` and tap 3 reads `t`, and there is no bias. `SiLU(x) =
 * x / (1 + exp(-x))`. The gate's `normalized_conv` is the convolution input, so this Op consumes
 * the words `ple_gate` produced. `updated_residual` is an observable output.
 *
 * `ple_conv_add_norm` adds the per-stream offset RMSNorm of that update:
 *
 *   normalized[d, s, t, b] = updated_residual[d, s, t, b] * (1 + next_hc_norm_weight[d + H * s])
 *                            / sqrt( sum_{d' = 0}^{H-1} stored_updated_residual[d', s, t, b]^2 / H + epsilon )
 *
 * and consumes the stored BF16 words of `updated_residual`. The trunk uses the unnormalized exit
 * at the PLE boundary because the PLE layer's combine already skipped its normalization; the
 * normalized exit is the one that feeds the next HC.
 *
 * Logical shapes:
 *   H  hidden width, 2560
 *   S  stream count, 4
 *   C  channels, 10240
 *   T  token extent, positive
 *   B  batch rows, positive
 *   history            BF16 [C, 9, B] contiguous, dimension 0 fastest
 *   normalized_conv    BF16 [C, T, B] contiguous, dimension 0 fastest
 *   residual           BF16 [H, S, T, B] contiguous, dimension 0 fastest
 *   gated_value        BF16 [C, T, B] contiguous, dimension 0 fastest
 *   kernel             BF16 [4, C]    contiguous, tap-major: element (j, c) at j * C + c
 *   offsets            I32 [B]        contiguous, column start of each row inside normalized_conv
 *   counts             I32 [B]        contiguous, live column count n[b] of each row
 *   next_hc_norm_weight BF16 [C]      contiguous, stream-major
 *   updated_residual   BF16 [H, S, T, B] contiguous, dimension 0 fastest
 *   normalized         BF16 [H, S, T, B] contiguous, dimension 0 fastest
 *   Element (d, s, t, b) of a four-dimensional operand is stored at
 *   d + H * s + C * (t + T * b); element (c, k, b) of a three-dimensional operand is stored at
 *   c + C * k + C * extent * b.
 *
 * Supported domain: the registered geometry above, 16-byte aligned bases, epsilon positive and
 * finite. `counts[b] <= T`, `offsets[b] + counts[b] <= T`, and the live columns of different rows
 * do not overlap. `n[b] = 0` writes nothing for that row.
 *
 * Numeric: the convolution sum and the SiLU are evaluated in FP32, and the statistic is
 * accumulated in FP32 over the stored BF16 update words; both are required implementation
 * profiles. The oracle evaluates the complete formula in FP64 from the represented inputs and
 * reproduces the BF16 storage rounding of `updated_residual`, which is an observable output; the
 * outputs are promoted and compared against that result, so their own storage rounding belongs to
 * the numerical criterion and not to the oracle.
 *
 * Effects: writes all of `updated_residual`, and for the normalized exit all of `normalized`. The
 * outputs are disjoint from every input. Read-only inputs may share backing and are preserved.
 * This Op does not advance the convolution history; `ple_state_extract` owns that transition.
 *
 * Workspace: none.
 *
 * Execution: enqueues on the given stream, allocates no device memory, keeps no persistent state
 * and supports CUDA Graph capture and replay at bound addresses.
 */
void ple_conv_add(const Tensor& history, const Tensor& normalized_conv, const Tensor& residual,
                  const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                  const Tensor& counts, Tensor& updated_residual, cudaStream_t stream);

void ple_conv_add_norm(const Tensor& history, const Tensor& normalized_conv, const Tensor& residual,
                       const Tensor& gated_value, const Tensor& kernel, const Tensor& offsets,
                       const Tensor& counts, const Tensor& next_hc_norm_weight, float epsilon,
                       Tensor& updated_residual, Tensor& normalized, cudaStream_t stream);

/**
 * Op: the convolution history transition of one accepted round.
 *
 * Math / indexing: for row `b` with `a = accepted_columns[b]`, channel `c` and `j` in `[0, 9)`,
 * with `k = a - 9 + j`:
 *
 *   after_history[c, j, b] = before_history[c, 9 + k, b]  when k < 0
 *                          = records_values[c, k, b]      otherwise
 *
 * `a = 0` copies the complete `before` history. The extraction is exact BF16 word copying; no
 * floating-point evaluation is involved, and signed zero is preserved.
 *
 * Logical shapes:
 *   C  channels, 10240
 *   B  batch rows, positive
 *   before_history   BF16 [C, 9, B] contiguous, dimension 0 fastest
 *   records_values   BF16 [C, T, B] contiguous, dimension 0 fastest, columns `[0, a)` only
 *   accepted_columns I32 [B]        contiguous
 *   after_history    BF16 [C, 9, B] contiguous, dimension 0 fastest
 *   Element (c, j, b) of a history tensor is stored at c + C * j + C * 9 * b.
 *
 * Supported domain: the registered geometry above, 16-byte aligned bases, `accepted_columns[b]` in
 * `[0, T]`.
 *
 * Numeric: exact word copy.
 *
 * Effects: writes all of `after_history`. It may exactly alias `before_history` or be disjoint
 * from it; every other overlap between the output and an input is forbidden. Read-only inputs are
 * preserved.
 *
 * Workspace: none.
 *
 * Execution: enqueues on the given stream, allocates no device memory and supports CUDA Graph
 * capture and replay at bound addresses.
 */
void ple_state_extract(const Tensor& before_history, const Tensor& records_values,
                       const Tensor& accepted_columns, Tensor& after_history, cudaStream_t stream);

} // namespace ninfer::ops
