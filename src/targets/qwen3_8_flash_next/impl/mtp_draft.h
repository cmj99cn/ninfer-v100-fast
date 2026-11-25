#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"

#include <ninfer/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Host-side router decision computed once on rank 0 and shared with the expert-only ranks through
// the owning MtpDraft (both ranks live in one Engine process and share its host address space).
struct DraftRouterChoice {
    static constexpr std::size_t kTopK = 10; // TextConfig::experts_per_token
    std::array<std::int32_t, kTopK> experts{};
    std::array<float, kTopK> scores{};
    float shared = 0.0F;
};

// One draft head across the Program's tensor-parallel rank group. Rank 0 owns the whole non-expert
// path (embedding, projections, QSA, hyper-connection mixing, shared expert and router) and its own
// contiguous expert range; every other rank runs only its contiguous expert range on the MoE input
// broadcast by rank 0. The router decision is computed once on rank 0 and passed to the peers in
// host memory. The per-rank FP32 accumulators are summed in FP32 before the single FP16 round, and
// rank 0 alone adds the shared term and runs the output head and argmax. Rank 0 alone holds the
// draft QSA state and chain scratch.
class MtpDraft {
public:
    static constexpr std::size_t hidden_elements = 2560 * 4;
    using RankTensors = std::array<Tensor, kMaximumDevices>;

    MtpDraft(const std::filesystem::path& artifact, const ExecutionContext& execution,
             const ops::CollectiveComms& comms);
    ~MtpDraft();

    MtpDraft(const MtpDraft&)            = delete;
    MtpDraft& operator=(const MtpDraft&) = delete;
    MtpDraft(MtpDraft&&)                 = delete;
    MtpDraft& operator=(MtpDraft&&)      = delete;

    // When `confidence` is non-null it receives the top-1 softmax probability of the produced
    // draft token. Passing null keeps the original kernel sequence and output bit-identical.
    TokenId run(const RankTensors& conditioning, TokenId next_token, std::int32_t position,
                RankTensors* chained_hidden = nullptr, float* confidence = nullptr);

    // A positive `min_p` stops the chain after the first step whose top-1 probability is below it;
    // the producing step is always kept. `min_p <= 0` keeps the full fixed-length chain.
    std::vector<TokenId> run_chain(const RankTensors& conditioning, TokenId first_token,
                                   std::uint32_t steps, std::int32_t start_position,
                                   float min_p = 0.0F);

    [[nodiscard]] int device() const noexcept;
    // Bytes held by one rank's routed expert banks, not the sum over ranks.
    [[nodiscard]] std::uint64_t expert_bytes() const noexcept;
    // Router host synchronizations per rank and draft step.
    [[nodiscard]] std::uint64_t host_syncs() const noexcept;

private:
    struct Impl;
    std::vector<std::unique_ptr<Impl>> ranks_;
    const ExecutionContext* execution_;
    const ops::CollectiveComms* comms_;
    DraftRouterChoice router_{};
    std::array<cudaEvent_t, kMaximumDevices> x_ready_{};
    std::array<cudaEvent_t, kMaximumDevices> x_after_{};
    std::array<cudaEvent_t, kMaximumDevices> reduce_before_{};
    std::array<cudaEvent_t, kMaximumDevices> reduce_after_{};
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
