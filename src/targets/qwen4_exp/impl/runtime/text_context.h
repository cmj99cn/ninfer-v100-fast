#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

struct TextLaneContext {
    std::vector<TokenId> prompt_tokens;
    std::uint32_t processed_prompt_tokens = 0;
    TokenId next_input = 0;
    bool decode_ready = false;
};

// The persistent text state of one lane: the per-layer GDN convolution and recurrent state pools,
// the QSA cache planes with its canonical tail and call records, and the PLE convolution history.
class TextState {
public:
    TextState(DeviceContext& device, const ModelView& model, std::uint32_t kv_capacity_tokens,
              std::uint32_t forward_tokens);

    void reset();
    void prepare_ple_rows(std::span<const TokenId> tokens);
    [[nodiscard]] std::int32_t base_position() const;

    [[nodiscard]] Tensor gdn_conv_state(std::size_t gdn_index) const;
    [[nodiscard]] Tensor gdn_recurrent_state(std::size_t gdn_index) const;
    [[nodiscard]] std::size_t gdn_state_index(std::size_t layer) const;
    [[nodiscard]] std::size_t gdn_state_entries() const noexcept;

    [[nodiscard]] ops::QsaCacheView cache() const;
    [[nodiscard]] ops::QsaTailStateView tail() const;
    [[nodiscard]] ops::QsaTailStateView before_tail() const;
    [[nodiscard]] ops::ConstQsaTailStateView const_tail() const;
    [[nodiscard]] ops::QsaRawRecordWriteView records(std::size_t entries) const;

    [[nodiscard]] Tensor ple_history() const;
    [[nodiscard]] const std::uint32_t* ple_rows() const noexcept;
    [[nodiscard]] const std::uint8_t* ple_staged_rows() const noexcept;
    [[nodiscard]] const ops::PleNf4Table& ple_table() const noexcept;

    [[nodiscard]] std::uint32_t next_position() const noexcept { return next_position_; }
    void advance_position(std::uint32_t tokens) noexcept { next_position_ += tokens; }

    [[nodiscard]] std::uint64_t ple_tail(std::size_t index) const noexcept { return ple_tail_[index]; }
    void advance_ple_tail(std::uint64_t id) noexcept {
        ple_tail_[0] = ple_tail_[1];
        ple_tail_[1] = id;
    }

    void publish_tail();

private:
    class DeviceScope {
    public:
        explicit DeviceScope(int device);
        ~DeviceScope();

        DeviceScope(const DeviceScope&)            = delete;
        DeviceScope& operator=(const DeviceScope&) = delete;

    private:
        int previous_ = -1;
    };

    DeviceScope device_scope_;
    const qwen4_exp::PleRowSource ple_source_;
    cudaStream_t stream_         = nullptr;
    std::size_t layers_          = 0;
    std::size_t forward_tokens_  = 0;
    std::size_t pages_           = 0;
    std::vector<std::size_t> gdn_layers_;
    std::vector<std::size_t> gdn_index_;
    std::vector<std::byte> ple_staged_host_;
    std::uint32_t next_position_ = 0;
    std::uint64_t ple_tail_[2]   = {0, 0};
    ops::PleHashControls ple_controls_{};
    ops::PleNf4Table ple_table_{};

    DeviceBuffer conv_states_;
    DeviceBuffer recurrent_states_;
    DeviceBuffer k_pages_;
    DeviceBuffer v_pages_;
    DeviceBuffer index_k_pages_;
    DeviceBuffer page_table_;
    DeviceBuffer tail_raw_keys_;
    DeviceBuffer tail_first_rope_;
    DeviceBuffer after_raw_keys_;
    DeviceBuffer after_first_rope_;
    DeviceBuffer record_raw_keys_;
    DeviceBuffer record_rope_positions_;
    DeviceBuffer record_base_positions_;
    DeviceBuffer ple_history_;
    DeviceBuffer ple_rows_;
    DeviceBuffer ple_staged_rows_;
    DeviceBuffer ple_divisors_;
};

// The text trunk: the embedding stem, the per-layer four-stream mixer and post-mixer, and the main
// head. It runs the fixed order of the artifact authority and calls the package's three execution
// leaves.
class TextTrunk {
public:
    TextTrunk(const ModelView& model, TextState& state, WorkspaceArena& activations,
             WorkspaceArena& workspace, cudaStream_t stream);

    TokenId prefill(std::span<const TokenId> tokens);
    TokenId decode(TokenId token);

private:
    TokenId forward(std::span<const TokenId> tokens);
    void run_gdn(std::size_t layer, std::int32_t tokens, const Tensor& block_input, Tensor& mixer_out);
    void run_qsa(std::size_t layer, std::int32_t tokens, const Tensor& block_input,
                 const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
                 Tensor& mixer_out);
    void run_ple(std::int32_t tokens, const Tensor& residual, Tensor& updated, Tensor& normalized);

    const ModelView& model_;
    TextState& state_;
    WorkspaceArena& activations_;
    WorkspaceArena& workspace_;
    cudaStream_t stream_;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
