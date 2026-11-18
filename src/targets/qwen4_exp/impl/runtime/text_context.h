#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "runtime/contract/types.h"
#include "targets/qwen4_exp/impl/runtime/instance.h"

#include <cuda_runtime.h>

#include <array>
#include <future>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS {

struct QsaLayerStateView {
    ops::QsaCacheView cache{};
    ops::QsaTailStateView before{};
    ops::ConstQsaTailStateView const_tail{};
    ops::QsaTailStateView after{};
    ops::QsaRawRecordWriteView records{};
    void (*publish)(void* context) = nullptr;
    void* publish_context = nullptr;
};

struct LastStageCapture {
    bool enabled = false;
    std::vector<std::uint16_t> words;
};

inline LastStageCapture*& last_stage_capture() noexcept {
    static thread_local LastStageCapture* capture = nullptr;
    return capture;
}

inline void capture_last_stage(const Tensor& logits, cudaStream_t stream) {
    LastStageCapture* capture = last_stage_capture();
    if (capture == nullptr || !capture->enabled) { return; }
    const std::int32_t vocab = TextConfig::vocab;
    const std::int32_t tokens = logits.ne[1];
    capture->words.resize(static_cast<std::size_t>(vocab));
    CUDA_CHECK(cudaMemcpyAsync(
        capture->words.data(),
        static_cast<const std::byte*>(logits.data) +
            static_cast<std::size_t>(tokens - 1) * static_cast<std::size_t>(vocab) * 2,
        static_cast<std::size_t>(vocab) * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

struct TextLaneContext {
    std::vector<TokenId> prompt_tokens;
    std::uint32_t processed_prompt_tokens = 0;
    TokenId next_input = 0;
    bool decode_ready = false;
};

// The persistent text state of one lane on one rank: the per-layer GDN convolution and recurrent
// state pools, the QSA cache planes with its canonical tail and call records, and the PLE
// convolution history. A tensor-parallel rank holds only its own heads.
class TextState {
public:
    TextState(DeviceContext& device, const ModelView& model, std::uint32_t kv_capacity_tokens,
              std::uint32_t forward_tokens);
    TextState(DeviceContext& device, const ModelView& model, std::uint32_t kv_capacity_tokens,
              std::uint32_t forward_tokens, int tp, int rank);
    ~TextState();

    void reset();
    void prepare_ple_rows(std::span<const TokenId> tokens);
    void await_ple_rows();
    void enqueue_ple_wait();
    [[nodiscard]] std::int32_t base_position() const;

    [[nodiscard]] Tensor gdn_conv_state(std::size_t gdn_index) const;
    [[nodiscard]] Tensor gdn_recurrent_state(std::size_t gdn_index) const;
    [[nodiscard]] std::optional<std::size_t> gdn_state_index(std::size_t layer) const;
    [[nodiscard]] std::size_t gdn_state_entries() const noexcept;

    [[nodiscard]] int device_id() const noexcept { return device_; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    [[nodiscard]] Tensor gdn_conv_kernel(std::size_t layer) const;

    [[nodiscard]] std::int32_t qsa_slot(std::size_t layer) const;
    [[nodiscard]] ops::QsaCacheView cache(std::int32_t slot) const;
    [[nodiscard]] ops::QsaTailStateView tail(std::int32_t slot) const;
    [[nodiscard]] ops::ConstQsaTailStateView const_tail(std::int32_t slot) const;
    [[nodiscard]] ops::QsaTailStateView before_tail(std::int32_t slot) const;
    void publish_tail(std::int32_t slot);

    [[nodiscard]] std::int32_t kv_heads() const noexcept { return kv_heads_; }
    [[nodiscard]] std::int32_t gdn_key_heads() const noexcept { return gdn_key_heads_; }
    [[nodiscard]] std::int32_t gdn_value_heads() const noexcept { return gdn_value_heads_; }
    [[nodiscard]] std::int32_t conv_channels() const noexcept { return conv_channels_; }

    [[nodiscard]] ops::QsaCacheView cache() const;
    [[nodiscard]] ops::QsaTailStateView tail() const;
    [[nodiscard]] ops::QsaTailStateView before_tail() const;
    [[nodiscard]] ops::ConstQsaTailStateView const_tail() const;
    [[nodiscard]] ops::QsaRawRecordWriteView records(std::size_t entries) const;

    [[nodiscard]] Tensor ple_history() const;
    [[nodiscard]] const std::uint32_t* ple_rows() const noexcept;
    [[nodiscard]] const std::uint8_t* ple_staged_rows() const noexcept;
    [[nodiscard]] const ops::PleNf4Table& ple_table() const noexcept;
    [[nodiscard]] std::int32_t* route_counter() const noexcept {
        return static_cast<std::int32_t*>(route_counter_.p);
    }

    [[nodiscard]] std::uint32_t next_position() const noexcept { return next_position_; }
    void advance_position(std::uint32_t tokens) noexcept { next_position_ += tokens; }
    void rewind_scalars(std::uint32_t position, std::uint64_t previous,
                        std::uint64_t current) noexcept {
        next_position_ = position;
        ple_tail_[0]   = previous;
        ple_tail_[1]   = current;
    }

    [[nodiscard]] std::uint64_t kv_payload_bytes() const noexcept;
    [[nodiscard]] std::uint64_t gdn_state_bytes() const noexcept;
    [[nodiscard]] std::uint64_t reserved_bytes() const noexcept;

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
    int device_                  = 0;
    int rank_                    = 0;
    cudaStream_t stream_         = nullptr;
    std::size_t layers_          = 0;
    std::size_t forward_tokens_  = 0;
    std::size_t pages_           = 0;
    std::int32_t kv_heads_       = 0;
    std::int32_t gdn_key_heads_  = 0;
    std::int32_t gdn_value_heads_ = 0;
    std::int32_t conv_channels_  = 0;
    std::vector<std::size_t> gdn_layers_;
    std::vector<std::optional<std::size_t>> gdn_index_;
    std::int32_t qsa_layers_ = 1;
    std::vector<std::int32_t> qsa_slot_of_;
    std::size_t qsa_page_bytes_ = 0;
    std::size_t qsa_index_page_bytes_ = 0;
    std::size_t qsa_tail_bytes_ = 0;
    std::size_t qsa_tail_rope_bytes_ = 0;
    std::unique_ptr<PinnedHostBuffer> ple_rows_host_;
    std::unique_ptr<PinnedHostBuffer> ple_staged_host_;
    cudaStream_t ple_stream_ = nullptr;
    cudaEvent_t ple_ready_   = nullptr;
    std::unique_ptr<HostWorkerPool> ple_workers_;
    std::future<void> ple_prefetch_;
    std::uint32_t next_position_ = 0;
    std::uint64_t ple_tail_[2]   = {0, 0};
    ops::PleHashControls ple_controls_{};
    ops::PleNf4Table ple_table_{};

    DeviceBuffer conv_kernels_;
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
    DeviceBuffer route_counter_;
};

// The text trunk: the embedding stem, the per-layer four-stream mixer and post-mixer, and the main
// head. It runs the fixed order of the artifact authority and calls the package's three execution
// leaves. The tensor-parallel form issues the same schedule on every rank and joins the two
// row-parallel projections with `allreduce_sum`.
class TextTrunk {
public:
    TextTrunk(const ModelView& model, TextState& state, WorkspaceArena& activations,
              WorkspaceArena& workspace, cudaStream_t stream);
    TextTrunk(const std::array<const ModelView*, kMaximumDevices>& models,
              const std::array<TextState*, kMaximumDevices>& states,
              const std::array<WorkspaceArena*, kMaximumDevices>& activations,
              const std::array<WorkspaceArena*, kMaximumDevices>& workspaces,
              ExecutionContext& execution, const ops::CollectiveComms& comms);

    TokenId prefill(std::span<const TokenId> tokens);
    TokenId decode(TokenId token);

    template <class AttentionWeights>
    void run_qsa_layer(std::int32_t tokens,
                       const std::array<const AttentionWeights*, kMaximumDevices>& weights,
                       const std::array<QsaLayerStateView, kMaximumDevices>& state,
                       const std::array<Tensor, kMaximumDevices>& block_input,
                       const std::array<Tensor, kMaximumDevices>& positions,
                       const std::array<Tensor, kMaximumDevices>& valid_columns,
                       const std::array<Tensor, kMaximumDevices>& table_rows,
                       std::array<Tensor, kMaximumDevices>& mixer_out,
                       const std::array<Tensor, kMaximumDevices>& staging);

private:
    TokenId forward(std::span<const TokenId> tokens);

    void run_gdn(std::size_t layer, std::int32_t tokens,
                 const std::array<Tensor, kMaximumDevices>& block_input,
                 std::array<Tensor, kMaximumDevices>& mixer_out,
                 const std::array<Tensor, kMaximumDevices>& staging);
    void run_qsa(std::size_t layer, std::int32_t tokens,
                 const std::array<Tensor, kMaximumDevices>& block_input,
                 const std::array<Tensor, kMaximumDevices>& positions,
                 const std::array<Tensor, kMaximumDevices>& valid_columns,
                 const std::array<Tensor, kMaximumDevices>& table_rows,
                 std::array<Tensor, kMaximumDevices>& mixer_out,
                 const std::array<Tensor, kMaximumDevices>& staging);
    void run_ple(std::int32_t tokens, std::array<Tensor, kMaximumDevices>& residual,
                 std::array<Tensor, kMaximumDevices>& updated,
                 std::array<Tensor, kMaximumDevices>& normalized);

    void row_parallel_linear(const std::array<Tensor, kMaximumDevices>& x,
                             const std::array<Weight, kMaximumDevices>& w,
                             const std::array<Tensor, kMaximumDevices>& out,
                             const std::array<Tensor, kMaximumDevices>& staging,
                             const std::array<WorkspaceArena*, kMaximumDevices>& workspaces) const;
    void reduce_rows(const std::array<Tensor, kMaximumDevices>& buffer,
                     const std::array<Tensor, kMaximumDevices>& staging) const;

    template <class Body>
    void for_each_rank(Body&& body) {
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        for (int rank = 0; rank < tp_; ++rank) {
            CUDA_CHECK(cudaSetDevice(states_[static_cast<std::size_t>(rank)]->device_id()));
            body(rank);
        }
        CUDA_CHECK(cudaSetDevice(previous));
    }

    [[nodiscard]] cudaStream_t stream_for(int rank) const noexcept {
        return states_[static_cast<std::size_t>(rank)]->stream();
    }

    std::array<const ModelView*, kMaximumDevices> models_{};
    std::array<TextState*, kMaximumDevices> states_{};
    std::array<WorkspaceArena*, kMaximumDevices> activations_{};
    std::array<WorkspaceArena*, kMaximumDevices> workspaces_{};
    ExecutionContext* execution_       = nullptr;
    const ops::CollectiveComms* comms_ = nullptr;
    int tp_                            = 1;
    cudaStream_t stream_               = nullptr;
};

} // namespace ninfer::targets::qwen4_exp::detail::NINFER_QWEN4EXP_RUNTIME_NS
