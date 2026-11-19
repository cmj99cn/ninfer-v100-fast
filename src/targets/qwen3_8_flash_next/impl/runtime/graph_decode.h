#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/diag_switches.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct GraphDecodeInputs {
    std::array<std::unique_ptr<PinnedHostBuffer>, kMaximumDevices> ids;
    std::array<std::unique_ptr<PinnedHostBuffer>, kMaximumDevices> positions;
    std::array<std::unique_ptr<PinnedHostBuffer>, kMaximumDevices> scalars;
    std::array<std::unique_ptr<PinnedHostBuffer>, kMaximumDevices> winner;
    std::unique_ptr<PinnedHostBuffer> logits_buffer;
    bool active = false;
};

inline GraphDecodeInputs& graph_decode_inputs() noexcept {
    static GraphDecodeInputs inputs;
    return inputs;
}

inline std::int32_t* graph_decode_scalar(int rank, const char* label) {
    auto* scalars = static_cast<std::int32_t*>(graph_decode_inputs().scalars[rank]->data());
    if (std::strcmp(label, "ple_offset") == 0) return scalars + 0;
    if (std::strcmp(label, "ple_count") == 0) return scalars + 1;
    if (std::strcmp(label, "live") == 0) return scalars + 2;
    if (std::strcmp(label, "row") == 0) return scalars + 3;
    throw std::runtime_error(std::string("unknown decode graph scalar label: ") + label);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail

#define NINFER_QWEN4EXP_GRAPH_IDS_SRC(rank, tokens)                                        \
    (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active             \
         ? static_cast<const void*>(                                                        \
               ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs()           \
                   .ids[static_cast<std::size_t>(rank)]->data())                            \
         : static_cast<const void*>((tokens).data()))
#define NINFER_QWEN4EXP_GRAPH_POSITIONS_SRC(rank, host_positions)                          \
    (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active             \
         ? static_cast<const void*>(                                                        \
               ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs()           \
                   .positions[static_cast<std::size_t>(rank)]->data())                      \
         : static_cast<const void*>((host_positions).data()))
#define NINFER_QWEN4EXP_GRAPH_SCALARS_SRC(rank, value, label)                              \
    (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active             \
         ? static_cast<const void*>(                                                        \
               ninfer::targets::qwen3_8_flash_next::detail::graph_decode_scalar(             \
                   (rank), (label)))                                                        \
         : static_cast<const void*>(&(value)))
#define NINFER_QWEN4EXP_GRAPH_WINNER_PTR()                                                 \
    (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active             \
         ? static_cast<void*>(                                                              \
               ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs()           \
                   .winner[0]->data())                                                      \
         : static_cast<void*>(&token))
#define NINFER_QWEN4EXP_GRAPH_WINNER_COPY(dst, src, bytes, stream)                         \
    do {                                                                                    \
        if (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {     \
            CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), cudaMemcpyDeviceToHost,        \
                                       (stream)));                                          \
        } else {                                                                            \
            CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost));           \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_MOE_ENABLED() \
    (!ninfer::targets::qwen3_8_flash_next::detail::diag_no_moe())
#define NINFER_QWEN4EXP_HAS_PLE() \
    (!ninfer::targets::qwen3_8_flash_next::detail::diag_no_ple())
#define NINFER_QWEN4EXP_GRAPH_PREPARE_PLE(rank, state, tokens)                             \
    do {                                                                                    \
        if (NINFER_QWEN4EXP_HAS_PLE() &&                                                    \
            !ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {    \
            (state).prepare_ple_rows(tokens);                                               \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_GRAPH_PLE_AWAIT(rank, state)                                       \
    do {                                                                                    \
        if (NINFER_QWEN4EXP_HAS_PLE() &&                                                    \
            !ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {    \
            (state).await_ple_rows();                                                       \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_GRAPH_ROUTED_FINISH(weights, stream)                               \
    do {                                                                                    \
        if (NINFER_QWEN4EXP_MOE_ENABLED() &&                                                \
            !ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {    \
            Variant::post_mixer_routed_finish((weights), (stream));                         \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_GRAPH_PREFETCH(weights, stream)                                    \
    do {                                                                                    \
        if (NINFER_QWEN4EXP_MOE_ENABLED() &&                                                \
            !ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {    \
            Variant::post_mixer_prefetch((weights), (stream));                              \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_GRAPH_FINISH_SYNC()                                                \
    do {                                                                                    \
        if (!ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {    \
            CUDA_CHECK(cudaStreamSynchronize(states_[0]->stream()));                        \
        }                                                                                   \
    } while (0)
#define NINFER_QWEN4EXP_LAST_STAGE(sample, logits)                                         \
    do {                                                                                    \
        if (ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs().active) {     \
            if (rank == 0) {                                                                \
                CUDA_CHECK(cudaMemcpyAsync(                                                  \
                    ninfer::targets::qwen3_8_flash_next::detail::graph_decode_inputs()       \
                        .logits_buffer->data(),                                                     \
                    static_cast<const std::byte*>((logits).data) +                           \
                        static_cast<std::size_t>(static_cast<std::int32_t>(T) - 1) *         \
                            static_cast<std::size_t>(TextConfig::vocab) * 2,                 \
                    static_cast<std::size_t>(TextConfig::vocab) * 2,                         \
                    cudaMemcpyDeviceToHost, stream_for(rank)));                              \
            }                                                                               \
        } else {                                                                            \
            capture_last_stage((logits), stream_for(rank));                                  \
        }                                                                                   \
    } while (0)
