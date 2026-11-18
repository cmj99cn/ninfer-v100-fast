#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

std::vector<TokenId> draft_chain(MtpDraft& draft, const Tensor& conditioning, TokenId token,
                                 std::uint32_t steps, std::int32_t start_position) {
    std::vector<TokenId> tokens;
    tokens.reserve(steps);
    if (steps == 0) { return tokens; }

    const std::size_t bytes = static_cast<std::size_t>(MtpDraft::hidden_elements) * sizeof(std::uint16_t);
    CUDA_CHECK(cudaSetDevice(draft.device()));
    ninfer::DeviceBuffer current(bytes);
    ninfer::DeviceBuffer next(bytes);
    CUDA_CHECK(cudaMemcpy(current.p, conditioning.data, bytes, cudaMemcpyDeviceToDevice));
    for (std::uint32_t step = 0; step < steps; ++step) {
        const Tensor current_view(current.p, DType::FP16, {2560, 4});
        Tensor next_view(next.p, DType::FP16, {2560, 4});
        token = draft.run(current_view, token, start_position + static_cast<std::int32_t>(step),
                          &next_view);
        tokens.push_back(token);
        std::swap(current, next);
    }
    return tokens;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
