#pragma once

#include "core/tensor.h"

#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class MtpDraft {
public:
    static constexpr std::size_t hidden_elements = 2560 * 4;

    explicit MtpDraft(const std::filesystem::path& artifact, int device);
    ~MtpDraft();

    MtpDraft(const MtpDraft&)            = delete;
    MtpDraft& operator=(const MtpDraft&) = delete;
    MtpDraft(MtpDraft&&)                 = delete;
    MtpDraft& operator=(MtpDraft&&)      = delete;

    TokenId run(const Tensor& conditioning, TokenId next_token, std::int32_t position,
                Tensor* chained_hidden = nullptr);

    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] std::uint64_t expert_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
