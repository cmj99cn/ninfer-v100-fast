#pragma once

#include <cstdlib>
#include <cstring>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace diag_detail {

inline bool env_flag(const char* name) noexcept {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace diag_detail

inline bool diag_no_moe() noexcept {
    static const bool enabled = diag_detail::env_flag("NINFER_DIAG_NO_MOE");
    return enabled;
}

inline bool diag_no_ple() noexcept {
    static const bool enabled = diag_detail::env_flag("NINFER_DIAG_NO_PLE");
    return enabled;
}

// Selects the FP16 text/output_head for the MTP draft instead of the registered Q4 proposal/head.
// The Q4 head is the default; the switch exists only for controlled A/B measurement.
inline bool diag_fp16_draft_head() noexcept {
    static const bool enabled = diag_detail::env_flag("NINFER_DIAG_FP16_DRAFT_HEAD");
    return enabled;
}

// Minimum top-1 probability for the MTP draft head to extend the chain. Zero (the default, and any
// negative value) leaves the fixed-length draft chain unchanged.
inline float draft_min_p() noexcept {
    static const float value = [] {
        const char* environment = std::getenv("NINFER_FLASH_NEXT_DRAFT_MIN_P");
        const float parsed = environment != nullptr ? static_cast<float>(std::atof(environment))
                                                    : 0.0F;
        return parsed > 0.0F ? parsed : 0.0F;
    }();
    return value;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
