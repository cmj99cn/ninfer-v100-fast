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

} // namespace ninfer::targets::qwen3_8_flash_next::detail
