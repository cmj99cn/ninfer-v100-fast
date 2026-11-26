#pragma once

#include "ninfer/types.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "none") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        // The drafter checkpoint was trained with a four-token block and never proposes more
        // than seven drafts; the broader DFlash window does not apply to it.
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,7]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

// DFlash2 weights are not embedded in the main artifact, so a missing companion path is a
// construction error rather than a silent fallback to plain decoding.
inline void validate_speculative_draft_artifact(const SpeculativeOptions& options,
                                                const std::filesystem::path& draft_artifact) {
    if (options.backend == SpeculativeBackend::DFlash2 && draft_artifact.empty()) {
        throw std::invalid_argument("--spec dflash2 requires --draft-model <path>");
    }
}

} // namespace ninfer::product
