#include "product/speculative_options.h"

#include <ninfer/engine.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using ninfer::Engine;
using ninfer::EngineOptions;
using ninfer::ProposalHead;
using ninfer::SpeculativeBackend;
using ninfer::SpeculativeOptions;
namespace product = ninfer::product;

template <class F>
bool throws_with(F&& fn, std::string_view needle) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string_view(error.what()).find(needle) != std::string_view::npos;
    }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int test_enum_and_parse() {
    static_assert(static_cast<std::uint8_t>(SpeculativeBackend::None) == 0);
    static_assert(static_cast<std::uint8_t>(SpeculativeBackend::Mtp) == 1);
    static_assert(static_cast<std::uint8_t>(SpeculativeBackend::DFlash) == 2);
    static_assert(static_cast<std::uint8_t>(SpeculativeBackend::DFlash2) == 3,
                  "DFlash2 must be appended without renumbering existing backends");

    int failures = 0;
    failures += check(product::parse_speculative_backend("none") == SpeculativeBackend::None,
                      "parse('none') is not the disabled backend");
    failures += check(product::parse_speculative_backend("mtp") == SpeculativeBackend::Mtp,
                      "parse('mtp') changed");
    failures += check(product::parse_speculative_backend("dflash") == SpeculativeBackend::DFlash,
                      "parse('dflash') changed");
    failures +=
        check(product::parse_speculative_backend("dflash2") == SpeculativeBackend::DFlash2,
              "parse('dflash2') did not select the DFlash2 backend");
    failures += check(std::string_view(product::speculative_backend_name(
                          SpeculativeBackend::DFlash2)) == "dflash2",
                      "speculative_backend_name(DFlash2) is not 'dflash2'");
    failures += check(throws_with([] { (void)product::parse_speculative_backend("dflash3"); },
                                  "invalid speculative backend"),
                      "an unknown speculative backend was accepted");
    return failures;
}

int test_validate() {
    int failures = 0;
    SpeculativeOptions options;
    options.backend      = SpeculativeBackend::DFlash2;
    options.proposal_head = ProposalHead::Full;

    options.draft_tokens = 1;
    failures += check(!throws_with([&] { product::validate_speculative_cli_options(options); }, ""),
                      "DFlash2 rejected the minimum draft window");
    options.draft_tokens = 7;
    failures += check(!throws_with([&] { product::validate_speculative_cli_options(options); }, ""),
                      "DFlash2 rejected the maximum draft window");
    options.draft_tokens = 0;
    failures += check(throws_with([&] { product::validate_speculative_cli_options(options); },
                                  "--spec dflash2 requires --draft-tokens in [1,7]"),
                      "DFlash2 accepted a zero draft window");
    options.draft_tokens = 8;
    failures += check(throws_with([&] { product::validate_speculative_cli_options(options); },
                                  "--spec dflash2 requires --draft-tokens in [1,7]"),
                      "DFlash2 accepted an eight-token draft window");

    options.backend      = SpeculativeBackend::Mtp;
    options.draft_tokens = 5;
    failures += check(!throws_with([&] { product::validate_speculative_cli_options(options); }, ""),
                      "MTP validation changed at the upper bound");
    options.draft_tokens = 6;
    failures += check(throws_with([&] { product::validate_speculative_cli_options(options); },
                                  "--spec mtp requires --draft-tokens in [1,5]"),
                      "MTP validation changed when rejecting six tokens");

    options.backend      = SpeculativeBackend::DFlash;
    options.draft_tokens = 15;
    failures += check(!throws_with([&] { product::validate_speculative_cli_options(options); }, ""),
                      "DFlash validation changed");
    options.draft_tokens = 16;
    failures += check(throws_with([&] { product::validate_speculative_cli_options(options); },
                                  "--spec dflash requires --draft-tokens in [1,15]"),
                      "DFlash validation changed when rejecting sixteen tokens");

    options.backend      = SpeculativeBackend::None;
    options.draft_tokens = 0;
    failures += check(!throws_with([&] { product::validate_speculative_cli_options(options); }, ""),
                      "none validation changed");
    options.draft_tokens = 3;
    failures += check(throws_with([&] { product::validate_speculative_cli_options(options); },
                                  "--draft-tokens and --lm-head-draft require --spec"),
                      "none still accepted a draft window");
    return failures;
}

int test_engine_requires_draft_model() {
    int failures = 0;
    EngineOptions options;
    options.artifact_path                 = "/nonexistent/flash-next.ninfer";
    options.speculative.backend           = SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens      = 3;
    options.speculative_draft_artifact.clear();

    failures += check(throws_with([&] { Engine engine(options); },
                                  "--spec dflash2 requires --draft-model <path>"),
                      "DFlash2 without --draft-model did not fail at construction");

    options.speculative_draft_artifact = "/nonexistent/dflash2.infernix";
    failures += check(!throws_with([&] { Engine engine(options); },
                                   "--spec dflash2 requires --draft-model <path>"),
                      "DFlash2 with a draft-model path reported a missing path");

    options.speculative.backend = SpeculativeBackend::Mtp;
    options.speculative_draft_artifact.clear();
    failures += check(!throws_with([&] { Engine engine(options); },
                                   "--spec dflash2 requires --draft-model <path>"),
                      "MTP was incorrectly required to carry a draft-model path");
    return failures;
}

} // namespace

int main() {
    return test_enum_and_parse() + test_validate() + test_engine_requires_draft_model();
}
