#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/dflash2_draft.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::StorageLayout;
namespace detail = ninfer::targets::qwen3_8_flash_next::detail;

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

std::map<std::string, std::string> valid_target_config() {
    return {{"architectures", "Qwen4ExpForCausalLM"},
            {"hidden_size", "2560"},
            {"vocab_size", "248320"},
            {"num_hidden_layers", "48"}};
}

int test_target_config_validation() {
    int failures = 0;
    auto config = valid_target_config();
    failures +=
        check(!throws_with([&] { detail::validate_dflash2_target_config(config); }, ""),
              "the matching target_config was rejected");

    config["architectures"] = "Qwen3ForCausalLM";
    failures += check(throws_with([&] { detail::validate_dflash2_target_config(config); },
                                  "Qwen4ExpForCausalLM"),
                      "a mismatched target architecture was accepted");
    config = valid_target_config();
    config["hidden_size"] = "4096";
    failures += check(throws_with([&] { detail::validate_dflash2_target_config(config); }, "2560"),
                      "a mismatched hidden size was accepted");
    config = valid_target_config();
    config["vocab_size"] = "151936";
    failures += check(throws_with([&] { detail::validate_dflash2_target_config(config); }, "248320"),
                      "a mismatched vocabulary was accepted");
    config = valid_target_config();
    config.erase("num_hidden_layers");
    failures += check(throws_with([&] { detail::validate_dflash2_target_config(config); },
                                  "num_hidden_layers"),
                      "a target_config without the layer count was accepted");
    return failures;
}

int test_companion_load_and_bind(const std::string& path) {
    int failures = 0;
    detail::DFlash2Draft draft(path);

    failures += check(draft.reader().identity().model_id == "qwen3.8-flash-next-dflash2" &&
                          draft.reader().identity().weights_id == "dflash2",
                      "the companion reader did not resolve its own artifact identity");

    const auto& tensors = draft.tensors();
    failures += check(tensors.size() == 66, "the dflash2 component did not bind all 66 tensors");
    std::size_t quantized = 0;
    std::size_t bf16      = 0;
    bool names_ok         = true;
    bool layouts_ok       = true;
    for (const auto& tensor : tensors) {
        names_ok = names_ok && tensor.name.starts_with("dflash2/");
        if (tensor.format == NumericFormat::W8G32_F16S) {
            ++quantized;
            layouts_ok = layouts_ok && tensor.layout == StorageLayout::RowSplitK128V1;
        } else if (tensor.format == NumericFormat::BF16) {
            ++bf16;
            layouts_ok = layouts_ok && tensor.layout == StorageLayout::ContiguousLeV1;
        }
    }
    failures += check(names_ok, "a bound tensor name is outside the dflash2 component");
    failures += check(quantized == 21 && bf16 == 45,
                      "the companion's 21 W8/45 BF16 tensor recipe changed");
    failures += check(layouts_ok, "a bound tensor kept the wrong storage layout");

    failures += check(draft.plan().device_objects.empty(),
                      "load-only DFlash2 binding allocated device materializations");
    failures += check(draft.plan().object_count == tensors.size(),
                      "the materialization plan does not cover every bound tensor");

    failures += check(draft.target_arch() == "Qwen4ExpForCausalLM" && draft.hidden() == 2560 &&
                          draft.vocab() == 248320,
                      "the companion's target_config was not retained");
    failures += check(draft.layers() == 5, "the companion is not a five-layer drafter");
    failures +=
        check(draft.target_layer_ids() == std::vector<std::uint32_t>({5, 15, 25, 35, 45}),
              "the companion's target layer ids changed");
    return failures;
}

} // namespace

int main() {
    const char* value = std::getenv("NINFER_DFLASH2_ARTIFACT");
    if (value == nullptr || value[0] == '\0') {
        std::cerr << "NINFER_DFLASH2_ARTIFACT is not set\n";
        return 77;
    }
    return test_target_config_validation() + test_companion_load_and_bind(value);
}
