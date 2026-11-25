#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/mtp_bindings.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void check_config() {
    require(flash::MtpConfig::draft_window == 3, "the MTP draft window is not 3");
    require(flash::MtpConfig::draft_window <= flash::MtpConfig::maximum_draft_window,
            "the MTP draft window exceeds the seven-draft contract");
    require(flash::MtpConfig::layers == 1, "the artifact declares one MTP layer");
}

void check_artifact(const char* path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const flash::BindingPlan text = flash::bind_artifact(binder, flash::WeightsProfile::Nvfp4, 1);
    const auto mtp = flash::mtp::bind_mtp_artifact(binder, text);
    require(mtp.binding_count == 37, "the MTP plan did not parse all 37 bindings");

    const auto& gate = mtp.layer.moe.routed_gate.library;
    const auto& up = mtp.layer.moe.routed_up.library;
    const auto& down = mtp.layer.moe.routed_down.library;
    require(gate.has_value() && up.has_value() && down.has_value(),
            "the MTP routed expert bank has no library geometry");
    require(gate->matrices == 512 && up->matrices == 512 && down->matrices == 512,
            "the MTP expert bank is not 512 matrices");
    require(gate->object == "weight/000096" && up->object == "weight/000096",
            "the MTP gate/up bank is not the artifact's 000096 object");
    require(down->object == "weight/000097",
            "the MTP down bank is not the artifact's 000097 object");
    require(binder.tensor_format("weight/000096") ==
                ninfer::artifact::NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            "the MTP gate/up bank is not the artifact's block FP8 bank");
    require(binder.tensor_format("weight/000097") ==
                ninfer::artifact::NumericFormat::FP8_E4M3FN_BLOCK128_BF16S,
            "the MTP down bank is not the artifact's block FP8 bank");

    require(mtp.token_embedding.parts.front().object == text.token_embedding.parts.front().object,
            "mtp/token_embedding does not share the text embedding object");
    require(mtp.output_head.parts.front().object == text.output_head.parts.front().object,
            "mtp/output_head does not share the text output head object");
    require(mtp.proposal_head.parameter == "proposal/head",
            "the MTP proposal head does not bind the artifact's proposal/head parameter");
    const auto& proposal = mtp.objects.at(mtp.proposal_head.parts.front().object);
    require(proposal.name == "weight/001310", "the MTP proposal head is not object 001310");
    require(binder.tensor_format(proposal.name) ==
                ninfer::artifact::NumericFormat::Q4G64_F16S,
            "the MTP proposal head is not the artifact's Q4 head");

    const std::set<std::string> roles{
        "token_embedding", "output_head", "fc_embedding", "fc_hidden",
        "pre_fc_embedding_norm", "pre_fc_hidden_norm",
    };
    require(roles.size() == 6, "the MTP global role list changed");
}

} // namespace

int main(int argc, char** argv) {
    try {
        check_config();
        if (argc > 1) { check_artifact(argv[1]); }
        else if (const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT")) { check_artifact(path); }
        else { std::cout << "skip: NINFER_FLASH_NEXT_ARTIFACT is not set\n"; return 77; }
        std::cout << "mtp bindings ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mtp bindings failed: " << error.what() << "\n";
        return 1;
    }
}
