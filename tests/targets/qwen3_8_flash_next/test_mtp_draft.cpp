#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;

namespace {

std::vector<std::uint16_t> read_hidden(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error(std::string("cannot open hidden fixture: ") + path); }
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    if (bytes.size() != 2 * static_cast<std::size_t>(
                            ninfer::targets::qwen3_8_flash_next::detail::MtpDraft::hidden_elements)) {
        throw std::runtime_error("hidden fixture has the wrong element count");
    }
    std::vector<std::uint16_t> data(bytes.size() / 2);
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return data;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const char* artifact = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        const char* hidden = argc > 2 ? argv[2] : std::getenv("NINFER_MTP_HIDDEN");
        if (artifact == nullptr || hidden == nullptr) {
            std::cout << "skip: MTP draft needs an artifact and a hidden fixture\n";
            return 77;
        }
        auto words = read_hidden(hidden);
        ninfer::DeviceBuffer conditioning(words.size() * 2);
        conditioning.copy_from_host(words.data(), words.size() * 2);
        ninfer::Tensor conditioning_view(conditioning.p, ninfer::DType::FP16,
                                         {2560, 4});
        flash::MtpDraft draft(artifact, 0);
        const ninfer::TokenId token = draft.run(conditioning_view, 504, 0);
        std::cout << "mtp draft token " << token << " expert_bytes " << draft.expert_bytes()
                  << " host_syncs " << draft.host_syncs() << "\n";
        if (token != 220) {
            std::cerr << "MTP draft token " << token << " does not match oracle 220\n";
            return 1;
        }
        if (draft.expert_bytes() > 2700000000ULL) {
            std::cerr << "MTP draft expert bank is still materialized in FP16: "
                      << draft.expert_bytes() << " bytes\n";
            return 1;
        }
        if (draft.host_syncs() > 1) {
            std::cerr << "MTP draft step performs " << draft.host_syncs()
                      << " host synchronizations, expected at most one for the router\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MTP draft failed: " << error.what() << "\n";
        return 1;
    }
}
