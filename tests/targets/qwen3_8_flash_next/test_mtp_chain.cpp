#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

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
    const auto expected = 2U * static_cast<std::size_t>(flash::MtpDraft::hidden_elements);
    if (bytes.size() != expected) {
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
            std::cout << "skip: MTP chain needs an artifact and a hidden fixture\n";
            return 77;
        }
        auto words = read_hidden(hidden);
        ninfer::DeviceBuffer conditioning(words.size() * 2);
        conditioning.copy_from_host(words.data(), words.size() * 2);
        ninfer::Tensor conditioning_view(conditioning.p, ninfer::DType::FP16, {2560, 4});
        flash::MtpDraft draft(artifact, 0);
        const std::vector<ninfer::TokenId> tokens =
            flash::draft_chain(draft, conditioning_view, 504, 3);
        std::cout << "mtp chain tokens";
        for (ninfer::TokenId token : tokens) { std::cout << ' ' << token; }
        std::cout << "\n";
        const std::vector<ninfer::TokenId> oracle{220, 18, 220};
        if (tokens != oracle) {
            std::cerr << "MTP chain tokens do not match FP64 oracle {220, 18, 220}\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MTP chain failed: " << error.what() << "\n";
        return 1;
    }
}
