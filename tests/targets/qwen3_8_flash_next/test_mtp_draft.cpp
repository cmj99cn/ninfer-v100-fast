#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <array>
#include <cuda_runtime.h>
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
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            std::cout << "skip: TP2 MTP draft needs two devices\n";
            return 77;
        }
        auto words = read_hidden(hidden);
        ninfer::ExecutionContext execution({0, 1});
        ninfer::ops::CollectiveComms comms(execution);
        std::array<ninfer::DeviceBuffer, 2> conditioning;
        flash::MtpDraft::RankTensors conditioning_ranks{};
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        for (int rank = 0; rank < 2; ++rank) {
            const auto slot = static_cast<std::size_t>(rank);
            CUDA_CHECK(cudaSetDevice(execution.dev[slot]->device));
            conditioning[slot] = ninfer::DeviceBuffer(words.size() * 2);
            conditioning[slot].copy_from_host(words.data(), words.size() * 2);
            conditioning_ranks[slot] =
                ninfer::Tensor(conditioning[slot].p, ninfer::DType::FP16, {2560, 4});
        }
        CUDA_CHECK(cudaSetDevice(previous));
        flash::MtpDraft draft(artifact, execution, comms);
        const ninfer::TokenId token = draft.run(conditioning_ranks, 504, 0);
        std::cout << "mtp draft token " << token << " expert_bytes " << draft.expert_bytes()
                  << " host_syncs " << draft.host_syncs() << "\n";
        // The first draft token is identical under the accepted Q4 proposal/head default and the
        // FP16 A/B switch. Later steps legitimately differ; committed output is protected by
        // verify and by the greedy equivalence test.
        if (token != 220) {
            std::cerr << "MTP draft token " << token << " does not match oracle 220\n";
            return 1;
        }
        if (draft.expert_bytes() < 1200000000ULL || draft.expert_bytes() > 1300000000ULL) {
            std::cerr << "MTP draft per-rank expert bank has an unexpected size: "
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
