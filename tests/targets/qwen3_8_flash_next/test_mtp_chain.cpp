#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "targets/qwen3_8_flash_next/impl/diag_switches.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <chrono>
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
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            std::cout << "skip: TP2 MTP chain needs two devices\n";
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
        constexpr std::uint32_t steps = 3;
        const auto begin = std::chrono::steady_clock::now();
        const std::vector<ninfer::TokenId> tokens =
            flash::draft_chain(draft, conditioning_ranks, 504, steps);
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - begin).count();
        std::cout << "mtp chain tokens";
        for (ninfer::TokenId token : tokens) { std::cout << ' ' << token; }
        std::cout << "\n";
        std::cout << "chain_ms " << milliseconds << " moe_host_syncs_total "
                  << draft.host_syncs() << " moe_host_syncs_per_step "
                  << (draft.host_syncs() / steps) << "\n";
        // The Q4 proposal/head is the accepted default: changing draft proposals is expected
        // semantics, and correctness of committed output is covered by verify and by the greedy
        // equivalence test. NINFER_DIAG_FP16_DRAFT_HEAD=1 keeps the original FP16 A/B oracle.
        const bool fp16 = flash::diag_fp16_draft_head();
        const std::vector<ninfer::TokenId> oracle =
            fp16 ? std::vector<ninfer::TokenId>{220, 27967, 813}
                 : std::vector<ninfer::TokenId>{220, 27967, 53};
        if (tokens != oracle) {
            std::cerr << "MTP chain tokens do not match the " << (fp16 ? "FP16" : "Q4")
                      << " oracle\n";
            return 1;
        }
        if (draft.host_syncs() / steps > 1) {
            std::cerr << "MTP draft chain performs " << (draft.host_syncs() / steps)
                      << " host synchronizations per step, expected at most one for the router\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MTP chain failed: " << error.what() << "\n";
        return 1;
    }
}
