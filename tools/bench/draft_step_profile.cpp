#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/allreduce.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"

#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
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
        const std::uint32_t warmup = argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 8U;
        const std::uint32_t steps = argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 24U;
        if (artifact == nullptr || hidden == nullptr) {
            std::cerr << "draft_step_profile: artifact and hidden fixture are required\n";
            return 1;
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
            std::cerr << "draft_step_profile: two devices are required\n";
            return 1;
        }
        const auto words = read_hidden(hidden);
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
        draft.run_chain(conditioning_ranks, 504, warmup, 0, 0.0F);

        const auto begin = std::chrono::steady_clock::now();
        draft.run_chain(conditioning_ranks, 504, steps, 0, 0.0F);
        const auto end = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(end - begin).count();
        const double n = static_cast<double>(steps > 0 ? steps : 1U);
        const double wall_per_step = wall_ms / n;
        std::cout << "draft_step_profile steps " << steps << "\n";
        std::cout << "step_host_wall_ms_per_step " << wall_per_step << "\n";
        std::cout << "draft_tokens_per_second "
                  << (wall_per_step > 0.0 ? 1000.0 / wall_per_step : 0.0) << "\n";
        std::cout << "host_syncs_total " << draft.host_syncs() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "draft_step_profile: " << error.what() << '\n';
        return 1;
    }
}
