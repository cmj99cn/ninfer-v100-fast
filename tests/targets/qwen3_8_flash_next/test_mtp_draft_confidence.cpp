#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include "targets/qwen3_8_flash_next/impl/config.h"
#include "targets/qwen3_8_flash_next/impl/diag_switches.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft.h"
#include "targets/qwen3_8_flash_next/impl/mtp_draft_kernels.h"
#include "targets/qwen3_8_flash_next/impl/speculative.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace flash = ninfer::targets::qwen3_8_flash_next::detail;

namespace {

constexpr int kVocab = ninfer::targets::qwen3_8_flash_next::detail::TextConfig::vocab;

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

// Independent FP64 softmax oracle over the represented FP16 logits. The top-1 probability is
// exp(max) / sum_v exp(v) = 1 / sum_v exp(v - max).
double top1_oracle(const std::vector<std::uint16_t>& words) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint16_t word : words) {
        maximum = std::max(maximum, static_cast<double>(__half2float(__ushort_as_half(word))));
    }
    double denominator = 0.0;
    for (std::uint16_t word : words) {
        const double value = static_cast<double>(__half2float(__ushort_as_half(word)));
        denominator += std::exp(value - maximum);
    }
    return 1.0 / denominator;
}

int check_kernel_oracle() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::cout << "skip: top-1 probability kernel needs a device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    struct Case {
        const char* name;
        std::vector<std::uint16_t> words;
    };
    std::vector<Case> cases;
    {
        Case flat{"flat", std::vector<std::uint16_t>(kVocab, __half_as_ushort(__float2half_rn(0.0F)))};
        cases.push_back(std::move(flat));
    }
    {
        Case sharp{"sharp", std::vector<std::uint16_t>(kVocab, __half_as_ushort(__float2half_rn(-8.0F)))};
        sharp.words[12345] = __half_as_ushort(__float2half_rn(12.0F));
        cases.push_back(std::move(sharp));
    }
    {
        Case spread{"spread", {}};
        spread.words.reserve(kVocab);
        for (int i = 0; i < kVocab; ++i) {
            const float value = -4.0F + static_cast<float>(i % 31) * 0.125F;
            spread.words.push_back(__half_as_ushort(__float2half_rn(value)));
        }
        cases.push_back(std::move(spread));
    }

    ninfer::DeviceBuffer logits(static_cast<std::size_t>(kVocab) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer probability(sizeof(float));
    int failures = 0;
    for (const Case& test_case : cases) {
        logits.copy_from_host(test_case.words.data(), logits.bytes);
        flash::mtp_top1_prob_f16(logits.p, static_cast<float*>(probability.p), kVocab, stream);
        float got = 0.0F;
        CUDA_CHECK(cudaMemcpyAsync(&got, probability.p, sizeof(got), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const double expected = top1_oracle(test_case.words);
        const double error = std::fabs(static_cast<double>(got) - expected);
        const double tolerance = 1.0e-4 * expected + 1.0e-7;
        std::cout << "top1 kernel " << test_case.name << " got " << got << " oracle " << expected
                  << " error " << error << "\n";
        if (!(got > 0.0F && got <= 1.0F && error <= tolerance)) {
            std::cerr << "top-1 probability kernel disagrees with FP64 oracle on "
                      << test_case.name << "\n";
            ++failures;
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures == 0 ? 0 : 1;
}

std::vector<float> chain_confidences(flash::MtpDraft& draft,
                                     const flash::MtpDraft::RankTensors& conditioning,
                                     ninfer::TokenId first, std::uint32_t steps,
                                     std::int32_t start_position) {
    const std::size_t bytes = flash::MtpDraft::hidden_elements * sizeof(std::uint16_t);
    ninfer::DeviceBuffer current(bytes);
    ninfer::DeviceBuffer next(bytes);
    CUDA_CHECK(cudaSetDevice(draft.device()));
    CUDA_CHECK(cudaMemcpy(current.p, conditioning[0].data, bytes, cudaMemcpyDeviceToDevice));

    std::vector<float> confidences;
    void* src = current.p;
    void* dst = next.p;
    for (std::uint32_t step = 0; step < steps; ++step) {
        flash::MtpDraft::RankTensors current_views{};
        flash::MtpDraft::RankTensors next_views{};
        current_views[0] = ninfer::Tensor(src, ninfer::DType::FP16, {2560, 4});
        next_views[0]    = ninfer::Tensor(dst, ninfer::DType::FP16, {2560, 4});
        float confidence = -1.0F;
        first = draft.run(current_views, first, start_position + static_cast<std::int32_t>(step),
                          &next_views, &confidence);
        confidences.push_back(confidence);
        std::swap(src, dst);
    }
    return confidences;
}

int check_integration(const char* artifact, const char* hidden) {
    if (artifact == nullptr || hidden == nullptr) {
        std::cout << "skip: MTP draft confidence needs an artifact and a hidden fixture\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: TP2 MTP confidence needs two devices\n";
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

    float confidence = -1.0F;
    const ninfer::TokenId with_confidence = draft.run(conditioning_ranks, 504, 0, nullptr,
                                                       &confidence);
    const ninfer::TokenId without_confidence = draft.run(conditioning_ranks, 504, 0);
    if (with_confidence != 220 || without_confidence != 220) {
        std::cerr << "draft token changed: with_conf " << with_confidence << " without_conf "
                  << without_confidence << " (oracle 220)\n";
        return 1;
    }
    if (!(confidence > 0.0F && confidence <= 1.0F)) {
        std::cerr << "draft top-1 probability out of range: " << confidence << "\n";
        return 1;
    }

    // The Q4 proposal/head is the accepted default: different draft proposals and per-step
    // confidences are expected, while verify and greedy equivalence guarantee committed output.
    // NINFER_DIAG_FP16_DRAFT_HEAD=1 retains the original FP16 A/B oracle.
    const bool fp16 = flash::diag_fp16_draft_head();
    const std::vector<ninfer::TokenId> oracle =
        fp16 ? std::vector<ninfer::TokenId>{220, 27967, 813}
             : std::vector<ninfer::TokenId>{220, 27967, 53};
    std::cout << "draft head " << (fp16 ? "fp16" : "q4") << "\n";
    const std::vector<float> confidences = chain_confidences(draft, conditioning_ranks, 504, 3, 0);
    std::cout << "draft confidences";
    for (float value : confidences) { std::cout << ' ' << value; }
    std::cout << "\n";
    for (float value : confidences) {
        if (!(value > 0.0F && value <= 1.0F)) {
            std::cerr << "chain step probability out of range: " << value << "\n";
            return 1;
        }
    }

    const float configured = flash::draft_min_p();
    std::size_t expected_default = confidences.size();
    if (configured > 0.0F) {
        for (std::size_t step = 0; step < confidences.size(); ++step) {
            if (confidences[step] < configured) {
                expected_default = step + 1;
                break;
            }
        }
    }
    const auto full = flash::draft_chain(draft, conditioning_ranks, 504, 3);
    const std::vector<ninfer::TokenId> expected_tokens(
        oracle.begin(), oracle.begin() + static_cast<std::ptrdiff_t>(expected_default));
    std::cout << "configured draft_min_p " << configured << " default chain size "
              << full.size() << " (expected " << expected_default << ")\n";
    if (full != expected_tokens) {
        std::cerr << "configured draft_min_p is not reflected by draft_chain\n";
        return 1;
    }
    const auto disabled = draft.run_chain(conditioning_ranks, 504, 3, 0, 0.0F);
    if (disabled != oracle) {
        std::cerr << "explicit threshold 0 changed the chain\n";
        return 1;
    }
    const auto terminated = draft.run_chain(conditioning_ranks, 504, 3, 0, 2.0F);
    if (terminated != std::vector<ninfer::TokenId>{220}) {
        std::cerr << "threshold above probability should keep exactly the first draft, got "
                  << terminated.size() << " tokens\n";
        return 1;
    }

    const float mid = confidences[1] + 1.0e-6F;
    const std::size_t expected = confidences[0] >= mid ? 2U : 1U;
    const auto stopped = draft.run_chain(conditioning_ranks, 504, 3, 0, mid);
    std::cout << "threshold " << mid << " stopped after " << stopped.size()
              << " drafts (expected " << expected << ")\n";
    if (stopped.size() != expected) {
        std::cerr << "per-step confidence termination disagrees with the observed probabilities\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int kernel_result = check_kernel_oracle();
        if (kernel_result != 0) { return kernel_result; }

        const char* artifact = argc > 1 ? argv[1] : std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        const char* hidden = argc > 2 ? argv[2] : std::getenv("NINFER_MTP_HIDDEN");
        return check_integration(artifact, hidden);
    } catch (const std::exception& error) {
        std::cerr << "MTP confidence test failed: " << error.what() << "\n";
        return 1;
    }
}
