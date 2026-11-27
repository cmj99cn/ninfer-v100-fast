#include "targets/qwen4_exp/impl/runtime/sampling.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    ninfer::ResolvedSamplingParameters request;
    request.temperature       = 0.85F;
    request.top_k             = 7;
    request.top_p             = 0.91F;
    request.min_p             = 0.23F;
    request.presence_penalty  = 0.4F;
    request.frequency_penalty = -0.3F;
    request.seed              = 0xFEDCBA9876543210ull;

    const ninfer::ops::SamplingConfig config =
        ninfer::targets::qwen4_exp::translate_sampling(request);
    require(config.temperature == request.temperature, "temperature was not propagated");
    require(config.top_k == request.top_k, "top_k was not propagated");
    require(config.top_p == request.top_p, "top_p was not propagated");
    require(config.min_p == request.min_p, "min_p was not propagated");
    require(config.presence_penalty == request.presence_penalty,
            "presence_penalty was not propagated");
    require(config.frequency_penalty == request.frequency_penalty,
            "frequency_penalty was not propagated");
    require(config.seed == request.seed, "seed was not propagated");
    require(config.token_counts == nullptr, "token_counts must be null without a counter array");

    const ninfer::ops::SamplingConfig greedy =
        ninfer::targets::qwen4_exp::translate_sampling(ninfer::ResolvedSamplingParameters{});
    require(greedy.temperature == 0.0F, "the default request must stay greedy");
    require(greedy.seed == 0, "the default request must carry seed zero");
    require(greedy.token_counts == nullptr, "the default token_counts must be null");

    std::cout << "dflash2 request sampling: PASS\n";
    return 0;
}
