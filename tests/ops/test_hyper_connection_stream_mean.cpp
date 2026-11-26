#include "ninfer/ops/hyper_connection_stream_mean.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion kCriterion{.absolute = 2.0e-2, .relative = 1.5e-2};

int run_case(const char* label, DType dtype, int hidden, int streams, int tokens, int batch,
             int taps) {
    const std::size_t input_count =
        static_cast<std::size_t>(hidden) * streams * tokens * batch;
    std::vector<float> input(input_count);
    fill_uniform(input, 0xD1F1A5u + static_cast<std::uint32_t>(hidden * 31 + taps), -3.0F, 3.0F);
    if (dtype == DType::BF16) {
        round_to_bf16(input);
    } else {
        round_to_f16(input);
    }
    DeviceBuffer device_input =
        dtype == DType::BF16 ? to_device_bf16(input) : to_device_f16(input);

    const std::size_t output_count =
        static_cast<std::size_t>(taps) * hidden * tokens * batch;
    std::vector<float> sentinel(output_count, 12345.0F);
    if (dtype == DType::BF16) {
        round_to_bf16(sentinel);
    } else {
        round_to_f16(sentinel);
    }
    DeviceBuffer device_output =
        dtype == DType::BF16 ? to_device_bf16(sentinel) : to_device_f16(sentinel);

    Tensor in(device_input.p, dtype, {hidden, streams, tokens, batch});
    for (int tap = 0; tap < taps; ++tap) {
        Tensor out;
        out.data  = static_cast<std::byte*>(device_output.p) +
                   static_cast<std::size_t>(tap) * hidden * 2;
        out.dtype = dtype;
        out.ne[0] = hidden;
        out.ne[1] = tokens;
        out.ne[2] = batch;
        out.ne[3] = 1;
        out.nb[0] = 2;
        out.nb[1] = static_cast<std::int64_t>(taps) * hidden * 2;
        out.nb[2] = static_cast<std::int64_t>(taps) * hidden * tokens * 2;
        out.nb[3] = 0;
        ops::hyper_connection_stream_mean(in, out, nullptr);
    }
    cuda_synchronize();

    const std::vector<double> actual = dtype == DType::BF16
                                           ? from_device_bf16(device_output, output_count)
                                           : from_device_f16(device_output, output_count);
    std::vector<double> expected(actual.size(), 0.0);
    const double inverse = 1.0 / static_cast<double>(streams);
    for (int tap = 0; tap < taps; ++tap) {
        for (int h = 0; h < hidden; ++h) {
            for (int t = 0; t < tokens; ++t) {
                for (int b = 0; b < batch; ++b) {
                    double sum = 0.0;
                    for (int s = 0; s < streams; ++s) {
                        const std::size_t source =
                            static_cast<std::size_t>(h) +
                            static_cast<std::size_t>(hidden) *
                                (static_cast<std::size_t>(s) +
                                 static_cast<std::size_t>(streams) *
                                     (static_cast<std::size_t>(t) +
                                      static_cast<std::size_t>(tokens) *
                                          static_cast<std::size_t>(b)));
                        sum += static_cast<double>(input[source]);
                    }
                    const std::size_t destination =
                        static_cast<std::size_t>(tap) * hidden + static_cast<std::size_t>(h) +
                        static_cast<std::size_t>(t) * taps * hidden +
                        static_cast<std::size_t>(b) * taps * hidden * tokens;
                    expected[destination] = sum * inverse;
                }
            }
        }
    }
    return verify_pointwise(label, actual, expected, kCriterion);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "skip: no CUDA device\n";
        return 0;
    }
    int failures = 0;
    failures += run_case("stream mean bf16 H=8 S=4 T=3 B=2 taps=3", DType::BF16, 8, 4, 3, 2, 3);
    failures += run_case("stream mean bf16 H=5 S=4 T=1 B=1 taps=5", DType::BF16, 5, 4, 1, 1, 5);
    failures += run_case("stream mean fp16 H=6 S=4 T=2 B=1 taps=2", DType::FP16, 6, 4, 2, 1, 2);
    if (failures == 0) { std::cout << "hyper_connection_stream_mean: PASS\n"; }
    return failures == 0 ? 0 : 1;
}
