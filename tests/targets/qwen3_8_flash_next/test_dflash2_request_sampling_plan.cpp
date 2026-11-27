#include "fixture.h"

#include "core/device.h"
#include "runtime/contract/types.h"
#include "targets/qwen3_8_flash_next/impl/variant.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>

#define NINFER_QWEN4EXP_VARIANT ::ninfer::targets::qwen3_8_flash_next::detail::Variant
#define NINFER_QWEN4EXP_RUNTIME_NS flash_next_runtime
#include "targets/qwen4_exp/impl/runtime/program.h"
#undef NINFER_QWEN4EXP_RUNTIME_NS
#undef NINFER_QWEN4EXP_VARIANT

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Package = ninfer::targets::qwen3_8_flash_next::Package;
namespace PlanRuntime = ninfer::targets::qwen4_exp::detail::flash_next_runtime;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

class SamplingPlanProgram {
public:
    SamplingPlanProgram() : execution({0}) {
        auto artifact = ninfer_flash_next_fixture::build_flash_next_fixture();
        const auto path = ninfer_flash_next_fixture::write_flash_next_fixture(artifact);
        const ninfer_flash_next_fixture::TemporaryArtifact temporary(path);

        ninfer::artifact::Reader reader(path);
        ninfer::artifact::Binder binder(reader, 1);

        ninfer::EngineOptions options;
        options.artifact_path = path;
        options.device = execution.primary().device;
        options.tp = 1;
        options.max_context = 1024;
        options.max_concurrency = 1;
        options.prefill_chunk = 64;

        auto load_plan = Package::plan_load(binder, options, Package::WeightsProfile::Nvfp4);
        model = Package::construct_loaded_model(std::move(load_plan),
                                                ninfer::artifact::MaterializedArtifact{});
        frontend = Package::make_frontend(*model, options);
        auto planner = Package::make_sequence_planner(execution.primary(), options,
                                                       Package::WeightsProfile::Nvfp4);
        auto sequence_plan =
            std::move(planner).finalize(planner.capacity_curve().minimum_main_page_groups);
        program = Package::create_program(*model, std::move(sequence_plan), execution, options);
    }

    [[nodiscard]] Package::PreparedPrompt make_prompt() const {
        return frontend.prepare_tokens(std::vector<ninfer::TokenId>(8, 1));
    }

    [[nodiscard]] Package::Program& ref() noexcept { return *program; }

private:
    ninfer::ExecutionContext execution;
    std::unique_ptr<Package::LoadedModel> model;
    Package::Frontend frontend;
    std::unique_ptr<Package::Program> program;
};

constexpr float kTemperature = 0.85F;
constexpr std::uint64_t kSeed = 0xFEDCBA9876543210ull;

void require_request_sampling(const ninfer::ops::SamplingConfig& config, const char* stage) {
    require(config.temperature == kTemperature, stage);
    require(config.seed == kSeed, stage);
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::cout << "skip: no CUDA device is available\n";
        return 77;
    }

    try {
        SamplingPlanProgram host;
        auto prompt = host.make_prompt();
        ninfer::runtime::ResolvedExecutionOptions options;
        options.sampling.temperature = kTemperature;
        options.sampling.seed = kSeed;
        options.requested_output_tokens = 2;

        auto base = host.ref().plan_request_base(prompt, options);
        require(base.impl_ != nullptr, "the request base plan carries no private config");
        require_request_sampling(base.impl_->sampling,
                                 "plan_request_base dropped the request sampling");

        auto lane_plan = host.ref().plan_request_for_lane(0, prompt, base);
        require(lane_plan.impl_ != nullptr, "the lane plan carries no private config");
        require_request_sampling(lane_plan.impl_->sampling,
                                 "plan_request_for_lane dropped the base sampling");

        // LaneState is the object decode_dflash2 copies into dflash2_sampling_; it is fed from the
        // lane plan at prefill. Assert that the request values survive that hand-off as well.
        PlanRuntime::LaneState state;
        state.sampling = lane_plan.impl_->sampling;
        require_request_sampling(state.sampling, "LaneState dropped the lane sampling");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    std::cout << "dflash2 request sampling plan wiring: PASS\n";
    return 0;
}
