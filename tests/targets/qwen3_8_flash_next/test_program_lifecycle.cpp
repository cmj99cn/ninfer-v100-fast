#include "fixture.h"

#include "core/device.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/roles.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer_flash_next_fixture::TemporaryArtifact;

using ninfer::artifact::Binder;
using ninfer::artifact::MaterializedArtifact;
using ninfer::artifact::Reader;

namespace roles = ninfer::targets::qwen3_8_flash_next::detail::roles;
using Package = ninfer::targets::qwen3_8_flash_next::Package;

constexpr std::uint32_t kVocab = 248320;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool has_frontend_resources(const Binder& binder) {
    return ninfer_flash_next_fixture::has_frontend_resources(binder);
}

void expect_throw(const std::function<void()>& function, const std::string& message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

template <class T>
concept HasModelId = requires(T value) { value.model_id; };

static_assert(!HasModelId<Package::SequencePlan>);
static_assert(!HasModelId<Package::SequencePlanner>);
static_assert(!HasModelId<Package::RequestBasePlan>);
static_assert(!HasModelId<Package::RequestPlan>);
static_assert(!HasModelId<Package::Program>);
static_assert(!HasModelId<Package::PreparedPrompt>);
static_assert(!HasModelId<Package::PreparedPromptData>);

static_assert(sizeof(Package::SequencePlan) == sizeof(void*));
static_assert(sizeof(Package::Program) == sizeof(void*));

class HostProgram {
public:
    explicit HostProgram(std::uint32_t max_context, const Package::WeightsProfile profile)
        : execution({0}) {
        auto artifact = ninfer_flash_next_fixture::build_flash_next_fixture();
        const auto path = ninfer_flash_next_fixture::write_flash_next_fixture(artifact);
        const TemporaryArtifact temporary(path);

        Reader reader(path);
        Binder binder(reader, 1);

        ninfer::EngineOptions options;
        options.device = execution.primary().device;
        options.tp = 1;
        options.max_context = max_context;
        options.max_concurrency = 1;
        options.prefill_chunk = 64;

        auto load_plan = Package::plan_load(binder, options, profile);
        model = Package::construct_loaded_model(std::move(load_plan), MaterializedArtifact{});
        frontend = Package::make_frontend(*model, options);

        auto planner = Package::make_sequence_planner(execution.primary(), options, profile);
        auto sequence_plan = std::move(planner).finalize(planner.capacity_curve().minimum_main_page_groups);
        program = Package::create_program(*model, std::move(sequence_plan), execution, options);
    }

    [[nodiscard]] Package::Program& program_ref() { return *program; }
    [[nodiscard]] Package::PreparedPrompt make_prompt(std::vector<ninfer::TokenId> tokens) const {
        return frontend.prepare_tokens(std::move(tokens));
    }

private:
    ninfer::ExecutionContext execution;
    std::unique_ptr<Package::LoadedModel> model;
    Package::Frontend frontend;
    std::unique_ptr<Package::Program> program;
};

void test_capacity_curve(ninfer::DeviceContext& device, const Package::WeightsProfile profile) {
    ninfer::EngineOptions small;
    small.device = device.device;
    small.tp = 1;
    small.max_context = 1024;
    small.max_concurrency = 1;
    small.prefill_chunk = 64;

    ninfer::EngineOptions large;
    large.device = device.device;
    large.tp = 1;
    large.max_context = 2048;
    large.max_concurrency = 1;
    large.prefill_chunk = 64;

    auto small_planner = Package::make_sequence_planner(device, small, profile);
    auto large_planner = Package::make_sequence_planner(device, large, profile);

    const auto& small_curve = small_planner.capacity_curve();
    const auto& large_curve = large_planner.capacity_curve();

    require(small_curve.main_page_tokens != 0, "the capacity curve has no page size");
    require(small_curve.minimum_main_page_groups != 0, "the capacity curve has no minimum group");
    require(small_curve.minimum_device_reservation_bytes != 0,
            "the capacity curve has no minimum reservation");
    require(small_curve.bytes_per_additional_main_page_group != 0,
            "the capacity curve has no page-group stride");

    require(large_curve.minimum_main_page_groups > small_curve.minimum_main_page_groups,
            "the capacity curve is not monotonic in requested max_context");
    require(large_curve.minimum_device_reservation_bytes > small_curve.minimum_device_reservation_bytes,
            "the capacity reservation is not monotonic in requested max_context");

    auto small_plan = std::move(small_planner).finalize(small_curve.minimum_main_page_groups);
    auto large_plan = std::move(large_planner).finalize(large_curve.minimum_main_page_groups);

    require(small_plan.capacity() != 0, "the finalized plan has no capacity");
    require(small_plan.kv_capacity() != 0, "the finalized plan has no KV capacity");
    require(small_plan.max_concurrency() != 0, "the finalized plan has no concurrency");
    require(small_plan.device_reservation_bytes() != 0, "the finalized plan has no device reservation");
    require(small_plan.workspace_capacity_bytes() != 0, "the finalized plan has no workspace capacity");
    require(small_plan.request_transient_capacity_bytes() != 0,
            "the finalized plan has no request transient capacity");

    require(large_plan.capacity() >= small_plan.capacity(),
            "the finalized capacity is not monotonic");
    require(large_plan.kv_capacity() >= small_plan.kv_capacity(),
            "the finalized KV capacity is not monotonic");
    require(large_plan.device_reservation_bytes() > small_plan.device_reservation_bytes(),
            "the finalized device reservation is not monotonic");
    require(large_plan.workspace_capacity_bytes() >= small_plan.workspace_capacity_bytes(),
            "the finalized workspace capacity is not monotonic");

    std::cout << "flash-next lifecycle capacity: " << small_plan.capacity() << " / "
              << large_plan.capacity() << " tokens\n";
}

void test_rejections(HostProgram& host) {
    auto& program = host.program_ref();

    bool rejected = false;
    try {
        auto prompt = host.make_prompt(std::vector<ninfer::TokenId>(65, 1));
        ninfer::runtime::ResolvedExecutionOptions request_options;
        (void)program.plan_request_base(prompt, request_options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "the family accepted a request beyond its finalized capacity");

    rejected = false;
    try {
        auto prompt = host.make_prompt({kVocab});
        ninfer::runtime::ResolvedExecutionOptions request_options;
        (void)program.plan_request_base(prompt, request_options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "the family accepted a token outside the vocabulary");

    std::cout << "flash-next lifecycle rejections: capacity and token domain\n";
}

void test_lane_lifecycle(HostProgram& host) {
    auto& program = host.program_ref();

    const std::vector<ninfer::TokenId> prompt_tokens(65, 1);
    auto prompt = host.make_prompt(prompt_tokens);
    ninfer::runtime::ResolvedExecutionOptions request_options;
    auto base = program.plan_request_base(prompt, request_options);
    require(base.summary().prompt_tokens == 65, "the request plan did not preserve the prompt length");

    auto lane_plan = program.plan_request_for_lane(0, prompt, base);
    require(program.admission_capacity().active_lanes == 1,
            "the family admission capacity does not expose its concurrency");
    require(program.can_admit_lane(0, lane_plan), "the family cannot admit an idle lane");

    ninfer::runtime::TransientRegion transient;
    auto first = program.start_prefill_lane(0, std::move(prompt), std::move(lane_plan),
                                            std::move(transient));
    require(first.processed_prompt_tokens > 0, "start_prefill_lane did not process a chunk");
    require(first.round.tokens.empty(), "start_prefill_lane produced a fabricated token");
    require(!first.complete, "start_prefill_lane completed a prompt that still has unprocessed tokens");

    auto second = program.advance_prefill_lane(0);
    require(second.complete, "advance_prefill_lane did not complete the short prompt");
    require(second.round.tokens.empty(), "advance_prefill_lane produced a fabricated token");

    const std::vector<std::uint32_t> lanes{0};
    const std::vector<ninfer::runtime::RoundBudget> budgets{
        ninfer::runtime::RoundBudget{.generated_tokens_remaining = 1}};
    expect_throw([&] { (void)program.decode_batch(lanes, budgets); },
                 "decode_batch did not reach the package post-mixer leaf");

    program.abort_lane(0);
    auto after_prompt = host.make_prompt(prompt_tokens);
    auto after_plan = program.plan_request_for_lane(0, after_prompt, base);
    require(program.can_admit_lane(0, after_plan), "abort_lane did not release the lane");

    std::cout << "flash-next lifecycle lane: prefill, decode, abort\n";
}

int test_real_artifact(const char* path, const Package::WeightsProfile profile) {
    Reader reader(path);
    Binder binder(reader, 1);
    if (!has_frontend_resources(binder)) {
        std::cout << "skip: the real artifact does not carry the frontend resources\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.tp = 1;

    auto plan = Package::plan_load(binder, options, profile);
    require(plan.bindings().parameters().size() == roles::kTextNameCount,
            "the real artifact does not cover the flash-next text role table");
    require(plan.materialization().device_count == 1,
            "the real artifact materialization is not a one-device plan");
    require(plan.materialization().object_count >= plan.bindings().objects.size(),
            "the real artifact materialization does not cover the binding objects");

    std::cout << "flash-next lifecycle real artifact: " << plan.materialization().object_count
              << " materialized objects\n";
    return 0;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "skip: no CUDA device is available\n";
        return 77;
    }

    try {
        const auto profile = Package::WeightsProfile::Nvfp4;

        ninfer::DeviceContext device(0);
        test_capacity_curve(device, profile);

        HostProgram small(64, profile);
        test_rejections(small);

        HostProgram large(128, profile);
        test_lane_lifecycle(large);

        const char* real_artifact = std::getenv("NINFER_FLASH_NEXT_ARTIFACT");
        if (real_artifact != nullptr && *real_artifact != '\0') {
            const int status = test_real_artifact(real_artifact, profile);
            if (status != 0) { return status; }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
