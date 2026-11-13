#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include <memory>
#include <variant>

namespace ninfer {

struct DeviceContext;
struct ExecutionContext;

namespace targets {

using Qwen3_6_27B          = qwen3_6_27b::Package;
using Qwen3_6_35BA3B       = qwen3_6_35b_a3b::Package;
using Qwen3_8_FlashNext    = qwen3_8_flash_next::Package;

struct LoadedQwen3_6_27B {
    std::unique_ptr<Qwen3_6_27B::LoadedModel> model;
    Qwen3_6_27B::Frontend frontend;

    LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                      const EngineOptions& options);
    ~LoadedQwen3_6_27B();

    LoadedQwen3_6_27B(const LoadedQwen3_6_27B&)            = delete;
    LoadedQwen3_6_27B& operator=(const LoadedQwen3_6_27B&) = delete;
};

struct Qwen3_6_27BInstance {
    using Package = Qwen3_6_27B;

    std::unique_ptr<LoadedQwen3_6_27B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_27B::Program> program;

    Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                        runtime::KvCapacityResolution resolution,
                        Qwen3_6_27B::SequencePlan sequence_plan, ExecutionContext& execution,
                        const EngineOptions& options);
    ~Qwen3_6_27BInstance();

    Qwen3_6_27BInstance(const Qwen3_6_27BInstance&)            = delete;
    Qwen3_6_27BInstance& operator=(const Qwen3_6_27BInstance&) = delete;
};

struct LoadedQwen3_6_35BA3B {
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> model;
    Qwen3_6_35BA3B::Frontend frontend;

    LoadedQwen3_6_35BA3B(std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model,
                         const EngineOptions& options);
    ~LoadedQwen3_6_35BA3B();

    LoadedQwen3_6_35BA3B(const LoadedQwen3_6_35BA3B&)            = delete;
    LoadedQwen3_6_35BA3B& operator=(const LoadedQwen3_6_35BA3B&) = delete;
};

struct Qwen3_6_35BA3BInstance {
    using Package = Qwen3_6_35BA3B;

    std::unique_ptr<LoadedQwen3_6_35BA3B> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_6_35BA3B::Program> program;

    Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                           runtime::KvCapacityResolution resolution,
                           Qwen3_6_35BA3B::SequencePlan sequence_plan, ExecutionContext& execution,
                           const EngineOptions& options);
    ~Qwen3_6_35BA3BInstance();

    Qwen3_6_35BA3BInstance(const Qwen3_6_35BA3BInstance&)            = delete;
    Qwen3_6_35BA3BInstance& operator=(const Qwen3_6_35BA3BInstance&) = delete;
};

struct LoadedQwen3_8_FlashNext {
    std::unique_ptr<Qwen3_8_FlashNext::LoadedModel> model;
    Qwen3_8_FlashNext::Frontend frontend;

    LoadedQwen3_8_FlashNext(std::unique_ptr<Qwen3_8_FlashNext::LoadedModel> stable_model,
                            const EngineOptions& options);
    ~LoadedQwen3_8_FlashNext();

    LoadedQwen3_8_FlashNext(const LoadedQwen3_8_FlashNext&)            = delete;
    LoadedQwen3_8_FlashNext& operator=(const LoadedQwen3_8_FlashNext&) = delete;
};

struct Qwen3_8_FlashNextInstance {
    using Package = Qwen3_8_FlashNext;

    std::unique_ptr<LoadedQwen3_8_FlashNext> loaded;
    runtime::KvCapacityResolution kv_capacity_resolution;
    runtime::RequestMemory request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<Qwen3_8_FlashNext::Program> program;

    Qwen3_8_FlashNextInstance(std::unique_ptr<LoadedQwen3_8_FlashNext> stable_loaded,
                              runtime::KvCapacityResolution resolution,
                              Qwen3_8_FlashNext::SequencePlan sequence_plan,
                              ExecutionContext& execution, const EngineOptions& options);
    ~Qwen3_8_FlashNextInstance();

    Qwen3_8_FlashNextInstance(const Qwen3_8_FlashNextInstance&)            = delete;
    Qwen3_8_FlashNextInstance& operator=(const Qwen3_8_FlashNextInstance&) = delete;
};

using ActiveTarget = std::variant<std::unique_ptr<Qwen3_6_27BInstance>,
                                  std::unique_ptr<Qwen3_6_35BA3BInstance>,
                                  std::unique_ptr<Qwen3_8_FlashNextInstance>>;

struct ConstructedTarget {
    ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
};

[[nodiscard]] ConstructedTarget construct_target(const EngineOptions& options,
                                                 ExecutionContext& execution,
                                                 SharedModel& shared);

} // namespace targets
} // namespace ninfer
