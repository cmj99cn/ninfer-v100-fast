#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "runtime/contract/types.h"
#include <ninfer/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen4_exp {

struct FrontendOptions {
    bool vision_enabled                    = false;
    std::uint32_t max_context              = 2048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
};

struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle preprocessor_config_json;
    artifact::ObjectHandle video_preprocessor_config_json;
};

struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::uint64_t> block_hashes;
    bool starts_in_reasoning = false;
};

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class PreparedPromptAccess;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                   std::uint32_t budget_remaining,
                                                   FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] std::vector<OutputDelta> commit_preview();
    [[nodiscard]] std::uint32_t reasoning_tokens() const;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend() noexcept;
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                     const StopPolicy& caller_stop,
                                                     const OutputOptions& output = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

[[nodiscard]] FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder);
[[nodiscard]] FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                                        const FrontendResourcePlan& plan);

} // namespace ninfer::targets::qwen4_exp
