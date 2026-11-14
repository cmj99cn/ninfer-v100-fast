#include <ninfer/targets/qwen4_exp/frontend.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen4_exp {

PreparedPrompt::PreparedPrompt() noexcept = default;
PreparedPrompt::~PreparedPrompt()         = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PromptSummary PreparedPrompt::summary() const {
    PromptSummary summary;
    if (data_ != nullptr) {
        summary.prompt_tokens = static_cast<std::uint32_t>(data_->token_ids.size());
    }
    return summary;
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept { return {}; }

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

runtime::OutputDecision OutputSession::preview(std::span<const TokenId> tokens,
                                                std::uint32_t budget_remaining,
                                                FinishReason limit_reason) {
    runtime::OutputDecision decision;
    for (const TokenId token : tokens) {
        if (decision.accepted_tokens == budget_remaining) {
            decision.finish_reason = limit_reason;
            return decision;
        }
        ++decision.accepted_tokens;
        if (std::find(stop_token_ids_.begin(), stop_token_ids_.end(), token) !=
            stop_token_ids_.end()) {
            decision.finish_reason = FinishReason::StopToken;
            return decision;
        }
    }
    if (decision.accepted_tokens == budget_remaining) {
        decision.finish_reason = limit_reason;
    }
    return decision;
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    runtime::OutputDecision decision;
    decision.finish_reason = reason;
    return decision;
}

std::vector<OutputDelta> OutputSession::commit_preview() { return {}; }

std::uint32_t OutputSession::reasoning_tokens() const { return 0; }

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    (void)input;
    (void)control;
    throw std::logic_error("qwen4_exp family frontend does not prepare chat-template input");
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    (void)input;
    (void)control;
    throw std::logic_error("qwen4_exp family frontend does not count chat-template input");
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    (void)allow_prefix_identity;
    auto data = std::make_unique<PreparedPromptData>();
    data->token_ids = std::move(token_ids);
    return PreparedPrompt(std::move(data));
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept { return {}; }

MediaCacheSummary Frontend::media_cache_summary() const { return {}; }

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output) const {
    (void)prompt;
    (void)output;
    OutputSession session;
    session.stop_token_ids_ = caller_stop.token_ids;
    return session;
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    static const StopPolicy policy;
    return policy;
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("PreparedPrompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::logic_error("PreparedPrompt is empty"); }
    return std::move(*prompt.data_);
}

Frontend make_frontend(const FrontendResources& resources, FrontendOptions options) {
    Frontend frontend;
    frontend.resources_ = resources;
    frontend.options_   = options;
    return frontend;
}

namespace {

artifact::ObjectHandle require_component_resource(artifact::Binder& binder,
                                                  const std::string& component,
                                                  const std::string& role) {
    const auto& resources = binder.directory().component_resources;
    const auto component_it = resources.find(component);
    if (component_it == resources.end()) {
        throw std::invalid_argument("the artifact has no frontend component resources for " + component);
    }
    const auto role_it = component_it->second.find(role);
    if (role_it == component_it->second.end()) {
        throw std::invalid_argument("the artifact has no frontend resource role " + role);
    }
    const artifact::ObjectHandle handle =
        binder.require_resource(role_it->second, artifact::ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

std::string take_string(artifact::MaterializedArtifact& materialized, artifact::ObjectHandle handle) {
    const auto bytes = materialized.take_resource_bytes(handle);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

} // namespace

FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder) {
    return FrontendResourcePlan{
        .tokenizer_json           = require_component_resource(binder, "text", "tokenizer.json"),
        .tokenizer_config_json    = require_component_resource(binder, "text", "tokenizer_config.json"),
        .chat_template_jinja      = require_component_resource(binder, "text", "chat_template.jinja"),
        .generation_config_json   = require_component_resource(binder, "text", "generation_config.json"),
        .preprocessor_config_json = require_component_resource(binder, "vision", "preprocessor_config.json"),
        .video_preprocessor_config_json =
            require_component_resource(binder, "vision", "video_preprocessor_config.json"),
    };
}

FrontendResources take_frontend_resources(artifact::MaterializedArtifact& materialized,
                                          const FrontendResourcePlan& plan) {
    return FrontendResources{
        .tokenizer_json           = take_string(materialized, plan.tokenizer_json),
        .tokenizer_config_json    = take_string(materialized, plan.tokenizer_config_json),
        .chat_template_jinja      = take_string(materialized, plan.chat_template_jinja),
        .generation_config_json   = take_string(materialized, plan.generation_config_json),
        .preprocessor_config_json = take_string(materialized, plan.preprocessor_config_json),
        .video_preprocessor_config_json =
            take_string(materialized, plan.video_preprocessor_config_json),
    };
}

} // namespace ninfer::targets::qwen4_exp
