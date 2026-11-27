#include <ninfer/targets/qwen4_exp/frontend.h>

#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace {

using Json  = nlohmann::json;
using OJson = nlohmann::ordered_json;
namespace q4 = ninfer::targets::qwen4_exp;

constexpr const char* kArtifact =
    "/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer";

int check(bool ok, const char* what) {
    if (!ok) { std::cerr << "FAIL: " << what << "\n"; return 1; }
    return 0;
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open " + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

q4::FrontendResources resources_from_artifact(const char* path) {
    const ninfer::artifact::Reader reader(path);
    const auto take = [&](const char* object) {
        const auto* descriptor = reader.find(object);
        if (descriptor == nullptr) { throw std::runtime_error(std::string("missing ") + object); }
        const auto payload = reader.payload(*descriptor).data;
        return std::string(reinterpret_cast<const char*>(payload.data()), payload.size());
    };
    return q4::FrontendResources{
        .tokenizer_json           = take("resource/text/tokenizer.json"),
        .tokenizer_config_json    = take("resource/text/tokenizer_config.json"),
        .chat_template_jinja      = take("resource/text/chat_template.jinja"),
        .generation_config_json   = take("resource/text/generation_config.json"),
        .preprocessor_config_json = take("resource/vision/preprocessor_config.json"),
        .video_preprocessor_config_json =
            take("resource/vision/video_preprocessor_config.json"),
    };
}

ninfer::ChatRole parse_role(const std::string& value) {
    if (value == "system") { return ninfer::ChatRole::System; }
    if (value == "developer") { return ninfer::ChatRole::Developer; }
    if (value == "user") { return ninfer::ChatRole::User; }
    if (value == "assistant") { return ninfer::ChatRole::Assistant; }
    if (value == "tool") { return ninfer::ChatRole::Tool; }
    throw std::runtime_error("unknown chat role in oracle fixture: " + value);
}

ninfer::PromptInput input_from(const OJson& value) {
    ninfer::PromptInput input;
    for (const auto& raw : value.at("messages")) {
        ninfer::ChatMessage message;
        message.role = parse_role(raw.at("role").get<std::string>());
        ninfer::MessagePart part;
        part.kind = ninfer::MessagePartKind::Text;
        part.text = raw.at("content").get<std::string>();
        message.parts.push_back(std::move(part));
        if (raw.contains("tool_calls")) {
            for (const auto& call : raw.at("tool_calls")) {
                message.tool_calls.push_back(ninfer::ToolCall{
                    .id             = call.value("id", ""),
                    .name           = call.at("name").get<std::string>(),
                    .arguments_json = call.at("arguments").dump(),
                });
            }
        }
        input.messages.push_back(std::move(message));
    }
    const auto& options = value.at("options");
    input.options.add_generation_prompt = options.value("add_generation_prompt", true);
    input.options.enable_thinking       = options.value("enable_thinking", true);
    if (options.contains("reasoning_effort")) {
        const std::string effort = options.at("reasoning_effort").get<std::string>();
        if (effort == "low") {
            input.options.reasoning_effort = ninfer::ReasoningEffort::Low;
        } else if (effort == "medium") {
            input.options.reasoning_effort = ninfer::ReasoningEffort::Medium;
        } else {
            input.options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
        }
    }
    if (options.contains("tools")) {
        for (const auto& tool : options.at("tools")) { input.options.tool_jsons.push_back(tool.dump()); }
    }
    return input;
}

int run_cases(const q4::Frontend& frontend, const OJson& fixture) {
    int failures = 0;
    for (const auto& value : fixture.at("cases")) {
        const std::string name = value.at("name").get<std::string>();
        ninfer::PromptInput input = input_from(value);
        std::vector<ninfer::TokenId> expected;
        for (const auto id : value.at("token_ids")) { expected.push_back(id.get<ninfer::TokenId>()); }

        const auto prepared = frontend.prepare(input);
        const auto& data    = q4::PreparedPromptAccess::view(prepared);
        failures += check(data.token_ids == expected, ("token ids for " + name).c_str());
        failures += check(frontend.count_tokens(std::move(input)) == expected.size(),
                          ("count_tokens for " + name).c_str());
        failures += check(prepared.summary().prompt_tokens == expected.size(),
                          ("summary for " + name).c_str());
    }
    return failures;
}

int run_output_semantics(const q4::Frontend& frontend) {
    int failures = 0;
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "hello"});
    input.messages.push_back(std::move(user));
    const auto prepared = frontend.prepare(input);

    failures += check(q4::PreparedPromptAccess::view(prepared).starts_in_reasoning,
                      "thinking generation prompt starts in reasoning");
    const auto defaults = frontend.default_stop_policy();
    const bool has_eos = std::find(defaults.token_ids.begin(), defaults.token_ids.end(), 248046) !=
                         defaults.token_ids.end();
    failures += check(has_eos, "default stop policy carries <|im_end|>");

    auto session = frontend.make_output_session(prepared, {}, {});
    const ninfer::TokenId think[] = {248068}; // <think>
    auto decision = session.preview(think, 8, ninfer::FinishReason::OutputLimit);
    failures += check(decision.finish_reason == ninfer::FinishReason::None,
                      "thinking prologue is accepted");
    const auto think_delta = session.commit_preview();
    failures += check(think_delta.size() == 1 &&
                          think_delta.front().channel == ninfer::OutputChannel::Reasoning &&
                          think_delta.front().text == "<think>",
                      "<think> is published on the reasoning channel");
    failures += check(session.reasoning_tokens() == 1, "thinking token counted on reasoning");

    const ninfer::TokenId eos[] = {248046}; // <|im_end|>
    decision = session.preview(eos, 8, ninfer::FinishReason::OutputLimit);
    failures += check(decision.finish_reason == ninfer::FinishReason::StopToken,
                      "generation stops on <|im_end|>");
    failures += check(session.commit_preview().empty(), "stop token is not published");

    ninfer::PromptInput no_thinking = input;
    no_thinking.options.enable_thinking = false;
    const auto plain = frontend.prepare(no_thinking);
    failures += check(!q4::PreparedPromptAccess::view(plain).starts_in_reasoning,
                      "no-thinking prompt does not start in reasoning");

    const auto tokens = q4::PreparedPromptAccess::view(prepared).token_ids;
    const auto by_tokens = frontend.prepare_tokens(tokens);
    failures += check(q4::PreparedPromptAccess::view(by_tokens).token_ids == tokens,
                      "prepare_tokens keeps raw token ids");
    return failures;
}

} // namespace

int main() {
    if (!std::filesystem::exists(kArtifact)) {
        std::cout << "skip: Flash-Next artifact is not available\n";
        return 77;
    }
    const std::string fixture_path =
        std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/frontend/flash_next_chat_reference.json";
    if (!std::filesystem::exists(fixture_path)) {
        std::cout << "skip: Flash-Next oracle fixture is not available\n";
        return 77;
    }
    try {
        const auto resources = resources_from_artifact(kArtifact);
        const auto frontend  = q4::make_frontend(
            resources, q4::FrontendOptions{.vision_enabled = false, .max_context = 8192});
        const auto fixture = OJson::parse(read_file(fixture_path));
        int failures = run_cases(frontend, fixture);
        failures += run_output_semantics(frontend);
        if (failures != 0) { return 1; }
        std::cout << "qwen4_exp frontend chat reference and output semantics match\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4_exp frontend test threw: " << error.what() << "\n";
        return 1;
    }
}
