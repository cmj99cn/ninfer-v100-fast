"""Independent chat-render/tokenization oracle for the qwen4_exp frontend.

Reads the frontend resources straight from the registered Flash-Next artifact, renders the
artifact's own chat_template.jinja with Jinja2 (not the C++ renderer), and tokenizes the rendered
string with the official HuggingFace `tokenizers` implementation. The emitted fixture is consumed
by `tests/targets/qwen4_exp/test_frontend.cpp`.

    /home/cmj/eval-venv/bin/python tools/parity/qwen3_8_flash_next/frontend_reference.py \
        > tests/fixtures/frontend/flash_next_chat_reference.json
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from jinja2 import Environment
from tokenizers import Tokenizer

ARTIFACT = Path("/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer")
MAGIC_PREFIX_BYTES = 32
PAYLOAD_ALIGNMENT = 4096


def read_resources(path: Path) -> dict[str, bytes]:
    with path.open("rb") as stream:
        prefix = stream.read(MAGIC_PREFIX_BYTES)
        if prefix[:8] != b"NINFER\x00\x03":
            raise ValueError("expected a NInfer v3 artifact")
        json_bytes = struct.unpack("<Q", prefix[8:16])[0]
        directory = json.loads(stream.read(json_bytes))
        payload_start = ((MAGIC_PREFIX_BYTES + json_bytes + PAYLOAD_ALIGNMENT - 1) //
                         PAYLOAD_ALIGNMENT) * PAYLOAD_ALIGNMENT
        objects = {obj["id"]: obj for obj in directory["objects"]}
        stream.seek(payload_start)
        payload = stream.read()
    out: dict[str, bytes] = {}
    for component in directory["components"].values():
        for role, object_id in component.get("resources", {}).items():
            obj = objects[object_id]
            begin = obj["offset"]
            out[role] = payload[begin:begin + obj["bytes"]]
    return out


LOW = ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
       "to the conclusion without unnecessary elaboration.")


def cases() -> list[dict[str, object]]:
    lookup = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    return [
        {
            "name": "plain",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "options": {"add_generation_prompt": True, "enable_thinking": True},
        },
        {
            "name": "no_thinking",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>\nold\n</think>\n\nhi"},
                {"role": "user", "content": "again"},
            ],
            "options": {"add_generation_prompt": True, "enable_thinking": False},
        },
        {
            "name": "tools_low_effort",
            "messages": [
                {"role": "system", "content": "be exact"},
                {"role": "user", "content": "weather in Paris?"},
            ],
            "options": {
                "add_generation_prompt": True,
                "enable_thinking": True,
                "reasoning_effort": "low",
                "tools": [lookup],
            },
        },
        {
            "name": "tool_call_history",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "lookup", "arguments": {"city": "Paris", "n": 2,
                                                     "nested": {"x": [1, 2]}}},
                ]},
                {"role": "tool", "content": "sunny"},
                {"role": "tool", "content": "20C"},
            ],
            "options": {"add_generation_prompt": True, "enable_thinking": True},
        },
        {
            # Jinja's tojson filter HTML-escapes <, >, & and ' after JSON serialization. The tool
            # schema and a non-string tool argument both pass through that filter.
            "name": "tools_html_escape",
            "messages": [
                {"role": "system", "content": "be exact"},
                {"role": "user", "content": "search for <tag> & 'quotes'?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"name": "search", "arguments": {
                        "nested": {"query<": ["a&", "b>", "it's"]},
                    }},
                ]},
            ],
            "options": {
                "add_generation_prompt": True,
                "enable_thinking": True,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "search<>&'",
                        "description": "it's a <tool> & more",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query<": {"type": "string", "description": "it's <b> & 'q'"},
                            },
                        },
                    },
                }],
            },
        },
    ]


def render(env: Environment, source: str, case: dict[str, object]) -> str:
    template = env.from_string(source)
    options = case["options"]
    variables: dict[str, object] = {
        "messages": case["messages"],
        "add_generation_prompt": options.get("add_generation_prompt", True),
        "enable_thinking": options.get("enable_thinking", True),
        # The public PromptOptions carries an explicit bool; serve defaults it to false.
        "preserve_thinking": options.get("preserve_thinking", False),
    }
    if "reasoning_effort" in options:
        variables["reasoning_effort"] = options["reasoning_effort"]
    if "tools" in options:
        variables["tools"] = options["tools"]
    return template.render(**variables)


def main() -> int:
    resources = read_resources(ARTIFACT)
    template = resources["chat_template.jinja"].decode("utf-8").rstrip("\n")
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != "4e9a95ab9dc39ca41b7f99eb9ffc24572fe897b64b316c08c7ac26f840e0c3e0":
        raise ValueError(f"unexpected Flash-Next template digest: {digest}")

    env = Environment()
    env.policies["json.dumps_kwargs"] = {"sort_keys": False}
    tokenizer = Tokenizer.from_buffer(resources["tokenizer.json"])

    out: dict[str, object] = {
        "template_sha256": digest,
        "cases": [],
    }
    for case in cases():
        text = render(env, template, case)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        out["cases"].append({
            "name": case["name"],
            "messages": case["messages"],
            "options": case["options"],
            "rendered": text,
            "token_ids": ids,
        })
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
