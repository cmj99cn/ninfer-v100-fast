# Stable CLI examples

> **本分支现状（2026-10，fork 维护者注）**：本仓库当前维护 `qwen3.8-27b` NVFP4 在
> 2×V100-SXM2-32GB / TP2 上的部署。自转换 NVFP4 制品包含 Vision，`--vision`/多模态示例可用；
> GGUF Q4_K_M legacy 制品拒绝 `--vision`。下文的 `Qwen3.6-27B/35B-A3B` 型号与旧制品路径是
> 上游内容，**本分支未验证**；请把 `$MODEL` 替换为你的 `qwen3.8-27b` NVFP4 制品。

This directory contains committed, offline `--messages` inputs for exercising the product CLI from
the repository root. It covers short text, chat history, images, video, mixed multimodal history,
hard thinking problems, long decode, and four long-context capacities. This is an operator-facing
example set, not a second correctness framework.

[`manifest.json`](manifest.json) lists each case, its intended observation, recommended runtime
budget, and its prepared-prompt token count.

## Quick start

Run from the repository root because media paths in the JSON files are repository-relative:

```bash
CLI=./build/apps/ninfer
MODEL=models/qwen3_8_27b_nvfp4.ninfer  # replace with your artifact path

$CLI "$MODEL" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8
```

Expected stdout is exactly `42`. `--no-thinking --greedy` is the normal comparison mode for simple
cases. Reasoning, progress, timings, memory, and MTP statistics are written to stderr; answer content
is written to stdout.

## Text and multimodal cases

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_chat_history.json \
  --no-thinking --greedy --max-new 32

$CLI "$MODEL" --messages examples/cli/messages/text_code_review.json \
  --no-thinking --greedy --max-new 256

for CASE in image_chart image_natural video_temporal multi_image_compare \
            mixed_image_video mixed_multiturn; do
  $CLI "$MODEL" --messages "examples/cli/messages/${CASE}.json" \
    --max-context 8192 --no-thinking --greedy --max-new 128 --vision
done
```

The controlled observations are:

| Case | Expected observation |
|---|---|
| `text_chat_history` | `Cedar|2041-09-17|37` |
| `text_code_review` | empty input divides by zero; add an explicit empty-input policy |
| `image_chart` | `NIFER VISION 731`; three red circles; blue square on the left |
| `image_natural` | mailbox `24`; sun on the right |
| `video_temporal` | red circle moves; green `3`; square still visible when `3` appears; ending `9` |
| `multi_image_compare` | two circles, three circles, and a new yellow star |
| `mixed_image_video` | `NIFER-9` |
| `mixed_multiturn` | `24-9` |

To exercise MTP through the same input path:

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Thinking cases

Do not pass `--no-thinking` for these inputs. Give reasoning enough room to complete and transition
to answer content; the model is free to stop before the requested maximum.

```bash
$CLI "$MODEL" --messages examples/cli/messages/thinking_logic_grid.json \
  --greedy --max-context 16384 --max-new 8192

$CLI "$MODEL" --messages examples/cli/messages/thinking_multimodal_checksum.json \
  --greedy --max-context 8192 --max-new 4096 --vision
```

The logic grid has one solution and must end with `CHECK=4606`. The multimodal case reads independent
facts from two images and one video, then must end with `CHECKSUM=2238`.

## Long decode

These cases deliberately get a generous budget. They run until the model's stop token, not to
discover the smallest `max-new` value that happens to fit one output.

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_decode_aime26_01.json \
  --greedy --max-context 262144 --max-new 65536
```

The answer must end with `\boxed{277}`. `long_decode_aime26_15.json` and
`long_decode_aime26_30.json` use the same budget with answers `83` and `393`.

## Long context

These prompt lengths include the chat template with thinking disabled. The inputs freeze meaningful
source excerpts, with one retrieval needle placed across the packet.

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_niah_8k.json \
  --max-context 8192 --kv-dtype bf16 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_64k.json \
  --max-context 65536 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_128k.json \
  --max-context 131072 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_256k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128
```

All four must output:

```text
ORCHID=493817; COLOR=COBALT
```

The committed prompt token counts were validated against the Qwen3.6-27B and Qwen3.6-35B-A3B
frontend profiles, which produce identical sequences for these files. Qwen3.8-27B uses the same CLI
surface but carries its own tokenizer and chat-template resources; inspect its prepared token count
when using these fixtures.

## Fixture construction

All PNG and MP4 media are project-authored deterministic scenes. The video is five seconds at 8 FPS
with forty H.264 frames. Runtime tests never depend on a network URL or mutable external content.

The committed generated files are the actual inputs. To intentionally rebuild media and the four
long-context JSON files from the current source tree:

```bash
python3 examples/cli/make_fixtures.py \
  --tokenizer /path/to/Qwen3.6-27B
```

Regeneration updates the frozen source snapshot when selected project files change, so generated
differences should be reviewed like any other fixture change.
