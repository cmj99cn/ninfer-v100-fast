"""Diagnosis: chat-template vs bare-text token ids for the Flash-Next artifact.

Uses the artifact-exported chat_template.jinja (the NInfer-modified template) via
transformers.AutoTokenizer, and writes id files for the FP64 NLL diagnostic.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer
from transformers import AutoTokenizer

ORACLE = Path("/tmp/opencode/oracle")


def main() -> int:
    jinja = (ORACLE / "chat_template.jinja").read_text()

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(ORACLE / "tokenizer.json", d / "tokenizer.json")
        cfg = json.loads((ORACLE / "tokenizer_config.json").read_text())
        cfg["chat_template"] = jinja
        (d / "tokenizer_config.json").write_text(json.dumps(cfg))
        tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)

        bare_text = "The capital of France is"
        bare_ids = tok(bare_text, add_special_tokens=False)["input_ids"]

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        chat_out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        chat_ids = list(chat_out["input_ids"])

        print("=== bare text ids ===")
        print(repr(bare_text), "->", bare_ids)
        print("=== chat-template ids ===")
        print("messages:", messages)
        print("count:", len(chat_ids))
        print(chat_ids)
        print("=== chat-template rendered text ===")
        rendered = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=False)
        print(repr(rendered))

        # Locate 'Paris' token in chat ids via incremental decode
        paris_enc = tok.encode(" Paris", add_special_tokens=False)
        paris_id = paris_enc[0] if isinstance(paris_enc, list) else paris_enc.ids[0]
        print("Paris token id:", paris_id,
              "positions in chat:", [i for i, v in enumerate(chat_ids) if v == paris_id])

        (ORACLE / "chat_ids.txt").write_text("\n".join(str(v) for v in chat_ids) + "\n")
        (ORACLE / "bare_ids.txt").write_text("\n".join(str(v) for v in bare_ids) + "\n")

        # Also encode the exact markdown §1 segment with official HF (sanity)
        doc = ROOT / "docs/maintainer/qwen3.8-flash-next-artifact.md"
        seg = doc.read_text(encoding="utf-8")[1129:1129 + 4000]
        hf = Tokenizer.from_file(str(ORACLE / "tokenizer.json"))
        print("§1 official-HF first 32:", hf.encode(seg).ids[:32])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
