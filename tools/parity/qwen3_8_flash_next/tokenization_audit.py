"""Diagnosis: compare artifact stdlib BPE ids against official HF tokenizers ids.

Read-only: reads the exported tokenizer resource and the §1 text from the artifact doc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer

from tools.bench.flash_next_text_corpus import load_encoder

ORACLE = Path("/tmp/opencode/oracle")
DOC = ROOT / "docs/maintainer/qwen3.8-flash-next-artifact.md"
A32 = Path("/tmp/A32.ids")


def read_ids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split() if line.strip()]


def diff_report(name: str, a: list[int], b: list[int], limit: int = 32) -> dict:
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    count = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    return {
        "name": name,
        "len_a": len(a),
        "len_b": len(b),
        "first_diff": first,
        "diff_count": count,
        "first_limited": a[:limit],
        "second_limited": b[:limit],
    }


def main() -> int:
    tok = Tokenizer.from_file(str(ORACLE / "tokenizer.json"))
    encoder = load_encoder(Path("/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer"))

    # §1 text: exact segment used to produce /tmp/A32.ids
    text = DOC.read_text(encoding="utf-8")
    segment = text[1129 : 1129 + 4000]
    print("=== §1 segment (repr, first 200 chars) ===")
    print(repr(segment[:200]))

    ours = encoder.encode(segment)[:32]
    official = tok.encode(segment).ids[:32]
    a32 = read_ids(A32)

    print("\n=== three-way first-32 ids ===")
    print(f"{'pos':>3} {'A32(stdlib@1129)':>18} {'stdlib-now':>12} {'official-HF':>12}  match?")
    for i in range(32):
        o = official[i] if i < len(official) else -1
        s = ours[i] if i < len(ours) else -1
        a = a32[i] if i < len(a32) else -1
        print(f"{i:>3} {a:>18} {s:>12} {o:>12}  {'OK' if a == s == o else 'DIFF'}")

    for label, x, y in [
        ("A32 vs stdlib-now", a32, ours),
        ("A32 vs official-HF", a32, official),
        ("stdlib-now vs official-HF", ours, official),
    ]:
        r = diff_report(label, x, y)
        print(f"\n[{label}] len={r['len_a']} vs {r['len_b']} "
              f"first_diff={r['first_diff']} diff_count={r['diff_count']}")

    # Also the natural sentence
    for phrase in ["The capital of France is", "The quick brown fox jumps over the lazy dog."]:
        o = tok.encode(phrase).ids
        s = encoder.encode(phrase)
        print(f"\n=== '{phrase}' ===")
        print(f"official : {o}")
        print(f"stdlib   : {s}")
        print(f"equal    : {o == s}  first_diff="
              f"{next((i for i in range(min(len(o), len(s))) if o[i] != s[i]), None)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
