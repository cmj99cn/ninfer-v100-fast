"""Encode real text with the tokenizer embedded in a Qwen3.8 Flash-Next artifact."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reference.qwen3_8_flash_next.bindings import V3Artifact


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_letter_or_mark(ch: str) -> bool:
    return _is_letter(ch) or _is_mark(ch)


_CONTRACTIONS = (("'s", 2), ("'t", 2), ("'re", 3), ("'ve", 3), ("'m", 2), ("'ll", 3), ("'d", 2))


def _pretokenize(text: str) -> list[str]:
    pieces: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        ch = text[index]
        if ch == "'":
            matched = False
            for spelling, length in _CONTRACTIONS:
                tail = text[index : index + length]
                if tail.lower() == spelling:
                    pieces.append(tail)
                    index += length
                    matched = True
                    break
            if matched:
                continue

        optional = 0
        if ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch):
            optional = 1
        if index + optional < size and (
            _is_letter_or_mark(text[index + optional])
        ):
            end = index + optional + 1
            while end < size and _is_letter_or_mark(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue
        if _is_letter_or_mark(ch):
            end = index + 1
            while end < size and _is_letter_or_mark(text[end]):
                end += 1
            pieces.append(text[index:end])
            index = end
            continue

        if _is_number(ch):
            pieces.append(ch)
            index += 1
            continue

        base = index
        if ch == " ":
            base = index + 1
        if base < size:
            end = base
            while end < size:
                current = text[end]
                if current.isspace() or _is_letter_or_mark(current) or _is_number(current):
                    break
                end += 1
            if end > base:
                while end < size and text[end] in "\r\n":
                    end += 1
                pieces.append(text[index:end])
                index = end
                continue

        if ch.isspace():
            end = index
            last_newline = -1
            while end < size and text[end].isspace():
                if text[end] in "\r\n":
                    last_newline = end
                end += 1
            if last_newline >= 0:
                pieces.append(text[index : last_newline + 1])
                index = last_newline + 1
            elif end < size and end - index >= 2:
                pieces.append(text[index : end - 1])
                index = end - 1
            else:
                pieces.append(text[index:end])
                index = end
            continue

        pieces.append(ch)
        index += 1
    return pieces


def _bytes_to_unicode() -> dict[int, str]:
    ordered = list(range(ord("!"), ord("~") + 1))
    ordered += list(range(ord("¡"), ord("¬") + 1))
    ordered += list(range(ord("®"), ord("ÿ") + 1))
    codes = list(ordered)
    extra = 0
    for value in range(256):
        if value not in ordered:
            ordered.append(value)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(ordered, (chr(code) for code in codes)))


class BpeEncoder:
    def __init__(self, tokenizer: dict):
        model = tokenizer["model"]
        self.vocab: dict[str, int] = model["vocab"]
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            self.ranks[(left, right)] = rank
        self.byte_map = _bytes_to_unicode()
        self.added = sorted(
            ((entry["content"], int(entry["id"])) for entry in tokenizer.get("added_tokens", [])),
            key=lambda entry: (-len(entry[0]), entry[1]),
        )
        self.cache: dict[str, tuple[int, ...]] = {}

    def _bpe(self, piece: str) -> tuple[str, ...]:
        symbols = list(piece)
        if len(symbols) < 2:
            return tuple(symbols)
        while True:
            best_rank = None
            best_pair = None
            for pair in zip(symbols, symbols[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                return tuple(symbols)
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == best_pair[0]
                    and symbols[index + 1] == best_pair[1]
                ):
                    merged.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _pretokenize(text):
            cached = self.cache.get(piece)
            if cached is None:
                byte_piece = "".join(self.byte_map[value] for value in piece.encode("utf-8"))
                symbols = self._bpe(byte_piece)
                cached = tuple(self.vocab[symbol] for symbol in symbols)
                self.cache[piece] = cached
            ids.extend(cached)
        return ids

    def encode(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", text)
        if not self.added:
            return self._encode_ordinary(normalized)
        ids: list[int] = []
        ordinary: list[str] = []
        index = 0
        while index < len(normalized):
            for content, token_id in self.added:
                if normalized.startswith(content, index):
                    if ordinary:
                        ids.extend(self._encode_ordinary("".join(ordinary)))
                        ordinary.clear()
                    ids.append(token_id)
                    index += len(content)
                    break
            else:
                ordinary.append(normalized[index])
                index += 1
        if ordinary:
            ids.extend(self._encode_ordinary("".join(ordinary)))
        return ids


def load_encoder(artifact: Path) -> BpeEncoder:
    with V3Artifact(artifact) as reader:
        tokenizer = json.loads(reader.resource("tokenizer.json"))
    return BpeEncoder(tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("text", type=Path)
    parser.add_argument("--start-char", type=int, default=0)
    parser.add_argument("--chars", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.text.read_text(encoding="utf-8")
    start = max(args.start_char, 0)
    end = len(text) if args.chars <= 0 else min(len(text), start + args.chars)
    segment = text[start:end]
    encoder = load_encoder(args.artifact)
    ids = encoder.encode(segment)
    if args.tokens > 0:
        ids = ids[: args.tokens]
    if len(ids) < 2:
        raise SystemExit("the encoded segment is too short")
    args.output.write_text("\n".join(str(value) for value in ids) + "\n", encoding="utf-8")
    print(
        "tokens={} chars={} first={} last={}".format(
            len(ids), len(segment), ids[0], ids[-1]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
