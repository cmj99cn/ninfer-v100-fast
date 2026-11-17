"""Locate an artifact PLE row inside the official IQ4_NL GGUF table."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gguf import GGUFReader

from tools.reference.qwen3_8_flash_next.config import CFG, FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel

CODE = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def dequant(chunk: np.ndarray) -> np.ndarray:
    block = chunk.reshape(-1, 5, 18)
    bits = block[:, :, 0].astype(np.uint16) | (block[:, :, 1].astype(np.uint16) << 8)
    sign = np.where((bits & 0x8000) != 0, -1.0, 1.0).astype(np.float32)
    exponent = (bits >> 10) & 31
    mantissa = bits & 1023
    safe_exponent = np.where(exponent >= 31, 15, exponent).astype(np.int32) - 15
    scales = np.where(
        exponent == 0,
        sign * np.ldexp(mantissa.astype(np.float32) / 1024.0, -14),
        sign * np.ldexp(1.0 + mantissa.astype(np.float32) / 1024.0, safe_exponent),
    )
    scales = np.where(exponent >= 31, 0.0, scales)
    low = CODE[block[:, :, 2:].reshape(-1, 16) & 15].reshape(-1, 5, 16)
    high = CODE[block[:, :, 2:].reshape(-1, 16) >> 4].reshape(-1, 5, 16)
    values = np.empty((block.shape[0], 160), dtype=np.float32)
    for b in range(5):
        values[:, b * 32 : b * 32 + 16] = scales[:, b, None] * low[:, b, :]
        values[:, b * 32 + 16 : b * 32 + 32] = scales[:, b, None] * high[:, b, :]
    return values


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("gguf", type=Path)
    p.add_argument("--token", type=int, default=760)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--shard", type=int, default=7)
    p.add_argument("--chunk", type=int, default=100000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    reader = GGUFReader(args.gguf, "r")
    tensor = reader.tensors[0]
    with RefModel(args.artifact, config=FlashNextConfig()) as model:
        weights = model._ple_weights()
        row = int(model.ple_rows([args.token])[0, args.head])
        shard, within = divmod(row, CFG.ple_shard_rows)
        target = np.asarray(model.store.ple_row(weights["table"][shard], within), dtype=np.float32)

    path = args.gguf
    best = (-1.0, -1)
    start = args.shard * CFG.ple_shard_rows
    stop = min(start + CFG.ple_shard_rows, 320001536)
    handle = path.open("rb")
    for base in range(start, stop, args.chunk):
        count = min(args.chunk, stop - base)
        handle.seek(int(tensor.data_offset) + base * 90)
        raw = np.frombuffer(handle.read(count * 90), dtype=np.uint8)
        values = dequant(raw)
        similarity = values @ target / (np.linalg.norm(values, axis=1) * np.linalg.norm(target))
        index = int(np.argmax(similarity))
        if similarity[index] > best[0]:
            best = (float(similarity[index]), base + index)
    handle.close()
    report = {
        "token": args.token,
        "head": args.head,
        "artifact_global_row": row,
        "artifact_shard": shard,
        "artifact_within": within,
        "gguf_best_row": best[1],
        "gguf_best_cosine": best[0],
        "gguf_shard_searched": args.shard,
    }
    print(report)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ple_table_scan.json").write_text(str(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
