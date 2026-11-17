"""Diagnose weight-compare between official BF16 safetensors and the flash-next artifact.

Reads only the safetensors header plus a few small tensor prefixes via HTTP Range,
and the matching BF16 objects from the .ninfer artifact through V3Artifact.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.reference.qwen3_8_flash_next.bindings import V3Artifact

INDEX = Path("/tmp/opencode/oracle/model.safetensors.index.json")
ARTIFACT = Path("/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer")
BASE = "https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/main/"

PREFIX_ELEMENTS = 32
GATE_ROWS = 8


def hf_range(url: str, start: int, end: int) -> bytes:
    out = subprocess.run(
        ["curl", "-sL", "--max-time", "120", "-r", f"{start}-{end}", url],
        capture_output=True,
        check=True,
    ).stdout
    if len(out) != end - start + 1:
        raise RuntimeError(
            f"short range read: asked {end-start+1} bytes, got {len(out)} from {url}"
        )
    return out


class SafeHeader:
    def __init__(self, shard: str):
        self.url = BASE + shard
        head = hf_range(self.url, 0, 7)
        self.header_len = int.from_bytes(head, "little")
        self.data_start = 8 + self.header_len
        raw = hf_range(self.url, 8, 8 + self.header_len - 1)
        self.header = json.loads(raw)

    def tensor_prefix(self, name: str, nbytes: int | None = None) -> tuple[np.ndarray, dict]:
        meta = self.header[name]
        start, end = meta["data_offsets"]
        dtype = meta["dtype"]
        shape = meta["shape"]
        itemsize = 2 if dtype in ("BF16", "F16") else 4
        total = (end - start) // itemsize
        if nbytes is not None:
            end = min(end, start + nbytes)
        raw = hf_range(self.url, self.data_start + start, self.data_start + end - 1)
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        return u16, {"dtype": dtype, "shape": shape, "elements": total,
                     "fetched_bytes": len(raw), "shard": self.url.rsplit("/", 1)[-1]}


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def read_parts(art: V3Artifact, parts, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    for part in parts:
        if remaining <= 0:
            break
        count = min(remaining, (part.end - part.begin) * 2)
        chunks.append(art.read_object(part.object, part.begin * 2, count))
        remaining -= count
    data = b"".join(chunks)
    if len(data) != nbytes:
        raise RuntimeError(f"short artifact part read: {len(data)} != {nbytes}")
    return data


def stats(official: np.ndarray, artifact: np.ndarray) -> dict:
    a = bf16_to_f32(official)
    b = bf16_to_f32(artifact)
    if a.shape != b.shape:
        return {"shape_mismatch": (a.shape, b.shape)}
    diff = a - b
    denom = np.maximum(np.abs(b), 1e-12)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {
        "n": int(a.size),
        "bitwise_equal": bool(np.array_equal(official, artifact)),
        "n_bit_mismatch": int(np.count_nonzero(official != artifact)),
        "max_abs": float(np.max(np.abs(diff))),
        "max_rel": float(np.max(np.abs(diff) / denom)),
        "cosine": cos,
        "mean_abs": float(np.mean(np.abs(diff))),
        "off_first8": [round(float(x), 6) for x in a[:8]],
        "art_first8": [round(float(x), 6) for x in b[:8]],
    }


def transform_scan(official: np.ndarray, artifact: np.ndarray) -> dict:
    a = bf16_to_f32(official)
    b = bf16_to_f32(artifact)
    if a.size != b.size:
        return {}
    candidates = {
        "identity": a,
        "reverse": a[::-1],
    }
    for dims in ((4, a.size // 4), (a.size // 4, 4)):
        if a.size % dims[0] == 0:
            try:
                candidates[f"reshape{dims}.T"] = a.reshape(dims).T.reshape(-1)
            except ValueError:
                pass
    out = {}
    for name, cand in candidates.items():
        denom = np.maximum(np.abs(b), 1e-12)
        out[name] = {
            "cosine": float(np.dot(cand, b) / (np.linalg.norm(cand) * np.linalg.norm(b) + 1e-30)),
            "max_abs": float(np.max(np.abs(cand - b))),
            "max_rel": float(np.max(np.abs(cand - b) / denom)),
        }
    nz = b != 0
    if np.any(nz):
        ratios = a[nz] / b[nz]
        out["ratio_a_over_b"] = {
            "min": float(np.min(ratios)), "max": float(np.max(ratios)),
            "median": float(np.median(ratios)),
        }
    return out


def main() -> int:
    wm = json.loads(INDEX.read_text())["weight_map"]
    specs = [
        ("model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
         "text/layers/0/attention_hc/norm", None, "attn_hc.norm L0"),
        ("model.language_model.layers.0.linear_attn.norm.weight",
         "text/layers/0/gdn/norm", None, "gdn.norm L0"),
        ("model.language_model.hyper_connection_mixer.hc_norm.weight",
         "text/final_hc/norm", None, "final_hc.norm top"),
        ("model.language_model.embed_tokens.weight",
         "text/token_embedding", PREFIX_ELEMENTS, "embed_tokens first 32 rows"),
        ("lm_head.weight", "text/output_head", PREFIX_ELEMENTS, "lm_head first 32 rows"),
        ("model.language_model.layers.0.mlp.gate.weight",
         "text/layers/0/moe/router", PREFIX_ELEMENTS, "mlp.gate / moe.router first 32 rows"),
    ]
    results = []
    headers: dict[str, SafeHeader] = {}
    with V3Artifact(ARTIFACT) as art:
        for official_name, binding, rows, label in specs:
            shard = wm[official_name]
            if shard not in headers:
                headers[shard] = SafeHeader(shard)
            hdr = headers[shard]
            nbytes = None if rows is None else rows * 2560 * 2
            off_u16, ometa = hdr.tensor_prefix(official_name, nbytes)
            parts = art.parts(binding)
            want = off_u16.size * 2
            art_u16 = np.frombuffer(read_parts(art, parts, want), dtype=np.uint16).copy()
            n = min(off_u16.size, art_u16.size)
            st = stats(off_u16[:n], art_u16[:n])
            rec = {"label": label, "official": official_name, "artifact_binding": binding,
                   "official_meta": ometa, "artifact_format": art.object(parts[0].object).format,
                   "comparison": st}
            if rows is None:
                rec["transforms"] = transform_scan(off_u16[:n], art_u16[:n])
            results.append(rec)
            print(json.dumps(rec, indent=1), flush=True)
    Path("/tmp/opencode/weight_compare_results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
