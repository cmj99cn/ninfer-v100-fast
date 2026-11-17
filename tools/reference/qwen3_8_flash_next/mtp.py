"""FP64 oracle for one Qwen3.8-Flash-Next MTP draft step.

The MTP layer is the artifact's indexed-attention layer followed by its own MoE and
hyper-connection mixer. Its input fusion follows the upstream
`Qwen4ExpMTP._fuse_residual_linear_shared` transcription: the next token embedding and
each of the four hidden streams are normalized and projected separately, then summed
into the four-stream residual that the MTP layer consumes.

The routed expert bank is the artifact's block-128x128 E4M3FN bank. Each code is decoded
with the exact BF16 block scale stored alongside it; the oracle never reads a production
kernel's staging or reduction profile.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from .bindings import V3Artifact, align_up
from .config import CFG
from .model import RefModel
from .ops import (
    bf16,
    bf16_from_u16,
    dense,
    e4m3_lut,
    f16,
    hc_combine_norm,
    hc_final_mix,
    hc_grouped_rmsnorm,
    hc_mix,
    qsa_index_block,
    qsa_indexer,
    qsa_norm_rope,
    qsa_sparse_gqa,
    silu,
    topk_route,
)
from .state import ModelState

_SEM = {
    "rope_head_pos": os.environ.get("NINFER_DIAG_ROPE_HEAD_POS", "same"),
}


def decode_block128(artifact: V3Artifact, object_id: str, matrix: int, row_begin: int,
                    row_count: int) -> np.ndarray:
    matrices, rows, columns = artifact.object(object_id).shape
    if row_begin % 128 or row_count % 128 or columns % 128:
        raise ValueError("block128 expert slice is not 128-aligned")
    code_bytes = rows * columns
    scale_offset = align_up(matrices * code_bytes, 256)
    blocks_per_matrix = (rows // 128) * (columns // 128)
    scale_bytes = blocks_per_matrix * 2
    codes = artifact.read_object(object_id, matrix * code_bytes + row_begin * columns,
                                 row_count * columns)
    scales = artifact.read_object(
        object_id,
        scale_offset + matrix * scale_bytes + (row_begin // 128) * (columns // 128) * 2,
        (row_count // 128) * (columns // 128) * 2,
    )
    values = e4m3_lut()[np.frombuffer(codes, dtype=np.uint8)].astype(np.float64)
    values = values.reshape(row_count, columns)
    words = bf16_from_u16(np.frombuffer(scales, dtype="<u2")).astype(np.float64)
    block_scales = words.reshape(row_count // 128, columns // 128)
    expanded = np.repeat(np.repeat(block_scales, 128, axis=0), 128, axis=1)
    return values * expanded


def gemma_rms(x: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + epsilon)
    return values * inv * (1.0 + np.asarray(weight, dtype=np.float64))


@dataclass
class MtpWeights:
    embedding_norm: np.ndarray
    hidden_norm: np.ndarray
    fc_embedding: np.ndarray
    fc_hidden: np.ndarray
    final_hc: dict
    attention_hc: dict
    moe_hc: dict
    qsa: dict
    moe: dict
    output_head: np.ndarray


class MtpRef:
    def __init__(self, artifact_path: str):
        self.artifact = V3Artifact(artifact_path)

    def close(self) -> None:
        self.artifact.close()

    def __enter__(self) -> "MtpRef":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _object(self, name: str) -> str:
        return self.artifact.single_object(name)

    def _contiguous(self, name: str) -> np.ndarray:
        object_id = self._object(name)
        shape = self.artifact.object(object_id).shape
        return decode_contiguous(self.artifact, object_id, shape)

    def _range(self, name: str, begin: int, count: int) -> np.ndarray:
        object_id = self._object(name)
        shape = self.artifact.object(object_id).shape
        columns = shape[-1]
        raw = self.artifact.read_object(object_id, begin * columns * 2, count * columns * 2)
        return decode_contiguous_bytes(raw, object_id, (count, columns))

    def _hc(self, prefix: str) -> dict:
        down = self._range(prefix + "/down", 0, CFG.hc_lowrank)
        inject = self._range(prefix + "/inject", CFG.hc_lowrank, CFG.hc_streams)
        norm = self._contiguous(prefix + "/norm").reshape(CFG.hc_streams, CFG.hidden)
        up = self._contiguous(prefix + "/up")
        return {"down": np.concatenate([down, inject]), "norm": norm, "up": up}

    def weights(self) -> MtpWeights:
        qsa_parent = self._object("mtp/layers/0/attention/key")
        assert qsa_parent == self._object("mtp/layers/0/attention/query")
        moe_parent = self._object("mtp/layers/0/moe/router")

        qsa = {
            "key": self._range("mtp/layers/0/attention/key", 0, CFG.kv_width),
            "value": self._range("mtp/layers/0/attention/value", CFG.kv_width, CFG.kv_width),
            "index_key": self._range("mtp/layers/0/attention/index_key", 2 * CFG.kv_width,
                                     CFG.index_key_width),
            "query": self._range("mtp/layers/0/attention/query",
                                 2 * CFG.kv_width + CFG.index_key_width, CFG.query_width),
            "gate": self._range("mtp/layers/0/attention/output_gate",
                                2 * CFG.kv_width + CFG.index_key_width + CFG.query_width,
                                CFG.query_width),
            "index_query": self._range(
                "mtp/layers/0/attention/index_query",
                2 * CFG.kv_width + CFG.index_key_width + 2 * CFG.query_width,
                CFG.index_query_width,
            ),
            "query_norm": self._contiguous("mtp/layers/0/attention/query_norm"),
            "key_norm": self._contiguous("mtp/layers/0/attention/key_norm"),
            "index_query_norm": self._contiguous("mtp/layers/0/attention/index_query_norm"),
            "index_key_norm": self._contiguous("mtp/layers/0/attention/index_key_norm"),
            "out": self._contiguous("mtp/layers/0/attention/output"),
        }
        moe = {
            "control": np.concatenate([
                self._range("mtp/layers/0/moe/router", 0, CFG.experts),
                self._range("mtp/layers/0/moe/shared_score", CFG.experts, 1),
            ]),
            "shared_up": np.concatenate([
                self._range("mtp/layers/0/moe/shared/gate", CFG.experts + 1,
                            CFG.shared_intermediate),
                self._range("mtp/layers/0/moe/shared/up",
                            CFG.experts + 1 + CFG.shared_intermediate, CFG.shared_intermediate),
            ]),
            "shared_down": self._contiguous("mtp/layers/0/moe/shared/down"),
            "gate_object": self._object("mtp/layers/0/moe/experts/gate"),
            "down_object": self._object("mtp/layers/0/moe/experts/down"),
        }
        final_hc = {
            "down": self._contiguous("mtp/final_hc/down"),
            "up": self._contiguous("mtp/final_hc/up"),
            "norm": self._contiguous("mtp/final_hc/norm").reshape(CFG.hc_streams, CFG.hidden),
        }
        return MtpWeights(
            embedding_norm=self._contiguous("mtp/pre_fc_embedding_norm"),
            hidden_norm=self._contiguous("mtp/pre_fc_hidden_norm"),
            fc_embedding=self._contiguous("mtp/fc_embedding"),
            fc_hidden=self._contiguous("mtp/fc_hidden"),
            final_hc=final_hc,
            attention_hc=self._hc("mtp/layers/0/attention_hc"),
            moe_hc=self._hc("mtp/layers/0/moe_hc"),
            qsa=qsa,
            moe=moe,
            output_head=self._contiguous("mtp/output_head"),
        )

    def _fuse(self, hidden: np.ndarray, token: int, weights: MtpWeights) -> np.ndarray:
        embedded = self.artifact.read_object(self._object("mtp/token_embedding"),
                                             token * CFG.hidden * 2, CFG.hidden * 2)
        embedding = bf16_from_u16(np.frombuffer(embedded, dtype="<u2")).astype(np.float64)
        input_embeds = dense(gemma_rms(embedding, weights.embedding_norm, CFG.rms_epsilon),
                             weights.fc_embedding)
        flat = np.asarray(hidden, dtype=np.float64).reshape(-1, CFG.residual_width)
        normalized = gemma_rms(flat, weights.hidden_norm, CFG.rms_epsilon)
        streams = normalized.reshape(-1, CFG.hc_streams, CFG.hidden)
        encoder = np.stack(
            [dense(stream, weights.fc_hidden) for stream in np.moveaxis(streams, -2, 0)],
            axis=-2,
        )
        fused = input_embeds.reshape(-1, CFG.hidden)[:, None, :] + encoder
        return fused.reshape(np.asarray(hidden).shape)

    def _qsa(self, hidden: np.ndarray, weights: dict, position: int) -> np.ndarray:
        query = dense(hidden, weights["query"]).reshape(1, CFG.q_heads, CFG.head_dim)
        gate = dense(hidden, weights["gate"]).reshape(1, CFG.q_heads, CFG.head_dim)
        projected = dense(hidden, np.concatenate([
            weights["key"], weights["value"], weights["index_query"], weights["index_key"]
        ]))
        key = projected[:, :CFG.kv_width].reshape(1, CFG.kv_heads, CFG.head_dim)
        value = projected[:, CFG.kv_width:2 * CFG.kv_width].reshape(1, CFG.kv_heads, CFG.head_dim)
        index_query = projected[:, 2 * CFG.kv_width:2 * CFG.kv_width + CFG.index_query_width]
        index_query = index_query.reshape(1, CFG.index_heads, CFG.index_head_dim)
        index_key = projected[:, 2 * CFG.kv_width + CFG.index_query_width:]
        index_key = index_key.reshape(1, CFG.index_kv_heads, CFG.index_head_dim)

        positions = np.asarray([position], dtype=np.int64)
        rope_positions = positions + 0
        if _SEM["rope_head_pos"] == "index_axis":
            index_positions = np.stack(
                [rope_positions // (2 ** head) for head in range(CFG.index_heads)], axis=1)
        else:
            index_positions = rope_positions
        query = qsa_norm_rope(query, weights["query_norm"], rope_positions, CFG.rms_epsilon,
                              CFG.rope_theta, CFG.rotary_dim)
        index_query = qsa_norm_rope(index_query, weights["index_query_norm"], index_positions,
                                    CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim)
        key = qsa_norm_rope(key, weights["key_norm"], rope_positions, CFG.rms_epsilon,
                            CFG.rope_theta, CFG.rotary_dim)
        index_blocks = qsa_index_block(index_key[0, 0], weights["index_key_norm"], 0,
                                       CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim)
        index_blocks = index_blocks.reshape(1, CFG.index_head_dim)
        selections = qsa_indexer(index_query, index_blocks, positions, CFG.index_block_budget)
        attention = qsa_sparse_gqa(query, gate, key, f16(value), selections, positions,
                                   CFG.index_compress, CFG.head_dim)
        return dense(attention.reshape(1, CFG.query_width), weights["out"])

    def _moe(self, hidden: np.ndarray, weights: dict) -> np.ndarray:
        ids, coefficients, shared = topk_route(hidden, weights["control"], CFG.experts_per_token)
        values = np.asarray(hidden, dtype=np.float64)
        projection = values @ weights["shared_up"].T
        activation = silu(projection[:, :CFG.shared_intermediate]) * \
            projection[:, CFG.shared_intermediate:]
        total = (np.asarray(shared, dtype=np.float64)[:, None] *
                 (activation @ weights["shared_down"].T))
        flat_ids = ids.reshape(-1)
        flat_coefficients = coefficients.reshape(-1).astype(np.float64)
        order = np.argsort(flat_ids, kind="stable")
        sorted_ids = flat_ids[order]
        start = 0
        while start < order.size:
            expert = int(sorted_ids[start])
            end = start + 1
            while end < order.size and int(sorted_ids[end]) == expert:
                end += 1
            chosen = order[start:end]
            token_index = chosen // CFG.experts_per_token
            coefficient = flat_coefficients[chosen]
            gate_up = np.concatenate([
                decode_block128(self.artifact, weights["gate_object"], expert, 0,
                                CFG.expert_intermediate),
                decode_block128(self.artifact, weights["gate_object"], expert,
                                CFG.expert_intermediate, CFG.expert_intermediate),
            ], axis=0)
            down = decode_block128(self.artifact, weights["down_object"], expert, 0, CFG.hidden)
            projected = values[token_index] @ gate_up.T
            activated = silu(projected[:, :CFG.expert_intermediate]) * \
                projected[:, CFG.expert_intermediate:]
            total[token_index] += coefficient[:, None] * (activated @ down.T)
            start = end
        return bf16(total)

    def draft(self, hidden: np.ndarray, next_token: int, position: int) -> tuple[int, np.ndarray]:
        weights = self.weights()
        hidden = np.asarray(hidden, dtype=np.float64).reshape(1, CFG.residual_width)
        residual = bf16(self._fuse(hidden, next_token, weights)).reshape(
            1, CFG.hc_streams, CFG.hidden)
        normalized = hc_grouped_rmsnorm(residual, weights.attention_hc["norm"], CFG.rms_epsilon)
        block_input, injection = hc_mix(normalized, weights.attention_hc["down"],
                                        weights.attention_hc["up"])
        mixer = self._qsa(block_input, weights.qsa, position)
        stage, normalized = hc_combine_norm(
            residual, mixer, injection, weights.moe_hc["norm"], CFG.rms_epsilon)
        block_input, injection = hc_mix(normalized, weights.moe_hc["down"], weights.moe_hc["up"])
        moe_out = self._moe(block_input, weights.moe)
        _, normalized = hc_combine_norm(
            stage, moe_out, injection, weights.final_hc["norm"], CFG.rms_epsilon)
        sample = hc_final_mix(normalized, weights.final_hc["down"], weights.final_hc["up"])[0]
        logits = weights.output_head.astype(np.float64) @ sample.astype(np.float64)
        return int(np.argmax(logits)), logits


def decode_contiguous(artifact: V3Artifact, object_id: str, shape) -> np.ndarray:
    return decode_contiguous_bytes(artifact.read_object(object_id), object_id, shape)


def decode_contiguous_bytes(raw: bytes, object_id: str, shape) -> np.ndarray:
    if object_id.startswith("resource/"):
        return bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(shape).astype(np.float64)
    return bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(shape).astype(np.float64)


def target_conditioning(artifact_path: str, ids: list[int]) -> np.ndarray:
    with RefModel(artifact_path, config=CFG.full()) as model:
        result = model.forward(ids, capture_tensors=True)
    return result["tensors"][(CFG.layers - 1, "residual_out")][-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=os.environ.get("NINFER_FLASH_NEXT_ARTIFACT"))
    parser.add_argument("--ids", required=True)
    parser.add_argument("--next-token", type=int, required=True)
    parser.add_argument("--position", type=int)
    parser.add_argument("--hidden", help="path to a cached conditioning vector")
    parser.add_argument("--save-hidden", help="write the conditioning vector here")
    args = parser.parse_args()
    if not args.weights:
        parser.error("--weights or NINFER_FLASH_NEXT_ARTIFACT is required")
    ids = [int(value) for value in args.ids.replace(",", " ").split()]
    if args.hidden:
        hidden = np.load(args.hidden)
    else:
        hidden = target_conditioning(args.weights, ids)
        if args.save_hidden:
            np.save(args.save_hidden, hidden)
    position = args.position if args.position is not None else len(ids)
    with MtpRef(args.weights) as model:
        token, logits = model.draft(hidden, args.next_token, position)
    order = np.argsort(-logits, kind="stable")[:3]
    print(json.dumps({"draft_token": token,
                      "top": [(int(value), float(logits[value])) for value in order]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
