"""Diagnostic: wire ignored/assumed text-config fields into the 4-layer reference.

No product code is touched. Each variant is selected with --variant and applied by
monkeypatching the reference module in-process. Reports teacher-forced NLL on A32 and
the `The capital of France is` -> ` Paris` top-5, with the task's cliff gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next import model as refmodel_mod
from tools.reference.qwen3_8_flash_next import ops as refops
from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.ops import bf16_from_u16, sigmoid
from tools.parity.qwen3_8_flash_next import layout_search


def _matrix_logits(model, samples, object_name):
    object_id = model._object(object_name)
    vocab = int(model.artifact.object(object_id).shape[0])
    row_bytes = model.cfg.hidden * 2
    values = np.asarray(samples, dtype=np.float64)
    tokens = values.shape[0]
    out = np.empty((tokens, vocab), dtype=np.float64)
    for start in range(0, vocab, 8192):
        count = min(8192, vocab - start)
        raw = model.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
        weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, model.cfg.hidden)
        out[:, start : start + count] = values @ weights.astype(np.float64).T
    return out


def apply_tie_embedding():
    """tie_word_embeddings=true: logits from token_embedding instead of output_head."""

    def _logits(self, sample):
        return _matrix_logits(self, sample[None, :], "text/token_embedding")[0]

    def position_logits(self, samples):
        return _matrix_logits(self, samples, "text/token_embedding")

    refmodel_mod.RefModel._logits = _logits
    refmodel_mod.RefModel.position_logits = position_logits


def apply_gate_identity():
    """output_gate_type != sigmoid: drop the sigmoid gate on attention output."""

    def qsa_sparse_gqa(query, output_gate, k_cache, v_cache, selections, positions, compress, head_dim):
        q = np.asarray(query, dtype=np.float64)
        key = np.asarray(k_cache, dtype=np.float64)
        value = np.asarray(v_cache, dtype=np.float64)
        tokens, q_heads, _ = q.shape
        kv_heads = key.shape[1]
        group = q_heads // kv_heads
        out = np.zeros((tokens, q_heads, head_dim), dtype=np.float64)
        scale = 1.0 / np.sqrt(head_dim)
        for t in range(tokens):
            position = int(positions[t])
            complete = (position + 1) // compress
            visible = set()
            for block in selections[t]:
                start = int(block) * compress
                visible.update(range(start, start + compress))
            visible.update(range(complete * compress, position + 1))
            index = np.asarray(sorted(j for j in visible if 0 <= j <= position), dtype=np.int64)
            if index.size == 0:
                continue
            for kvh in range(kv_heads):
                heads = slice(kvh * group, (kvh + 1) * group)
                scores = q[t, heads, :] @ key[index, kvh, :].T * scale
                scores = scores - np.max(scores, axis=1, keepdims=True)
                weights = np.exp(scores)
                weights = weights / np.sum(weights, axis=1, keepdims=True)
                out[t, heads, :] = weights @ value[index, kvh, :]
        return refops.bf16(out)

    refmodel_mod.qsa_sparse_gqa = qsa_sparse_gqa


def apply_topk_no_norm():
    """norm_topk_prob=false: raw full-softmax expert weights instead of selected renormalization."""

    def topk_route(x, control, k):
        xf = np.asarray(x, dtype=np.float64)
        logits = xf @ np.asarray(control, dtype=np.float64).T
        routed = logits[:, :-1]
        full = np.exp(routed - np.max(routed, axis=1, keepdims=True))
        full = full / np.sum(full, axis=1, keepdims=True)
        order = np.argsort(-routed, axis=1, kind="stable")[:, :k]
        weights = np.take_along_axis(full, order, axis=1)
        shared = sigmoid(logits[:, -1])
        return order.astype(np.int32), refops.fp32(weights), refops.fp32(shared)

    refmodel_mod.topk_route = topk_route


VARIANTS = {
    "baseline": None,
    "tie_embedding": apply_tie_embedding,
    "gate_identity": apply_gate_identity,
    "topk_no_norm": apply_topk_no_norm,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt-ids", default="760,6511,314,9338,369")
    parser.add_argument("--paris-id", type=int, default=11751)
    args = parser.parse_args()

    setup = VARIANTS[args.variant]
    if setup is not None:
        setup()
    tokens = [int(value) for value in args.tokens.read_text().split()]
    prompt_ids = [int(value) for value in args.prompt_ids.replace(",", " ").split()]
    config = FlashNextConfig(layers=args.layers)
    report = layout_search.evaluate(args.artifact, config, layout_search.CURRENT, tokens, prompt_ids,
                                    args.paris_id)
    report["variant"] = args.variant
    cliff = report["mean_nll"] < 8.0 or report["prompt_paris"]["probability"] > 0.05
    report["cliff"] = cliff
    top = ", ".join(f"{e['text']!r}:{e['probability']:.3g}" for e in report["prompt_top5"])
    print(
        f"variant={args.variant:16s} nll={report['mean_nll']:8.4f} "
        f"paris_p={report['prompt_paris']['probability']:.4g} "
        f"paris_nll={report['prompt_paris']['nll']:7.4f} top5=[{top}] "
        f"t={report['total_seconds']:.1f}s CLIFF={cliff}",
        flush=True,
    )
    out = Path("/tmp/config_audit_" + args.variant + ".json")
    out.write_text(__import__("json").dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
