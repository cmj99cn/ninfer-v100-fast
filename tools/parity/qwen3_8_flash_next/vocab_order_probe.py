"""Id-order probes for the Flash-Next artifact's embedding and output head.

Three questions, all answerable from the artifact alone:

1. Do the last ``vocab - tokenizer_size`` rows carry content, or are they the padding region?
2. Is the output head in the tokenizer's id order (high-norm rows are high-frequency tokens)?
3. Do the embedding and the output head agree with each other on the id order (the diagonal of
   the embedding/head cosine matrix dominates the off-diagonal)?

A positive answer to (2) and (3) means the input side and the output side agree on the token
ids, so a wrong prediction cannot come from a permuted vocabulary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.bindings import V3Artifact
from tools.reference.qwen3_8_flash_next.ops import bf16_from_u16


def read_rows(artifact, object_id, ids, hidden):
    row_bytes = hidden * 2
    out = np.empty((len(ids), hidden), dtype=np.float32)
    for index, token in enumerate(ids):
        raw = artifact.read_object(object_id, int(token) * row_bytes, row_bytes)
        out[index] = bf16_from_u16(np.frombuffer(raw, dtype="<u2"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--sample", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {}
    with V3Artifact(args.artifact) as artifact:
        embedding_id = artifact.single_object("text/token_embedding")
        head_id = artifact.single_object("text/output_head")
        vocab = int(artifact.object(head_id).shape[0])

        tail = list(range(248000, vocab))
        tail_embedding = read_rows(artifact, embedding_id, tail, args.hidden)
        tail_head = read_rows(artifact, head_id, tail, args.hidden)
        report["tail"] = {
            "first_id": tail[0],
            "count": len(tail),
            "embedding_norm_min": float(np.min(np.linalg.norm(tail_embedding, axis=1))),
            "embedding_norm_max": float(np.max(np.linalg.norm(tail_embedding, axis=1))),
            "embedding_zero_rows": int(
                np.sum(np.all(tail_embedding == 0.0, axis=1))
            ),
            "head_norm_min": float(np.min(np.linalg.norm(tail_head, axis=1))),
            "head_norm_max": float(np.max(np.linalg.norm(tail_head, axis=1))),
            "head_zero_rows": int(np.sum(np.all(tail_head == 0.0, axis=1))),
        }

        ids = np.arange(0, 248044, dtype=np.int64)
        rng = np.random.default_rng(20261117)
        sample = np.sort(rng.choice(ids, size=args.sample, replace=False))
        embedding = read_rows(artifact, embedding_id, sample, args.hidden)
        head = read_rows(artifact, head_id, sample, args.hidden)

        embedding_norm = np.linalg.norm(embedding, axis=1)
        head_norm = np.linalg.norm(head, axis=1)
        top_head = np.argsort(-head_norm)[:16]
        top_embedding = np.argsort(-embedding_norm)[:16]

        unit_embedding = embedding / embedding_norm[:, None]
        unit_head = head / head_norm[:, None]
        cosine = unit_embedding @ unit_head.T
        diagonal = np.diag(cosine)
        off_diagonal = cosine[~np.eye(cosine.shape[0], dtype=bool)]
        best = np.argmax(cosine, axis=1)
        report["agreement"] = {
            "sample_size": int(sample.size),
            "diagonal_cosine_mean": float(np.mean(diagonal)),
            "off_diagonal_cosine_mean": float(np.mean(off_diagonal)),
            "off_diagonal_cosine_std": float(np.std(off_diagonal)),
            "diagonal_z_score": float(
                (np.mean(diagonal) - np.mean(off_diagonal)) / (np.std(off_diagonal) + 1e-12)
            ),
            "argmax_agreement_rate": float(np.mean(best == np.arange(sample.size))),
            "argmax_offset_histogram": {
                str(int(offset)): int(count)
                for offset, count in zip(
                    *np.unique(best - np.arange(sample.size), return_counts=True)
                )
                if abs(int(offset)) < 64
            },
        }

        tokenizer_text = artifact.resource("tokenizer.json").decode("utf-8")
        vocabulary = json.loads(tokenizer_text)["model"]["vocab"]
        inverse = {int(index): word for word, index in vocabulary.items()}
        report["head_top_norm"] = [
            {
                "token": int(sample[index]),
                "text": inverse.get(int(sample[index]), "<added>"),
                "norm": float(head_norm[index]),
            }
            for index in top_head
        ]
        report["embedding_top_norm"] = [
            {
                "token": int(sample[index]),
                "text": inverse.get(int(sample[index]), "<added>"),
                "norm": float(embedding_norm[index]),
            }
            for index in top_embedding
        ]

    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
