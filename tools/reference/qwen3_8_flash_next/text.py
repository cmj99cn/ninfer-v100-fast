from __future__ import annotations

import numpy as np


def parse_ids(text):
    return [int(value) for value in text.replace(",", " ").split()]


def greedy(model, ids, steps):
    sequence = [int(token) for token in ids]
    produced = []
    logits = None
    for _ in range(steps):
        result = model.forward(sequence)
        logits = result["logits"]
        token = int(np.argmax(logits))
        produced.append(token)
        sequence.append(token)
    return produced, sequence, logits


def prefill(model, ids):
    result = model.forward(ids)
    logits = result["logits"]
    order = np.argsort(-logits, kind="stable")[:2]
    return int(order[0]), float(logits[order[0]] - logits[order[1]]), result
