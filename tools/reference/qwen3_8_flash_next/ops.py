from __future__ import annotations

import math

import numpy as np

from tools.artifact.numeric import decode_e2m1_word, decode_e4m3fn_word

from .config import CFG


def bf16(x):
    a = np.asarray(x, dtype=np.float32)
    u = np.ascontiguousarray(a).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def bf16_from_u16(words):
    u = np.asarray(words, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


def f16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def fp32(x):
    return np.asarray(x, dtype=np.float32)


def silu(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def dense(x, weight, round_bf16=True):
    out = np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    return bf16(out) if round_bf16 else out


def rms_offset(x, weight, epsilon):
    xf = np.asarray(x, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + epsilon)
    return xf * inv * (1.0 + np.asarray(weight, dtype=np.float64))


def hc_expand_norm(embedding, norm_weight, epsilon):
    emb = np.asarray(embedding, dtype=np.float64)
    streams = norm_weight.shape[0]
    inv = 1.0 / np.sqrt(np.mean(emb * emb, axis=-1, keepdims=True) + epsilon)
    residual = np.repeat(emb[:, None, :], streams, axis=1)
    normalized = emb[:, None, :] * (1.0 + np.asarray(norm_weight, dtype=np.float64))[None, :, :]
    normalized = normalized * inv[:, None, :]
    return bf16(residual), bf16(normalized)


def hc_grouped_rmsnorm(residual, norm_weight, epsilon):
    x = np.asarray(residual, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon)
    return bf16(x * inv * (1.0 + np.asarray(norm_weight, dtype=np.float64))[None, :, :])


def hc_combine(residual, block_output, injection):
    x = np.asarray(residual, dtype=np.float64)
    out = x + np.asarray(injection, dtype=np.float64)[:, :, None] * np.asarray(
        block_output, dtype=np.float64
    )[:, None, :]
    return bf16(out)


def hc_combine_norm(residual, block_output, injection, next_norm_weight, epsilon):
    updated = hc_combine(residual, block_output, injection)
    return updated, hc_grouped_rmsnorm(updated, next_norm_weight, epsilon)


def hc_mix(normalized, down_injection, up, epsilon=None):
    norm = np.asarray(normalized, dtype=np.float64)
    tokens, streams, hidden = norm.shape
    channels = streams * hidden
    flat = norm.reshape(tokens, channels)
    down = np.asarray(down_injection, dtype=np.float64)
    rank = up.shape[1]
    low = flat @ down[:rank].T
    inject_rows = flat @ down[rank : rank + streams].T
    activated = silu(low / streams)
    gates = activated @ np.asarray(up, dtype=np.float64).T
    gates = gates.reshape(tokens, streams, hidden)
    block = np.sum(sigmoid(gates) * norm, axis=1) / streams
    injection = fp32(2.0 * sigmoid(inject_rows / streams))
    return bf16(block), injection


def hc_final_mix(normalized, down, up):
    norm = np.asarray(normalized, dtype=np.float64)
    tokens, streams, hidden = norm.shape
    channels = streams * hidden
    flat = norm.reshape(tokens, channels)
    low = flat @ np.asarray(down, dtype=np.float64).T
    activated = silu(low / streams)
    gates = activated @ np.asarray(up, dtype=np.float64).T
    gates = gates.reshape(tokens, streams, hidden)
    block = np.sum(sigmoid(gates) * norm, axis=1) / streams
    return bf16(block)


def rope(x, positions, theta, rotary_dim):
    xf = np.asarray(x, dtype=np.float64)
    half = rotary_dim // 2
    freq = np.power(float(theta), -2.0 * np.arange(half, dtype=np.float64) / rotary_dim)
    position = np.asarray(positions, dtype=np.float64)
    while position.ndim < xf.ndim:
        position = position[..., None]
    angle = position * freq
    cos = np.cos(angle)
    sin = np.sin(angle)
    out = xf.copy()
    lower = xf[..., :half]
    upper = xf[..., half:rotary_dim]
    out[..., :half] = lower * cos - upper * sin
    out[..., half:rotary_dim] = upper * cos + lower * sin
    return out


def qsa_norm_rope(x, norm_weight, positions, epsilon, theta, rotary_dim):
    return bf16(rope(rms_offset(x, norm_weight, epsilon), positions, theta, rotary_dim))


def qsa_index_block(raw_index_key, norm_weight, position, epsilon, theta, rotary_dim):
    mean = bf16(np.mean(np.asarray(raw_index_key, dtype=np.float64), axis=0))
    return qsa_norm_rope(mean[None, :], norm_weight, np.array([position]), epsilon, theta, rotary_dim)[0]


def qsa_indexer(index_query, index_key, positions, budget):
    query = np.asarray(index_query, dtype=np.float64)
    keys = np.asarray(index_key, dtype=np.float64)
    tokens = query.shape[0]
    counts = np.minimum((np.asarray(positions, dtype=np.int64) + 1) // 4, keys.shape[0])
    width = int(counts.max()) if tokens else 0
    scores = np.zeros((tokens, width), dtype=np.float64)
    for g in range(width):
        positive = np.maximum(0.0, np.einsum("thd,d->th", query, keys[g]))
        scores[:, g] = np.sum(positive, axis=1)
    arange = np.arange(width)
    scores = np.where(arange[None, :] >= counts[:, None], -np.inf, scores)
    selections = []
    for t in range(tokens):
        count = int(counts[t])
        selected = int(min(budget, count))
        order = np.lexsort((arange[:count], -scores[t, :count]))[:selected]
        selections.append(np.sort(order.astype(np.int32)))
    return selections


def qsa_sparse_gqa(query, output_gate, k_cache, v_cache, selections, positions, compress, head_dim):
    q = np.asarray(query, dtype=np.float64)
    gate = np.asarray(output_gate, dtype=np.float64)
    key = np.asarray(k_cache, dtype=np.float64)
    value = np.asarray(v_cache, dtype=np.float64)
    tokens, q_heads, _ = q.shape
    kv_heads = key.shape[1]
    group = q_heads // kv_heads
    out = np.zeros((tokens, q_heads, head_dim), dtype=np.float64)
    scale = 1.0 / math.sqrt(head_dim)
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
            attention = weights @ value[index, kvh, :]
            out[t, heads, :] = attention * sigmoid(gate[t, heads, :])
    return bf16(out)


def gdn_gating(x, a_weight, b_weight, a_log, dt_bias):
    xf = np.asarray(x, dtype=np.float64)
    a = xf @ np.asarray(a_weight, dtype=np.float64).T
    b = xf @ np.asarray(b_weight, dtype=np.float64).T
    softplus = np.logaddexp(0.0, a + np.asarray(dt_bias, dtype=np.float64)[None, :])
    g = -np.exp(np.asarray(a_log, dtype=np.float64))[None, :] * softplus
    beta = sigmoid(b)
    return fp32(g), fp32(beta)


def normalize_qk(x, epsilon):
    xf = np.asarray(x, dtype=np.float64)
    return xf / np.sqrt(np.sum(xf * xf, axis=-1, keepdims=True) + epsilon)


def gated_delta_net(q, k, v, g, beta, state, scale, epsilon):
    qf = np.asarray(q, dtype=np.float64)
    kf = np.asarray(k, dtype=np.float64)
    vf = np.asarray(v, dtype=np.float64)
    gf = np.asarray(g, dtype=np.float64)
    bf = np.asarray(beta, dtype=np.float64)
    tokens, qk_heads, dim = qf.shape
    value_heads = vf.shape[1]
    group = value_heads // qk_heads
    head_map = np.arange(value_heads, dtype=np.int64) // group
    qn = normalize_qk(qf, epsilon)[:, head_map, :]
    kn = normalize_qk(kf, epsilon)[:, head_map, :]
    matrix = np.array(state, dtype=np.float64, copy=True)
    out = np.zeros((tokens, value_heads, dim), dtype=np.float64)
    for t in range(tokens):
        alpha = np.exp(gf[t])
        sk = np.einsum("hij,hj->hi", matrix, kn[t])
        delta = bf[t][:, None] * (vf[t] - alpha[:, None] * sk)
        matrix = alpha[:, None, None] * matrix + delta[:, :, None] * kn[t][:, None, :]
        out[t] = scale * np.einsum("hij,hj->hi", matrix, qn[t])
    return out


def gated_rmsnorm(x, weight, z, epsilon):
    xf = np.asarray(x, dtype=np.float64)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + epsilon)
    out = xf * inv * np.asarray(weight, dtype=np.float64) * silu(z)
    return bf16(out)


def causal_conv_silu(x, kernel, history):
    xf = np.asarray(x, dtype=np.float64)
    tokens, channels = xf.shape
    hist = np.asarray(history, dtype=np.float64)
    window = hist.shape[1]
    out = np.zeros((tokens, channels), dtype=np.float64)
    for tap in range(kernel.shape[0]):
        weight = np.asarray(kernel[tap], dtype=np.float64)
        for t in range(tokens):
            source = t - window + tap
            if source >= 0:
                out[t] += weight * xf[source]
            else:
                out[t] += weight * hist[:, window + source]
    return bf16(silu(out))


def update_conv_history(history, x):
    xf = np.asarray(x, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64)
    window = hist.shape[1] if hist.ndim == 2 else hist.shape[0]
    combined = np.concatenate([hist, xf], axis=0)
    return bf16(combined[-window:])


def topk_route(x, control, k):
    xf = np.asarray(x, dtype=np.float64)
    logits = xf @ np.asarray(control, dtype=np.float64).T
    routed = logits[:, :-1]
    order = np.argsort(-routed, axis=1, kind="stable")[:, :k]
    selected = np.take_along_axis(routed, order, axis=1)
    selected = selected - np.max(selected, axis=1, keepdims=True)
    weights = np.exp(selected)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    shared = sigmoid(logits[:, -1])
    return order.astype(np.int32), fp32(weights), fp32(shared)


def swiglu(x, gate_up):
    xf = np.asarray(x, dtype=np.float64)
    projected = xf @ np.asarray(gate_up, dtype=np.float64).T
    half = projected.shape[-1] // 2
    return silu(projected[..., :half]) * projected[..., half:]


def ple_hash_rows(ids, tail, multipliers, vocab_sizes, offsets, eos):
    older_tail = int(tail[0])
    previous_tail = int(tail[1])
    rows = []
    for raw in ids:
        token = int(raw)
        older = eos if previous_tail == eos else older_tail
        pair = (token * int(multipliers[0])) ^ (previous_tail * int(multipliers[1]))
        triple = pair ^ (older * int(multipliers[2]))
        token_rows = []
        for head in range(16):
            word = pair if head < 8 else triple
            signed = word if word < (1 << 63) else word - (1 << 64)
            token_rows.append(signed % int(vocab_sizes[head]) + int(offsets[head]))
        rows.append(token_rows)
        older_tail, previous_tail = previous_tail, token
    return np.asarray(rows, dtype=np.uint32)


def e2m1_lut():
    return np.asarray([decode_e2m1_word(word) for word in range(16)], dtype=np.float64)


def e4m3_lut():
    return np.asarray([decode_e4m3fn_word(word) for word in range(256)], dtype=np.float64)


def ple_decode_row(row_bytes, divisor):
    raw = np.frombuffer(bytes(row_bytes), dtype=np.uint8)
    code_bytes = raw[: len(raw) - 10]
    scales = raw[len(raw) - 10 : len(raw)]
    low = code_bytes & 0x0F
    high = code_bytes >> 4
    codes = np.empty(code_bytes.size * 2, dtype=np.uint8)
    codes[0::2] = low
    codes[1::2] = high
    group = np.arange(codes.size) // 16
    values = e2m1_lut()[codes] * e4m3_lut()[scales[group]] / float(divisor)
    return bf16(values)


def ple_embedding(row_tables, rows):
    tokens = rows.shape[0]
    embedding = np.zeros((tokens, CFG.hidden), dtype=np.float64)
    for t in range(tokens):
        for head in range(CFG.ple_heads):
            start = head * CFG.ple_head_width
            embedding[t, start : start + CFG.ple_head_width] = row_tables[rows[t, head]]
    return bf16(embedding)


def ple_gate(residual, projected_key, projected_value, norm_key, norm_query, norm_conv, epsilon):
    res = np.asarray(residual, dtype=np.float64)
    tokens, streams, hidden = res.shape
    channels = streams * hidden
    key = np.asarray(projected_key, dtype=np.float64).reshape(tokens, streams, hidden)
    key_inv = 1.0 / np.sqrt(np.mean(key * key, axis=-1, keepdims=True) + epsilon)
    query_inv = 1.0 / np.sqrt(np.mean(res * res, axis=-1, keepdims=True) + epsilon)
    normalized_key = key * key_inv * (1.0 + np.asarray(norm_key, dtype=np.float64))[None, :, :]
    normalized_query = res * query_inv * (1.0 + np.asarray(norm_query, dtype=np.float64))[None, :, :]
    dot = np.sum(normalized_key * normalized_query, axis=-1) / math.sqrt(hidden)
    gate = sigmoid(np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), CFG.ple_gate_floor)))
    gated = bf16(gate[:, :, None] * np.asarray(projected_value, dtype=np.float64)[:, None, :])
    conv_inv = 1.0 / np.sqrt(np.mean(gated * gated, axis=-1, keepdims=True) + epsilon)
    normalized_conv = bf16(
        gated * conv_inv * (1.0 + np.asarray(norm_conv, dtype=np.float64))[None, :, :]
    )
    return bf16(gated.reshape(tokens, channels)), bf16(normalized_conv.reshape(tokens, channels))


def ple_conv_add_norm(history, normalized_conv, residual, gated_value, kernel, next_norm_weight, epsilon):
    values = np.asarray(normalized_conv, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)
    gated = np.asarray(gated_value, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64)
    tokens = values.shape[0]
    channels = values.shape[1]
    window = hist.shape[1]
    kernel = np.asarray(kernel, dtype=np.float64)
    conv = np.zeros((tokens, channels), dtype=np.float64)
    for t in range(tokens):
        for tap in range(kernel.shape[0]):
            source = t - window + tap * CFG.ple_conv_dilation
            if source >= 0:
                conv[t] += kernel[tap] * values[source]
            else:
                conv[t] += kernel[tap] * hist[:, window + source]
    updated = bf16(res.reshape(tokens, channels) + gated + silu(conv))
    streams = res.shape[1]
    hidden = res.shape[2]
    updated4 = updated.reshape(tokens, streams, hidden)
    inv = 1.0 / np.sqrt(np.mean(updated4 * updated4, axis=-1, keepdims=True) + epsilon)
    normalized = bf16(
        updated4 * inv * (1.0 + np.asarray(next_norm_weight, dtype=np.float64))[None, :, :]
    )
    return updated, normalized.reshape(tokens, channels)


def ple_state_extract(history, values, accepted):
    hist = np.asarray(history, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    window = hist.shape[1]
    after = np.zeros_like(hist)
    for j in range(window):
        k = accepted - window + j
        after[:, j] = hist[:, window + k] if k < 0 else vals[k]
    return bf16(after)


def _close(actual, expected, atol=1e-6):
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    return a.shape == e.shape and bool(np.allclose(a, e, rtol=0.0, atol=atol))


def run_self_tests(verbose=False):
    results = []

    def check(name, actual, expected, atol=1e-6):
        ok = _close(actual, expected, atol)
        results.append((name, ok))
        if verbose:
            print(("PASS " if ok else "FAIL ") + name)
            if not ok:
                print("  actual  ", np.asarray(actual).tolist())
                print("  expected", np.asarray(expected).tolist())

    residual, normalized = hc_expand_norm(
        [[2.0, 0.0, 0.0, 0.0]], np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]), 0.0
    )
    check(
        "hc_expand_norm",
        np.concatenate([residual.reshape(-1, 1), normalized.reshape(-1, 1)], axis=1),
        np.array([[2.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )

    residual = np.array([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]])
    check(
        "hc_grouped_rmsnorm",
        hc_grouped_rmsnorm(residual, np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]), 0.0),
        np.array([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]),
    )

    combined = hc_combine(
        np.array([[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[2.0, 4.0, 6.0, 8.0]]),
        np.array([[0.5, 2.0]]),
    )
    check(
        "hc_combine",
        combined,
        np.array([[[2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 12.0, 16.0]]]),
    )

    updated, combined_norm = hc_combine_norm(
        np.array([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]]),
        np.array([[2.0, 2.0, 2.0, 2.0]]),
        np.array([[0.0, 1.0]]),
        np.zeros((2, 4)),
        0.0,
    )
    check("hc_combine_norm.updated", updated, np.array([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]))
    check("hc_combine_norm.normalized", combined_norm, np.array([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]))

    norm4 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    block, injection = hc_mix(norm4, np.zeros((4, 4)), np.zeros((4, 2)))
    check("hc_mix.zero", np.concatenate([block, injection], axis=1), np.array([[1.0, 1.5, 1.0, 1.0]]))
    down = np.zeros((4, 4))
    down[0] = [2.0, 0.0, 0.0, 0.0]
    up = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    block, injection = hc_mix(norm4, down, up)
    gate = 0.6750375273768237
    check("hc_mix.silu", np.concatenate([block, injection], axis=1), np.array([[gate * 2.0, gate * 3.0, 1.0, 1.0]]), 1e-2)

    sample = hc_final_mix(norm4, down[:2], up)
    check("hc_final_mix", sample, np.array([[gate * 2.0, gate * 3.0]]), 1e-2)

    rotated = rope(
        np.array([[[1.0, 0.0, 10.0, 0.0]]]), np.array([1]), 1.0, 4
    )
    c1, s1 = 0.5403023058681398, 0.8414709848078965
    check("rope", rotated, np.array([[[c1 - 10.0 * s1, 0.0, 10.0 * c1 + s1, 0.0]]]), 1e-6)

    block_key = qsa_index_block(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]), np.zeros(2), 1, 0.0, 1.0, 2
    )
    inv = 1.0 / math.sqrt(20.5)
    check(
        "qsa_index_block",
        block_key,
        np.array([(4.0 * c1 - 5.0 * s1) * inv, (5.0 * c1 + 4.0 * s1) * inv]),
        1e-2,
    )

    index_q = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    index_k = np.array([[1.0, 0.0], [0.0, 1.0]])
    check("qsa_indexer.all", np.concatenate(qsa_indexer(index_q, index_k, np.array([7]), 2)), np.array([0, 1]))
    check("qsa_indexer.budget", np.concatenate(qsa_indexer(index_q, index_k, np.array([7]), 1)), np.array([0]))

    sparse = qsa_sparse_gqa(
        np.array([[[1.0, 0.0]]]),
        np.zeros((1, 1, 2)),
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        [np.array([], dtype=np.int32)],
        np.array([1]),
        4,
        2,
    )
    check("qsa_sparse_gqa", sparse, np.array([[[0.33488077466332844, 0.16511922533667156]]]), 2e-3)

    g, beta = gdn_gating(
        np.array([[1.0, 0.0]]),
        np.array([[2.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([math.log(3.0), math.log(3.0)]),
        np.array([0.0, 0.0]),
    )
    check("gdn_gating.g", g, np.array([[-6.3807840331289185, -2.0794415416798357]]), 1e-6)
    check("gdn_gating.beta", beta, np.array([[0.5, 0.5]]), 1e-6)

    recurrence = gated_delta_net(
        np.array([[[1.0, 0.0]]]),
        np.array([[[1.0, 0.0]]]),
        np.array([[[1.0, 0.0]]]),
        np.zeros((1, 1)),
        np.ones((1, 1)),
        np.zeros((1, 2, 2)),
        128 ** -0.5,
        1e-6,
    )
    check("gated_delta_net", recurrence, np.array([[[0.08838825926005918, 0.0]]]), 1e-6)

    check(
        "gated_rmsnorm",
        gated_rmsnorm(np.array([[[1.0, 1.0]]]), np.array([1.0, 1.0]), np.array([[[1.0, 1.0]]]), 0.0),
        np.array([[[0.7310585786300049, 0.7310585786300049]]]),
        1e-2,
    )

    ids, coefficients, shared = topk_route(
        np.array([[1.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        1,
    )
    check("topk_route.ids", ids, np.array([[0]], dtype=np.int32))
    check("topk_route.coefficients", coefficients, np.array([[1.0]]))
    check("topk_route.shared", shared, np.array([0.5]))
    ids, coefficients, shared = topk_route(
        np.array([[1.0, 1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
        2,
    )
    check("topk_route.tie", ids, np.array([[0, 1]], dtype=np.int32))
    check("topk_route.tie_coefficients", coefficients, np.array([[0.5, 0.5]]))

    controls = (
        np.array([1, 1, 1], dtype=np.uint64),
        np.array([7] * 16, dtype=np.uint64),
        np.array([7 * head for head in range(16)], dtype=np.uint64),
    )
    hashed = ple_hash_rows([2], (0, 0), *controls, 0)
    check("ple_hash_rows", hashed, np.array([[2 + 7 * head for head in range(16)]], dtype=np.uint64))
    negative = ple_hash_rows([1 << 63], (0, 0), *controls, 0)
    check("ple_hash_rows.negative", negative, np.array([[6 + 7 * head for head in range(16)]], dtype=np.uint64))

    row = bytearray(90)
    for index in range(80):
        low = index % 8
        high = (index + 1) % 8
        row[index] = low | (high << 4)
    for index in range(10):
        row[80 + index] = 0x38
    decoded = ple_decode_row(row, 2.0)
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected = []
    for index in range(80):
        expected.extend([magnitudes[index % 8] / 2.0, magnitudes[(index + 1) % 8] / 2.0])
    check("ple_decode_row", decoded, np.array(expected), 1e-6)

    ones = np.ones((1, 2, 4))
    gated, normalized_conv = ple_gate(
        ones, np.ones((1, 8)), np.ones((1, 4)), np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)), 0.0
    )
    gate_value = 0.8044296825069569
    check("ple_gate", np.concatenate([gated, normalized_conv], axis=1), np.array([[gate_value] * 8 + [1.0] * 8]), 1e-2)

    kernel = np.zeros((4, 8))
    kernel[3] = 1.0
    updated, normalized = ple_conv_add_norm(
        np.zeros((8, 9)),
        np.ones((4, 8)),
        np.zeros((4, 2, 4)),
        np.ones((4, 8)),
        kernel,
        np.zeros((2, 4)),
        0.0,
    )
    expected_updated = np.full((4, 8), 1.0 + 0.7310585786300049)
    check("ple_conv_add_norm.updated", updated, expected_updated, 1e-2)
    check("ple_conv_add_norm.normalized", normalized, np.ones((4, 8)), 1e-6)

    after = ple_state_extract(np.arange(8 * 9, dtype=np.float64).reshape(8, 9), np.arange(8 * 4, dtype=np.float64).reshape(4, 8), 3)
    expected_after = np.zeros((8, 9))
    for j in range(9):
        k = 3 - 9 + j
        expected_after[:, j] = np.arange(8 * 9).reshape(8, 9)[:, 9 + k] if k < 0 else np.arange(8 * 4).reshape(4, 8)[k]
    check("ple_state_extract", after, expected_after)

    return results


def self_test_exit_code(verbose=False):
    results = run_self_tests(verbose)
    failures = [name for name, ok in results if not ok]
    if verbose and failures:
        print("failures: " + ", ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(self_test_exit_code(True))
