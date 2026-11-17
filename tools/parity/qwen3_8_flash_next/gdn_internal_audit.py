"""GDN layer-0 internal per-quantity audit (diagnostic only).

Dumps every intermediate of the FP64 reference GDN for a given prompt, and
independently recomputes the same intermediates from the upstream-authoritative
Qwen4ExpTextGatedDeltaNet formula (tfm_modeling_qwen4_exp.py), using the SAME
artifact weight arrays and the SAME layer-0 GDN input.  Compares q/k/v -> z/g/
beta -> conv -> state S -> o_recurrent -> o_normalized -> mixer output.

No product code is imported; only tools.reference and tools.artifact are used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.reference.qwen3_8_flash_next.config import FlashNextConfig
from tools.reference.qwen3_8_flash_next.model import RefModel
from tools.reference.qwen3_8_flash_next.ops import (
    bf16,
    causal_conv_silu,
    dense,
    gated_delta_net,
    gated_rmsnorm,
    gdn_gating,
    normalize_qk,
    sigmoid,
    silu,
)
from tools.reference.qwen3_8_flash_next.state import GdnState


def to_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def reference_dump(model, weights, hidden, state, cfg):
    """Replicates tools/reference/.../model.py::_gdn, capturing intermediates."""
    tokens = hidden.shape[0]
    q = dense(hidden, weights["q"])
    k = dense(hidden, weights["k"])
    v = dense(hidden, weights["v"])
    z = dense(hidden, weights["z"])
    g, beta = gdn_gating(hidden, weights["a"], weights["b"], weights["a_log"], weights["dt_bias"])

    conv_in = np.empty((tokens, cfg.gdn_conv_channels), dtype=np.float64)
    conv_in[:, 0 : cfg.gdn_key_width] = q
    conv_in[:, cfg.gdn_key_width : 2 * cfg.gdn_key_width] = k
    conv_in[:, 2 * cfg.gdn_key_width :] = v
    conv = causal_conv_silu(conv_in, weights["conv"], state.history)

    q_conv = conv[:, 0 : cfg.gdn_key_width].reshape(tokens, cfg.gdn_k_heads, cfg.gdn_head_dim)
    k_conv = conv[:, cfg.gdn_key_width : 2 * cfg.gdn_key_width].reshape(tokens, cfg.gdn_k_heads, cfg.gdn_head_dim)
    v_conv = conv[:, 2 * cfg.gdn_key_width :].reshape(tokens, cfg.gdn_v_heads, cfg.gdn_head_dim)
    z_view = z.reshape(tokens, cfg.gdn_v_heads, cfg.gdn_head_dim)

    out, states = gated_delta_net_capture(
        q_conv, k_conv, v_conv, g, beta, state.recurrent, cfg.gdn_scale, cfg.rms_epsilon
    )
    normalized = gated_rmsnorm(out, weights["norm"], z_view, cfg.rms_epsilon)
    mixer = dense(normalized.reshape(tokens, cfg.gdn_value_width), weights["out"])
    return {
        "q": q,
        "k": k,
        "v": v,
        "z": z,
        "g": g,
        "beta": beta,
        "conv_in": conv_in,
        "conv": conv,
        "q_conv": q_conv,
        "k_conv": k_conv,
        "v_conv": v_conv,
        "z_view": z_view,
        "state_init": state.recurrent,
        "state_per_token": states,
        "o_recurrent": out,
        "o_normalized": normalized,
        "mixer": mixer,
    }


def gated_delta_net_capture(q, k, v, g, beta, state, scale, epsilon):
    """Copy of reference gated_delta_net with per-token state capture."""
    qf, kf, vf = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    gf, bf = np.asarray(g, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    tokens, qk_heads, dim = qf.shape
    value_heads = vf.shape[1]
    group = value_heads // qk_heads
    head_map = np.arange(value_heads, dtype=np.int64) // group
    qn = normalize_qk(qf, epsilon)[:, head_map, :]
    kn = normalize_qk(kf, epsilon)[:, head_map, :]
    write_key, write_val = kn, vf
    matrix = np.array(state, dtype=np.float64, copy=True)
    out = np.zeros((tokens, value_heads, dim), dtype=np.float64)
    states = []
    for t in range(tokens):
        alpha = np.exp(gf[t])
        sk = np.einsum("hij,hj->hi", matrix, write_key[t])
        delta = bf[t][:, None] * (write_val[t] - alpha[:, None] * sk)
        matrix = alpha[:, None, None] * matrix + delta[:, :, None] * write_key[t][:, None, :]
        states.append(matrix.copy())
        out[t] = scale * np.einsum("hij,hj->hi", matrix, qn[t])
    return out, np.stack(states)


def l2norm_upstream(x, eps=1e-6):
    """fla l2norm: x / sqrt(sum(x^2)+eps) over last dim."""
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def upstream_dump(weights, hidden, state, cfg):
    """Independent recomputation of tfm_modeling_qwen4_exp.py Qwen4ExpTextGatedDeltaNet.

    Uses the SAME weight arrays and layer-0 GDN input as reference_dump, but the
    math is transcribed directly from upstream modeling (not from our ops.py).
    """
    x = np.asarray(hidden, dtype=np.float64)
    tokens = x.shape[0]
    kd, vd = cfg.gdn_k_heads, cfg.gdn_v_heads
    hd = cfg.gdn_head_dim
    kw, vw = cfg.gdn_key_width, cfg.gdn_value_width
    group = vd // kd

    # in_proj_qkv / in_proj_z / in_proj_a / in_proj_b (bias=False in upstream).
    w_qkv = np.concatenate([weights["q"], weights["k"], weights["v"]], axis=0)
    mixed_qkv = x @ w_qkv.T  # identical to concatenating reference dense(q),dense(k),dense(v)
    q_raw = mixed_qkv[:, 0:kw]
    k_raw = mixed_qkv[:, kw : 2 * kw]
    v_raw = mixed_qkv[:, 2 * kw :]
    z = x @ weights["z"].T
    a = x @ weights["a"].T
    b = x @ weights["b"].T

    # causal_conv1d_fn: depthwise causal conv (padding=k-1) + activation(silu).
    kernel = weights["conv"]  # (taps, channels), upstream conv1d.weight[c, tap] = kernel[tap, c]
    taps = kernel.shape[0]
    window = taps - 1
    hist = np.asarray(state.history, dtype=np.float64)
    conv = np.zeros_like(mixed_qkv)
    for tap in range(taps):
        for t in range(tokens):
            src = t - window + tap
            if src >= 0:
                conv[t] += kernel[tap] * mixed_qkv[src]
            else:
                conv[t] += kernel[tap] * hist[:, window + src]
    conv = silu(conv)
    q = conv[:, 0:kw].reshape(tokens, kd, hd)
    k = conv[:, kw : 2 * kw].reshape(tokens, kd, hd)
    v = conv[:, 2 * kw :].reshape(tokens, vd, hd)
    z = z.reshape(tokens, vd, hd)

    beta = sigmoid(b)
    softplus = np.logaddexp(0.0, a + np.asarray(weights["dt_bias"], np.float64)[None, :])
    g = -np.exp(np.asarray(weights["a_log"], np.float64))[None, :] * softplus

    # torch_recurrent_gated_delta_rule, use_qk_l2norm_in_kernel=True
    qn = l2norm_upstream(q, cfg.rms_epsilon)
    kn = l2norm_upstream(k, cfg.rms_epsilon)
    qn = np.repeat(qn, group, axis=1)
    kn = np.repeat(kn, group, axis=1)
    qn = qn / math.sqrt(hd)

    S = np.zeros((vd, hd, hd), dtype=np.float64)
    out = np.zeros((tokens, vd, hd), dtype=np.float64)
    states = []
    for i in range(tokens):
        decay = np.exp(g[i])[:, None, None]
        S = S * decay
        kv_mem = np.einsum("hij,hj->hi", S, kn[i])
        delta = (v[i] - kv_mem) * beta[i][:, None]
        S = S + kn[i][:, None, :] * delta[:, :, None]
        states.append(S.copy())
        out[i] = np.einsum("hij,hj->hi", S, qn[i])

    # Qwen4ExpTextRMSNormGated.forward: weight * norm(x) * activation(z), activation=sigmoid
    inv = 1.0 / np.sqrt(np.mean(out * out, axis=-1, keepdims=True) + cfg.rms_epsilon)
    normalized = np.asarray(weights["norm"], np.float64)[None, None, :] * (out * inv) * sigmoid(z)
    nflat = normalized.reshape(tokens, vw)
    mixer = nflat @ weights["out"].T

    return {
        "q": q_raw,
        "k": k_raw,
        "v": v_raw,
        "z": z,
        "g": g,
        "beta": beta,
        "conv": conv,
        "q_conv": q,
        "k_conv": k,
        "v_conv": v,
        "z_view": z,
        "state_init": np.zeros_like(S),
        "state_per_token": np.stack(states),
        "o_recurrent": out,
        "o_normalized": normalized,
        "mixer": mixer,
    }


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def compare(ref, up, name):
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(up, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    ratio = nb / (na + 1e-30)
    cos = cosine(a, b)
    diff = float(np.max(np.abs(a.ravel() - b.ravel())))
    return {"quantity": name, "norm_ref": na, "norm_up": nb, "ratio_up/ref": ratio, "cos": cos, "max_abs_diff": diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    cfg = FlashNextConfig(layers=1)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with RefModel(args.artifact, config=cfg) as model:
        cap = model.forward(tokens, capture_tensors=True)
        hidden = np.asarray(cap["tensors"][(0, "attention_block")], dtype=np.float64)
        weights = model._gdn_weights(0)
        gstate = GdnState.fresh()
        ref = reference_dump(model, weights, hidden, gstate, cfg)

    up = upstream_dump(weights, hidden, gstate, cfg)

    quantities = [
        "q", "k", "v", "z", "g", "beta", "conv", "q_conv", "k_conv", "v_conv",
        "state_per_token", "o_recurrent", "o_normalized", "mixer",
    ]
    rows = [compare(ref[n], up[n], n) for n in quantities]

    dump = {
        "artifact": args.artifact,
        "tokens": tokens,
        "hidden_norm": float(np.linalg.norm(hidden)),
        "quantities": {n: {"reference": to_json(ref[n]), "upstream": to_json(up[n])} for n in quantities},
        "comparison": rows,
    }
    (outdir / "gdn_layer0_dump.json").write_text(json.dumps(dump))
    (outdir / "gdn_layer0_compare.json").write_text(json.dumps(rows, indent=2))

    print(f"tokens={tokens}  hidden_norm={np.linalg.norm(hidden):.6f}")
    print(f"{'quantity':<16} {'||ref||':>12} {'||up||':>12} {'up/ref':>9} {'cos':>9} {'max|diff|':>12}")
    for r in rows:
        print(f"{r['quantity']:<16} {r['norm_ref']:>12.5f} {r['norm_up']:>12.5f} "
              f"{r['ratio_up/ref']:>9.5f} {r['cos']:>9.5f} {r['max_abs_diff']:>12.3e}")


if __name__ == "__main__":
    main()
