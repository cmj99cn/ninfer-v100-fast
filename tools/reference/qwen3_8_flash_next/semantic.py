"""Environment-driven semantic-diagnostic switches (tools-only, default = current behavior).

Every knob defaults to the shipped transcription so the 4-layer baseline is unchanged.
The parity driver sets one knob per subprocess; no product code is involved.
"""

from __future__ import annotations

import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


SEM = {
    "rope_mode": _get("NINFER_DIAG_ROPE_MODE", "halfsplit"),
    "rope_sign_flip": _get("NINFER_DIAG_ROPE_SIGN_FLIP", "0") == "1",
    "rope_pos_offset": int(_get("NINFER_DIAG_ROPE_POS_OFFSET", "0")),
    "rope_head_pos": _get("NINFER_DIAG_ROPE_HEAD_POS", "same"),
    "hc_mix_mode": _get("NINFER_DIAG_HC_MIX_MODE", "current"),
    "hc_combine_mode": _get("NINFER_DIAG_HC_COMBINE_MODE", "current"),
    "hc_stream_perm": _get("NINFER_DIAG_HC_STREAM_PERM", "identity"),
    "hc_norm_mode": _get("NINFER_DIAG_HC_NORM_MODE", "per_stream"),
    "hc_norm_gain": _get("NINFER_DIAG_HC_NORM_GAIN", "one_plus_w"),
    "hc_combine_order": _get("NINFER_DIAG_HC_COMBINE_ORDER", "update_then_norm"),
    "hc_block_scale": _get("NINFER_DIAG_HC_BLOCK_SCALE", "none"),
    "hc_residual_round": _get("NINFER_DIAG_HC_RESIDUAL_ROUND", "round"),
    "hc_inject_coeff": _get("NINFER_DIAG_HC_INJECT_COEFF", "default"),
    "gdn_gate_mode": _get("NINFER_DIAG_GDN_GATE_MODE", "current"),
    "gdn_update_mode": _get("NINFER_DIAG_GDN_UPDATE_MODE", "current"),
    "gdn_conv_mode": _get("NINFER_DIAG_GDN_CONV_MODE", "current"),
    "gdn_norm_gate": _get("NINFER_DIAG_GDN_NORM_GATE", "sigmoid"),
    "gdn_norm_gain": _get("NINFER_DIAG_GDN_NORM_GAIN", "w"),
    "rope_full_dim": _get("NINFER_DIAG_ROPE_FULL_DIM", "0") == "1",
    "ablate": _get("NINFER_DIAG_ABLATE", ""),
}
