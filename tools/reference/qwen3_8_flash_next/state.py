from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CFG


@dataclass
class GdnState:
    history: np.ndarray
    recurrent: np.ndarray

    @staticmethod
    def fresh():
        return GdnState(
            np.zeros((CFG.gdn_conv_channels, CFG.gdn_conv_kernel - 1), dtype=np.float32),
            np.zeros((CFG.gdn_v_heads, CFG.gdn_head_dim, CFG.gdn_head_dim), dtype=np.float64),
        )


@dataclass
class QsaState:
    key_cache: np.ndarray
    value_cache: np.ndarray
    index_key: np.ndarray
    index_query: np.ndarray
    query: np.ndarray
    output_gate: np.ndarray

    @staticmethod
    def empty():
        return QsaState(
            np.zeros((0, CFG.kv_heads, CFG.head_dim), dtype=np.float32),
            np.zeros((0, CFG.kv_heads, CFG.head_dim), dtype=np.float32),
            np.zeros((0, CFG.index_head_dim), dtype=np.float32),
            np.zeros((0, CFG.index_heads, CFG.index_head_dim), dtype=np.float32),
            np.zeros((0, CFG.q_heads, CFG.head_dim), dtype=np.float32),
            np.zeros((0, CFG.q_heads, CFG.head_dim), dtype=np.float32),
        )


@dataclass
class PleState:
    history: np.ndarray = field(
        default_factory=lambda: np.zeros((CFG.residual_width, CFG.ple_conv_window), dtype=np.float32)
    )
    tail: tuple[int, int] = (CFG.eos_token_id, CFG.eos_token_id)


@dataclass
class ModelState:
    gdn: dict[int, GdnState] = field(default_factory=dict)
    qsa: dict[int, QsaState] = field(default_factory=dict)
    ple: PleState = field(default_factory=PleState)

    @staticmethod
    def fresh():
        return ModelState(
            {layer: GdnState.fresh() for layer in range(CFG.full_layers) if not CFG.is_qsa_layer(layer)},
            {layer: QsaState.empty() for layer in range(CFG.full_layers) if CFG.is_qsa_layer(layer)},
            PleState(),
        )
