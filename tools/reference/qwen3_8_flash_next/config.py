from __future__ import annotations

import os
from dataclasses import dataclass

_DIAG_ROPE_THETA = os.environ.get("NINFER_DIAG_ROPE_THETA")
_DIAG_ROTARY_DIM = os.environ.get("NINFER_DIAG_ROTARY_DIM")
_DIAG_RMS_EPS = os.environ.get("NINFER_DIAG_RMS_EPS")
_DIAG_GDN_SCALE = os.environ.get("NINFER_DIAG_GDN_SCALE")


@dataclass(frozen=True)
class FlashNextConfig:
    hidden: int = 2560
    layers: int = 4
    full_layers: int = 48
    vocab: int = 248320
    hc_streams: int = 4
    hc_lowrank: int = 320
    q_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    index_heads: int = 4
    index_kv_heads: int = 1
    index_head_dim: int = 128
    index_compress: int = 4
    index_budget: int = 2048
    gdn_k_heads: int = 16
    gdn_v_heads: int = 48
    gdn_head_dim: int = 128
    gdn_conv_kernel: int = 4
    experts: int = 512
    experts_per_token: int = 10
    expert_intermediate: int = 640
    shared_intermediate: int = 640
    ple_layer: int = 1
    ple_ngram: int = 3
    ple_heads_per_ngram: int = 8
    ple_shards: int = 128
    ple_shard_rows: int = 2500012
    ple_head_width: int = 160
    ple_conv_kernel: int = 4
    ple_conv_dilation: int = 3
    ple_conv_window: int = 9
    rope_theta: float = float(_DIAG_ROPE_THETA) if _DIAG_ROPE_THETA is not None else 1.0e7
    rotary_dim: int = int(_DIAG_ROTARY_DIM) if _DIAG_ROTARY_DIM is not None else 64
    partial_rotary: float = 0.25
    rms_epsilon: float = float(_DIAG_RMS_EPS) if _DIAG_RMS_EPS is not None else 1.0e-6
    ple_gate_floor: float = 1.0e-6
    eos_token_id: int = 248044

    @classmethod
    def full(cls) -> "FlashNextConfig":
        return cls(layers=cls.full_layers)

    @property
    def is_full(self) -> bool:
        return self.layers == self.full_layers

    @property
    def residual_width(self) -> int:
        return self.hidden * self.hc_streams

    @property
    def query_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def index_query_width(self) -> int:
        return self.index_heads * self.index_head_dim

    @property
    def index_key_width(self) -> int:
        return self.index_kv_heads * self.index_head_dim

    @property
    def index_block_budget(self) -> int:
        return self.index_budget // self.index_compress

    @property
    def gdn_key_width(self) -> int:
        return self.gdn_k_heads * self.gdn_head_dim

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_v_heads * self.gdn_head_dim

    @property
    def gdn_conv_channels(self) -> int:
        return 2 * self.gdn_key_width + self.gdn_value_width

    @property
    def ple_heads(self) -> int:
        return (self.ple_ngram - 1) * self.ple_heads_per_ngram

    @property
    def gdn_scale(self) -> float:
        if _DIAG_GDN_SCALE is not None:
            return float(_DIAG_GDN_SCALE)
        return self.gdn_head_dim ** -0.5

    def is_qsa_layer(self, layer: int) -> bool:
        return layer % 4 == 3


CFG = FlashNextConfig()
