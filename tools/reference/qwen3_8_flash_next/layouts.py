"""Fused-object row-layout candidates for the Flash-Next diagnostic search.

The artifact fuses several upstream parameters into one row-major parent object:

* the QSA `query` and `output_gate` share one ``[2 * query_width, hidden]`` parent,
* the GDN `q`, `k`, `v` and `z` share one ``[2 * (key_width + value_width), hidden]`` parent,
* the GDN `a` and `b` control projections share one ``[2 * control_rows, hidden]`` parent,
* the MoE `router`, `shared_score`, `shared_gate` and `shared_up` share one
  ``[experts + 1 + 2 * shared_width, hidden]`` parent,
* the QSA `key`, `value`, `index_query` and `index_key` share one parent,
* every HC `down` parent holds the 320 down rows and the 4 injection rows.

This module parameterises the role -> row-interval mapping so that a diagnostic run can
substitute one candidate mapping at a time and measure the consequence.  The default
layout reproduces the mapping that the current artifact and the current product code use,
bit for bit; it is not a second opinion about that mapping.

Every candidate is a pure row permutation of a parent object: no value is scaled, rounded
or reinterpreted.  A candidate that restores the model therefore identifies a row order
error that is fixable by reordering rows inside the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def _rows(begin: int, count: int) -> np.ndarray:
    return np.arange(begin, begin + count, dtype=np.int64)


@dataclass(frozen=True)
class Layout:
    """One role -> row-interval assignment for every fused parent."""

    name: str = "current"
    qsa_query_gate: str = "query_then_gate"
    gdn_qkvz: str = "q_k_v_z"
    gdn_control: str = "a_then_b"
    conv_order: str = "tap_channel"
    moe_dense: str = "router_score_gate_up"
    qsa_kv: str = "k_v_indexq_indexk"
    hc_inject: str = "down_first"
    output_head: str = "identity"
    value_head_seed: int = 0

    def with_(self, **changes) -> "Layout":
        return replace(self, **changes)

    # -- QSA: query and output gate ------------------------------------------
    def qsa_query_gate_rows(self, cfg) -> tuple[np.ndarray, np.ndarray]:
        width = cfg.query_width
        dim = cfg.head_dim
        if self.qsa_query_gate == "query_then_gate":
            return _rows(0, width), _rows(width, width)
        if self.qsa_query_gate == "gate_then_query":
            return _rows(width, width), _rows(0, width)
        if self.qsa_query_gate == "head_interleaved":
            query = np.empty(width, dtype=np.int64)
            gate = np.empty(width, dtype=np.int64)
            for head in range(cfg.q_heads):
                base = head * (2 * dim)
                query[head * dim : (head + 1) * dim] = _rows(base, dim)
                gate[head * dim : (head + 1) * dim] = _rows(base + dim, dim)
            return query, gate
        raise ValueError("unknown qsa_query_gate " + self.qsa_query_gate)

    # -- QSA: key, value, index query, index key ------------------------------
    def qsa_kv_rows(self, cfg) -> dict[str, np.ndarray]:
        kv = cfg.kv_width
        index_query = cfg.index_query_width
        index_key = cfg.index_key_width
        if self.qsa_kv == "k_v_indexq_indexk":
            return {
                "key": _rows(0, kv),
                "value": _rows(kv, kv),
                "index_query": _rows(2 * kv, index_query),
                "index_key": _rows(2 * kv + index_query, index_key),
            }
        if self.qsa_kv == "k_v_indexk_indexq":
            return {
                "key": _rows(0, kv),
                "value": _rows(kv, kv),
                "index_key": _rows(2 * kv, index_key),
                "index_query": _rows(2 * kv + index_key, index_query),
            }
        if self.qsa_kv == "v_k_indexq_indexk":
            return {
                "key": _rows(kv, kv),
                "value": _rows(0, kv),
                "index_query": _rows(2 * kv, index_query),
                "index_key": _rows(2 * kv + index_query, index_key),
            }
        raise ValueError("unknown qsa_kv " + self.qsa_kv)

    # -- GDN: q, k, v, z -------------------------------------------------------
    def gdn_rows(self, cfg) -> dict[str, np.ndarray]:
        key_width = cfg.gdn_key_width
        value_width = cfg.gdn_value_width
        key_dim = cfg.gdn_head_dim
        key_heads = cfg.gdn_k_heads
        value_heads = cfg.gdn_v_heads
        option = self.gdn_qkvz
        if option == "q_k_v_z":
            return {
                "query": _rows(0, key_width),
                "key": _rows(key_width, key_width),
                "value": _rows(2 * key_width, value_width),
                "z": _rows(2 * key_width + value_width, value_width),
            }
        if option == "k_q_v_z":
            return {
                "query": _rows(key_width, key_width),
                "key": _rows(0, key_width),
                "value": _rows(2 * key_width, value_width),
                "z": _rows(2 * key_width + value_width, value_width),
            }
        if option == "q_k_z_v":
            return {
                "query": _rows(0, key_width),
                "key": _rows(key_width, key_width),
                "value": _rows(2 * key_width + value_width, value_width),
                "z": _rows(2 * key_width, value_width),
            }
        if option in ("qk_interleaved_vz_segmented", "qk_interleaved_vz_interleaved"):
            query = np.empty(key_width, dtype=np.int64)
            key = np.empty(key_width, dtype=np.int64)
            for head in range(key_heads):
                base = head * (2 * key_dim)
                query[head * key_dim : (head + 1) * key_dim] = _rows(base, key_dim)
                key[head * key_dim : (head + 1) * key_dim] = _rows(base + key_dim, key_dim)
        else:
            query = _rows(0, key_width)
            key = _rows(key_width, key_width)
        if option in ("q_k_v_z", "qk_interleaved_vz_segmented"):
            value = _rows(2 * key_width, value_width)
            z = _rows(2 * key_width + value_width, value_width)
        elif option in ("q_k_vz_interleaved", "qk_interleaved_vz_interleaved"):
            value = np.empty(value_width, dtype=np.int64)
            z = np.empty(value_width, dtype=np.int64)
            heads = np.arange(value_heads)
            if self.value_head_seed:
                heads = np.random.default_rng(self.value_head_seed).permutation(value_heads)
            for head in range(value_heads):
                source = int(heads[head])
                base = 2 * key_width + source * (2 * key_dim)
                value[head * key_dim : (head + 1) * key_dim] = _rows(base, key_dim)
                z[head * key_dim : (head + 1) * key_dim] = _rows(base + key_dim, key_dim)
        elif option == "q_k_vz_grouped16":
            # 48 value heads arranged as 16 key-head groups of 3; each group stores
            # its 3 v rows then its 3 z rows (block interleaving at group granularity).
            value = np.empty(value_width, dtype=np.int64)
            z = np.empty(value_width, dtype=np.int64)
            per_group = value_heads // key_heads
            for group in range(key_heads):
                group_base = 2 * key_width + group * (per_group * 2 * key_dim)
                for sub in range(per_group):
                    head = group * per_group + sub
                    value[head * key_dim : (head + 1) * key_dim] = _rows(group_base + sub * key_dim, key_dim)
                    z[head * key_dim : (head + 1) * key_dim] = _rows(
                        group_base + per_group * key_dim + sub * key_dim, key_dim
                    )
        else:
            raise ValueError("unknown gdn_qkvz " + option)
        return {"query": query, "key": key, "value": value, "z": z}

    # -- GDN: a and b control projections --------------------------------------
    def gdn_control_rows(self, cfg) -> dict[str, np.ndarray]:
        count = cfg.gdn_v_heads
        if self.gdn_control == "a_then_b":
            return {"a": _rows(0, count), "b": _rows(count, count)}
        if self.gdn_control == "interleaved":
            a = np.empty(count, dtype=np.int64)
            b = np.empty(count, dtype=np.int64)
            for index in range(count):
                a[index] = 2 * index
                b[index] = 2 * index + 1
            return {"a": a, "b": b}
        raise ValueError("unknown gdn_control " + self.gdn_control)

    # -- GDN: causal convolution kernel ----------------------------------------
    def conv_kernel(self, stored: np.ndarray) -> np.ndarray:
        kernel = np.asarray(stored)
        if self.conv_order == "tap_channel":
            return kernel
        if self.conv_order == "channel_tap":
            taps, channels = kernel.shape
            return kernel.reshape(channels, taps).T.copy()
        raise ValueError("unknown conv_order " + self.conv_order)

    # -- MoE: router, shared score, shared gate, shared up ---------------------
    def moe_rows(self, cfg) -> dict[str, np.ndarray]:
        experts = cfg.experts
        shared = cfg.shared_intermediate
        option = self.moe_dense
        if option == "router_score_gate_up":
            return {
                "router": _rows(0, experts),
                "score": _rows(experts, 1),
                "gate": _rows(experts + 1, shared),
                "up": _rows(experts + 1 + shared, shared),
            }
        if option == "score_at_end":
            return {
                "router": _rows(0, experts),
                "score": _rows(experts + 2 * shared, 1),
                "gate": _rows(experts, shared),
                "up": _rows(experts + shared, shared),
            }
        if option == "gate_up_swapped":
            return {
                "router": _rows(0, experts),
                "score": _rows(experts, 1),
                "gate": _rows(experts + 1 + shared, shared),
                "up": _rows(experts + 1, shared),
            }
        if option == "score_at_end_gate_up_swapped":
            return {
                "router": _rows(0, experts),
                "score": _rows(experts + 2 * shared, 1),
                "gate": _rows(experts + shared, shared),
                "up": _rows(experts, shared),
            }
        if option == "score_first":
            return {
                "router": _rows(1, experts),
                "score": _rows(0, 1),
                "gate": _rows(experts + 1, shared),
                "up": _rows(experts + 1 + shared, shared),
            }
        raise ValueError("unknown moe_dense " + option)

    # -- HC: down rows and injection rows --------------------------------------
    def hc_rows(self, cfg) -> dict[str, np.ndarray]:
        rank = cfg.hc_lowrank
        streams = cfg.hc_streams
        if self.hc_inject == "down_first":
            return {"down": _rows(0, rank), "inject": _rows(rank, streams)}
        if self.hc_inject == "inject_first":
            return {"down": _rows(streams, rank), "inject": _rows(0, streams)}
        raise ValueError("unknown hc_inject " + self.hc_inject)

    # -- output head: row permutation ------------------------------------------
    def output_head_rows(self, cfg) -> np.ndarray | None:
        vocab = cfg.vocab
        if self.output_head == "identity":
            return None
        if self.output_head == "swap_halves":
            half = vocab // 2
            return np.concatenate([_rows(half, half), _rows(0, half)])
        if self.output_head == "reverse_blocks":
            block = 256
            blocks = vocab // block
            order = np.arange(blocks, dtype=np.int64)[::-1]
            base = (order[:, None] * block + np.arange(block, dtype=np.int64)[None, :]).reshape(-1)
            return base
        raise ValueError("unknown output_head " + self.output_head)


CURRENT = Layout()


CANDIDATES: dict[str, dict[str, str]] = {
    "current": {},
    "qsa_head_interleaved": {"qsa_query_gate": "head_interleaved"},
    "qsa_gate_then_query": {"qsa_query_gate": "gate_then_query"},
    "gdn_vz_interleaved": {"gdn_qkvz": "q_k_vz_interleaved"},
    "gdn_vz_grouped16": {"gdn_qkvz": "q_k_vz_grouped16"},
    "gdn_qk_vz_interleaved": {"gdn_qkvz": "qk_interleaved_vz_interleaved"},
    "gdn_qk_interleaved": {"gdn_qkvz": "qk_interleaved_vz_segmented"},
    "gdn_qk_swapped": {"gdn_qkvz": "k_q_v_z"},
    "gdn_vz_swapped": {"gdn_qkvz": "q_k_z_v"},
    "gdn_ab_interleaved": {"gdn_control": "interleaved"},
    "conv_channel_tap": {"conv_order": "channel_tap"},
    "moe_score_at_end": {"moe_dense": "score_at_end"},
    "moe_gate_up_swapped": {"moe_dense": "gate_up_swapped"},
    "moe_score_at_end_swapped": {"moe_dense": "score_at_end_gate_up_swapped"},
    "moe_score_first": {"moe_dense": "score_first"},
    "qsa_kv_indexk_indexq": {"qsa_kv": "k_v_indexk_indexq"},
    "qsa_kv_value_key_swapped": {"qsa_kv": "v_k_indexq_indexk"},
    "hc_inject_first": {"hc_inject": "inject_first"},
    "head_swap_halves": {"output_head": "swap_halves"},
    "head_reverse_blocks": {"output_head": "reverse_blocks"},
    "control_value_heads_1": {"gdn_qkvz": "q_k_vz_interleaved", "value_head_seed": 1},
    "control_value_heads_2": {"gdn_qkvz": "q_k_vz_interleaved", "value_head_seed": 2},
    "control_value_heads_3": {"gdn_qkvz": "q_k_vz_interleaved", "value_head_seed": 3},
    "control_value_heads_4": {"gdn_qkvz": "q_k_vz_interleaved", "value_head_seed": 4},
    "best_vz_moefirst": {"gdn_qkvz": "q_k_vz_interleaved", "moe_dense": "score_first"},
}


def candidate(name: str, base: Layout | None = None) -> Layout:
    """Build a layout by applying one named candidate on top of ``base``."""
    if name not in CANDIDATES:
        raise KeyError("unknown layout candidate " + name)
    start = base if base is not None else CURRENT
    return start.with_(name=name, **CANDIDATES[name])
