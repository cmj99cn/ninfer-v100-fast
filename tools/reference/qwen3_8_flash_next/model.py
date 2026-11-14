from __future__ import annotations

import numpy as np

from .bindings import V3Artifact
from .config import CFG
from .ops import (
    bf16,
    bf16_from_u16,
    causal_conv_silu,
    dense,
    f16,
    gated_delta_net,
    gated_rmsnorm,
    gdn_gating,
    hc_combine,
    hc_combine_norm,
    hc_expand_norm,
    hc_final_mix,
    hc_mix,
    ple_conv_add_norm,
    ple_gate,
    ple_hash_rows,
    qsa_index_block,
    qsa_indexer,
    qsa_norm_rope,
    qsa_sparse_gqa,
    silu,
    topk_route,
)
from .state import GdnState, ModelState, PleState
from .weights import WeightStore


class RefModel:
    def __init__(self, artifact_path):
        self.artifact = V3Artifact(artifact_path)
        self.store = WeightStore(self.artifact)
        self._ids = {}
        self._weights = {}

    def close(self):
        self.artifact.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _object(self, name):
        object_id = self._ids.get(name)
        if object_id is None:
            object_id = self.artifact.single_object(name)
            self._ids[name] = object_id
        return object_id

    def _contiguous(self, name):
        return self.store.contiguous(self._object(name))

    def _hc(self, layer, kind):
        key = (layer, kind)
        if key not in self._weights:
            down = self.store.contiguous_rows(self._object(f"text/layers/{layer}/{kind}/down"), 0, CFG.hc_lowrank + CFG.hc_streams)
            norm = self._contiguous(f"text/layers/{layer}/{kind}/norm").reshape(CFG.hc_streams, CFG.hidden)
            up = self._contiguous(f"text/layers/{layer}/{kind}/up")
            self._weights[key] = {"down": down, "norm": norm, "up": up}
        return self._weights[key]

    def _final_hc(self):
        if "final_hc" not in self._weights:
            self._weights["final_hc"] = {
                "down": self._contiguous("text/final_hc/down"),
                "up": self._contiguous("text/final_hc/up"),
                "norm": self._contiguous("text/final_hc/norm").reshape(CFG.hc_streams, CFG.hidden),
            }
        return self._weights["final_hc"]

    def _tail_norm(self):
        return self._contiguous(f"text/layers/{CFG.layers}/attention_hc/norm").reshape(
            CFG.hc_streams, CFG.hidden
        )

    def _gdn_weights(self, layer):
        key = ("gdn", layer)
        if key not in self._weights:
            qkvz = self.store.w8(self._object(f"text/layers/{layer}/gdn/query"))
            control = self._contiguous(f"text/layers/{layer}/gdn/a_projection")
            self._weights[key] = {
                "q": qkvz[0 : CFG.gdn_key_width],
                "k": qkvz[CFG.gdn_key_width : 2 * CFG.gdn_key_width],
                "v": qkvz[2 * CFG.gdn_key_width : 2 * CFG.gdn_key_width + CFG.gdn_value_width],
                "z": qkvz[CFG.gdn_conv_channels : CFG.gdn_conv_channels + CFG.gdn_value_width],
                "a": control[0 : CFG.gdn_v_heads],
                "b": control[CFG.gdn_v_heads : 2 * CFG.gdn_v_heads],
                "a_log": self._contiguous(f"text/layers/{layer}/gdn/a_log"),
                "dt_bias": self._contiguous(f"text/layers/{layer}/gdn/dt_bias"),
                "conv": self._contiguous(f"text/layers/{layer}/gdn/convolution"),
                "norm": self._contiguous(f"text/layers/{layer}/gdn/norm"),
                "out": self.store.w8(self._object(f"text/layers/{layer}/gdn/output")),
            }
        return self._weights[key]

    def _qsa_weights(self, layer):
        key = ("qsa", layer)
        if key not in self._weights:
            query_gate = self.store.w8(self._object(f"text/layers/{layer}/attention/query"))
            key_value = self._contiguous(f"text/layers/{layer}/attention/key")
            self._weights[key] = {
                "query": query_gate[0 : CFG.query_width],
                "gate": query_gate[CFG.query_width : 2 * CFG.query_width],
                "kv": key_value[0 : 2 * CFG.kv_width + CFG.index_query_width + CFG.index_key_width],
                "query_norm": self._contiguous(f"text/layers/{layer}/attention/query_norm"),
                "key_norm": self._contiguous(f"text/layers/{layer}/attention/key_norm"),
                "index_query_norm": self._contiguous(f"text/layers/{layer}/attention/index_query_norm"),
                "index_key_norm": self._contiguous(f"text/layers/{layer}/attention/index_key_norm"),
                "out": self.store.w8(self._object(f"text/layers/{layer}/attention/output")),
            }
        return self._weights[key]

    def _moe_weights(self, layer):
        key = ("moe", layer)
        if key not in self._weights:
            parent = self._contiguous(f"text/layers/{layer}/moe/router")
            gate_object = self.artifact.parts(f"text/layers/{layer}/moe/experts/gate")[0].object
            down_object = self.artifact.parts(f"text/layers/{layer}/moe/experts/down")[0].object
            self._weights[key] = {
                "control": parent[0 : CFG.experts + 1],
                "shared_up": parent[CFG.experts + 1 : CFG.experts + 1 + 2 * CFG.shared_intermediate],
                "shared_down": self._contiguous(f"text/layers/{layer}/moe/shared/down"),
                "gate_object": gate_object,
                "down_object": down_object,
            }
        return self._weights[key]

    def _ple_weights(self):
        if "ple" not in self._weights:
            key_value = self._contiguous(f"text/layers/{CFG.ple_layer}/ple/key")
            self._weights["ple"] = {
                "key": key_value[0:CFG.residual_width],
                "value": key_value[CFG.residual_width : CFG.residual_width + CFG.hidden],
                "norm_key": self._contiguous(f"text/layers/{CFG.ple_layer}/ple/norm_key").reshape(CFG.hc_streams, CFG.hidden),
                "norm_query": self._contiguous(f"text/layers/{CFG.ple_layer}/ple/norm_query").reshape(CFG.hc_streams, CFG.hidden),
                "norm_conv": self._contiguous(f"text/layers/{CFG.ple_layer}/ple/norm_conv").reshape(CFG.hc_streams, CFG.hidden),
                "conv": self._contiguous(f"text/layers/{CFG.ple_layer}/ple/convolution"),
                "table": [part.object for part in self.artifact.parts(f"text/layers/{CFG.ple_layer}/ple/table")],
            }
        return self._weights["ple"]

    def _embedding_rows(self, ids):
        object_id = self._object("text/token_embedding")
        row_bytes = CFG.hidden * 2
        rows = np.empty((len(ids), CFG.hidden), dtype=np.float32)
        for index, token in enumerate(ids):
            raw = self.artifact.read_object(object_id, int(token) * row_bytes, row_bytes)
            rows[index] = bf16_from_u16(np.frombuffer(raw, dtype="<u2"))
        return rows

    def _logits(self, sample):
        object_id = self._object("text/output_head")
        vocab = int(self.artifact.object(object_id).shape[0])
        row_bytes = CFG.hidden * 2
        vector = np.asarray(sample, dtype=np.float64)
        logits = np.empty(vocab, dtype=np.float64)
        chunk = 8192
        for start in range(0, vocab, chunk):
            count = min(chunk, vocab - start)
            raw = self.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
            weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, CFG.hidden)
            logits[start : start + count] = weights.astype(np.float64) @ vector
        return logits

    def ple_rows(self, ids):
        multipliers, vocab_sizes, offsets = self.artifact.ple_hash_controls()
        return ple_hash_rows(
            list(ids), (CFG.eos_token_id, CFG.eos_token_id), multipliers, vocab_sizes, offsets,
            CFG.eos_token_id,
        )

    def _gdn(self, layer, hidden, weights, state):
        tokens = hidden.shape[0]
        q = dense(hidden, weights["q"])
        k = dense(hidden, weights["k"])
        v = dense(hidden, weights["v"])
        z = dense(hidden, weights["z"])
        g, beta = gdn_gating(hidden, weights["a"], weights["b"], weights["a_log"], weights["dt_bias"])
        conv_in = np.empty((tokens, CFG.gdn_conv_channels), dtype=np.float64)
        conv_in[:, 0 : CFG.gdn_key_width] = q
        conv_in[:, CFG.gdn_key_width : 2 * CFG.gdn_key_width] = k
        conv_in[:, 2 * CFG.gdn_key_width :] = v
        conv = causal_conv_silu(conv_in, weights["conv"], state.history)
        q_conv = conv[:, 0 : CFG.gdn_key_width].reshape(tokens, CFG.gdn_k_heads, CFG.gdn_head_dim)
        k_conv = conv[:, CFG.gdn_key_width : 2 * CFG.gdn_key_width].reshape(tokens, CFG.gdn_k_heads, CFG.gdn_head_dim)
        v_conv = conv[:, 2 * CFG.gdn_key_width :].reshape(tokens, CFG.gdn_v_heads, CFG.gdn_head_dim)
        z_view = z.reshape(tokens, CFG.gdn_v_heads, CFG.gdn_head_dim)
        out = gated_delta_net(q_conv, k_conv, v_conv, g, beta, state.recurrent, CFG.gdn_scale, CFG.rms_epsilon)
        normalized = gated_rmsnorm(out, weights["norm"], z_view, CFG.rms_epsilon)
        return dense(normalized.reshape(tokens, CFG.gdn_value_width), weights["out"])

    def _qsa(self, layer, hidden, weights):
        tokens = hidden.shape[0]
        query = dense(hidden, weights["query"]).reshape(tokens, CFG.q_heads, CFG.head_dim)
        gate = dense(hidden, weights["gate"]).reshape(tokens, CFG.q_heads, CFG.head_dim)
        projected = dense(hidden, weights["kv"])
        key = projected[:, 0 : CFG.kv_width].reshape(tokens, CFG.kv_heads, CFG.head_dim)
        value = projected[:, CFG.kv_width : 2 * CFG.kv_width].reshape(tokens, CFG.kv_heads, CFG.head_dim)
        index_query = projected[:, 2 * CFG.kv_width : 2 * CFG.kv_width + CFG.index_query_width].reshape(
            tokens, CFG.index_heads, CFG.index_head_dim
        )
        index_key = projected[:, 2 * CFG.kv_width + CFG.index_query_width :].reshape(
            tokens, CFG.index_kv_heads, CFG.index_head_dim
        )
        positions = np.arange(tokens, dtype=np.int64)
        query = qsa_norm_rope(query, weights["query_norm"], positions, CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim)
        index_query = qsa_norm_rope(index_query, weights["index_query_norm"], positions, CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim)
        key = qsa_norm_rope(key, weights["key_norm"], positions, CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim)
        value_cache = f16(value)
        blocks = tokens // CFG.index_compress
        index_blocks = np.zeros((blocks, CFG.index_head_dim), dtype=np.float32)
        for block in range(blocks):
            start = block * CFG.index_compress
            raw = index_key[start : start + CFG.index_compress, 0, :]
            index_blocks[block] = qsa_index_block(
                raw, weights["index_key_norm"], start, CFG.rms_epsilon, CFG.rope_theta, CFG.rotary_dim
            )
        selections = qsa_indexer(index_query, index_blocks, positions, CFG.index_block_budget)
        attention = qsa_sparse_gqa(
            query, gate, key, value_cache, selections, positions, CFG.index_compress, CFG.head_dim
        )
        return dense(attention.reshape(tokens, CFG.query_width), weights["out"]), selections

    def _moe(self, layer, hidden, weights):
        tokens = hidden.shape[0]
        ids, coefficients, shared = topk_route(hidden, weights["control"], CFG.experts_per_token)
        xf = np.asarray(hidden, dtype=np.float64)
        projection = xf @ np.asarray(weights["shared_up"], dtype=np.float64).T
        activation = silu(projection[:, : CFG.shared_intermediate]) * projection[:, CFG.shared_intermediate :]
        down = activation @ np.asarray(weights["shared_down"], dtype=np.float64).T
        total = np.asarray(shared, dtype=np.float64)[:, None] * down
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
            gate_up = self.store.nvfp4(weights["gate_object"], expert).astype(np.float64)
            expert_down = self.store.nvfp4(weights["down_object"], expert).astype(np.float64)
            projected = xf[token_index] @ gate_up.T
            activated = silu(projected[:, : CFG.expert_intermediate]) * projected[:, CFG.expert_intermediate :]
            total[token_index] += coefficient[:, None] * (activated @ expert_down.T)
            start = end
        return bf16(total)

    def _ple(self, residual, weights, rows, state):
        tokens = residual.shape[0]
        row_tables = {}
        row_values = np.empty((tokens, CFG.ple_heads, CFG.ple_head_width), dtype=np.float32)
        for token in range(tokens):
            for head in range(CFG.ple_heads):
                global_row = int(rows[token, head])
                shard = global_row // CFG.ple_shard_rows
                shard_row = global_row % CFG.ple_shard_rows
                key = (shard, shard_row)
                value = row_tables.get(key)
                if value is None:
                    value = self.store.ple_row(weights["table"][shard], shard_row)
                    row_tables[key] = value
                row_values[token, head] = value
        embedding = bf16(row_values.reshape(tokens, CFG.hidden))
        projected_key = dense(embedding, weights["key"])
        projected_value = dense(embedding, weights["value"])
        gated, normalized_conv = ple_gate(
            residual,
            projected_key,
            projected_value,
            weights["norm_key"],
            weights["norm_query"],
            weights["norm_conv"],
            CFG.rms_epsilon,
        )
        updated, normalized = ple_conv_add_norm(
            state.history,
            normalized_conv,
            residual,
            gated,
            weights["conv"],
            self._hc(CFG.ple_layer, "attention_hc")["norm"],
            CFG.rms_epsilon,
        )
        return updated.reshape(tokens, CFG.hc_streams, CFG.hidden), normalized.reshape(
            tokens, CFG.hc_streams, CFG.hidden
        )

    def forward(self, ids, capture=False):
        tokens = list(ids)
        state = ModelState.fresh()
        embedding = self._embedding_rows(tokens)
        residual, normalized = hc_expand_norm(
            embedding, self._hc(0, "attention_hc")["norm"], CFG.rms_epsilon
        )
        captured = {"ple_rows": self.ple_rows(tokens), "qsa": {}}
        for layer in range(CFG.layers):
            attention_hc = self._hc(layer, "attention_hc")
            block_input, injection = hc_mix(normalized, attention_hc["down"], attention_hc["up"])
            if CFG.is_qsa_layer(layer):
                mixer, selections = self._qsa(layer, block_input, self._qsa_weights(layer))
                if capture:
                    captured["qsa"][layer] = selections
            else:
                mixer = self._gdn(layer, block_input, self._gdn_weights(layer), state.gdn[layer])
            moe_hc = self._hc(layer, "moe_hc")
            stage, normalized = hc_combine_norm(
                residual, mixer, injection, moe_hc["norm"], CFG.rms_epsilon
            )
            block_input, injection = hc_mix(normalized, moe_hc["down"], moe_hc["up"])
            moe_out = self._moe(layer, block_input, self._moe_weights(layer))
            if layer + 1 == CFG.ple_layer:
                residual = hc_combine(stage, moe_out, injection)
                residual, normalized = self._ple(residual, self._ple_weights(), captured["ple_rows"], state.ple)
            else:
                next_norm = (
                    self._hc(layer + 1, "attention_hc")["norm"]
                    if layer + 1 < CFG.layers
                    else self._tail_norm()
                )
                residual, normalized = hc_combine_norm(
                    stage, moe_out, injection, next_norm, CFG.rms_epsilon
                )
        final_hc = self._final_hc()
        sample = hc_final_mix(normalized[-1:], final_hc["down"], final_hc["up"])[0]
        logits = self._logits(sample)
        captured["sample"] = sample
        captured["logits"] = logits
        return captured

    def next_token(self, ids):
        result = self.forward(ids)
        logits = result["logits"]
        order = np.argsort(-logits, kind="stable")[:2]
        token = int(order[0])
        margin = float(logits[order[0]] - logits[order[1]])
        return token, margin, logits

    def generate(self, ids, steps):
        sequence = list(int(token) for token in ids)
        produced = []
        logits = None
        for _ in range(steps):
            result = self.forward(sequence)
            logits = result["logits"]
            token = int(np.argmax(logits))
            produced.append(token)
            sequence.append(token)
        return produced, logits
