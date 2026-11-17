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
from .layouts import CURRENT, Layout
from .semantic import SEM
from .state import GdnState, ModelState, PleState
from .weights import WeightStore


class RefModel:
    def __init__(self, artifact_path, config=CFG, layout: Layout | None = None):
        self.artifact = V3Artifact(artifact_path)
        self.store = WeightStore(self.artifact)
        self.cfg = config
        self.layout = layout if layout is not None else CURRENT
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
            parent = self.store.contiguous(self._object(f"text/layers/{layer}/{kind}/down"))
            rows = self.layout.hc_rows(self.cfg)
            down = parent[rows["down"]]
            inject = parent[rows["inject"]]
            norm = self._contiguous(f"text/layers/{layer}/{kind}/norm").reshape(self.cfg.hc_streams, self.cfg.hidden)
            up = self._contiguous(f"text/layers/{layer}/{kind}/up")
            self._weights[key] = {"down": np.concatenate([down, inject]), "norm": norm, "up": up}
        return self._weights[key]

    def _final_hc(self):
        if "final_hc" not in self._weights:
            self._weights["final_hc"] = {
                "down": self._contiguous("text/final_hc/down"),
                "up": self._contiguous("text/final_hc/up"),
                "norm": self._contiguous("text/final_hc/norm").reshape(self.cfg.hc_streams, self.cfg.hidden),
            }
        return self._weights["final_hc"]

    def _tail_norm(self):
        name = (
            f"text/layers/{self.cfg.layers}/attention_hc/norm"
            if not self.cfg.is_full
            else "text/final_hc/norm"
        )
        return self._contiguous(name).reshape(self.cfg.hc_streams, self.cfg.hidden)

    def _gdn_weights(self, layer):
        key = ("gdn", layer)
        if key not in self._weights:
            qkvz = self.store.w8(self._object(f"text/layers/{layer}/gdn/query"))
            control = self._contiguous(f"text/layers/{layer}/gdn/a_projection")
            rows = self.layout.gdn_rows(self.cfg)
            control_rows = self.layout.gdn_control_rows(self.cfg)
            self._weights[key] = {
                "q": qkvz[rows["query"]],
                "k": qkvz[rows["key"]],
                "v": qkvz[rows["value"]],
                "z": qkvz[rows["z"]],
                "a": control[control_rows["a"]],
                "b": control[control_rows["b"]],
                "a_log": self._contiguous(f"text/layers/{layer}/gdn/a_log"),
                "dt_bias": self._contiguous(f"text/layers/{layer}/gdn/dt_bias"),
                "conv": self.layout.conv_kernel(self._contiguous(f"text/layers/{layer}/gdn/convolution")),
                "norm": self._contiguous(f"text/layers/{layer}/gdn/norm"),
                "out": self.store.w8(self._object(f"text/layers/{layer}/gdn/output")),
            }
        return self._weights[key]

    def _qsa_weights(self, layer):
        key = ("qsa", layer)
        if key not in self._weights:
            query_gate = self.store.w8(self._object(f"text/layers/{layer}/attention/query"))
            key_value = self._contiguous(f"text/layers/{layer}/attention/key")
            query_rows, gate_rows = self.layout.qsa_query_gate_rows(self.cfg)
            kv_rows = self.layout.qsa_kv_rows(self.cfg)
            self._weights[key] = {
                "query": query_gate[query_rows],
                "gate": query_gate[gate_rows],
                "kv": key_value[
                    np.concatenate(
                        [kv_rows["key"], kv_rows["value"], kv_rows["index_query"], kv_rows["index_key"]]
                    )
                ],
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
            rows = self.layout.moe_rows(self.cfg)
            gate_object = self.artifact.parts(f"text/layers/{layer}/moe/experts/gate")[0].object
            down_object = self.artifact.parts(f"text/layers/{layer}/moe/experts/down")[0].object
            self._weights[key] = {
                "control": parent[np.concatenate([rows["router"], rows["score"]])],
                "shared_up": parent[np.concatenate([rows["gate"], rows["up"]])],
                "shared_down": self._contiguous(f"text/layers/{layer}/moe/shared/down"),
                "gate_object": gate_object,
                "down_object": down_object,
            }
        return self._weights[key]

    def _ple_weights(self):
        if "ple" not in self._weights:
            key_value = self._contiguous(f"text/layers/{self.cfg.ple_layer}/ple/key")
            self._weights["ple"] = {
                "key": key_value[0:self.cfg.residual_width],
                "value": key_value[self.cfg.residual_width : self.cfg.residual_width + self.cfg.hidden],
                "norm_key": self._contiguous(f"text/layers/{self.cfg.ple_layer}/ple/norm_key").reshape(self.cfg.hc_streams, self.cfg.hidden),
                "norm_query": self._contiguous(f"text/layers/{self.cfg.ple_layer}/ple/norm_query").reshape(self.cfg.hc_streams, self.cfg.hidden),
                "norm_conv": self._contiguous(f"text/layers/{self.cfg.ple_layer}/ple/norm_conv").reshape(self.cfg.hc_streams, self.cfg.hidden),
                "conv": self._contiguous(f"text/layers/{self.cfg.ple_layer}/ple/convolution"),
                "table": [part.object for part in self.artifact.parts(f"text/layers/{self.cfg.ple_layer}/ple/table")],
            }
        return self._weights["ple"]

    def _embedding_rows(self, ids):
        object_id = self._object("text/token_embedding")
        row_bytes = self.cfg.hidden * 2
        rows = np.empty((len(ids), self.cfg.hidden), dtype=np.float32)
        for index, token in enumerate(ids):
            raw = self.artifact.read_object(object_id, int(token) * row_bytes, row_bytes)
            rows[index] = bf16_from_u16(np.frombuffer(raw, dtype="<u2"))
        return rows

    def _read_rows(self, object_id, source_rows, row_bytes):
        hidden = self.cfg.hidden
        out = np.empty((source_rows.size, hidden), dtype=np.float32)
        index = 0
        total = int(source_rows.size)
        while index < total:
            stop = index + 1
            while stop < total and int(source_rows[stop]) == int(source_rows[stop - 1]) + 1:
                stop += 1
            begin = int(source_rows[index])
            count = stop - index
            raw = self.artifact.read_object(object_id, begin * row_bytes, count * row_bytes)
            out[index:stop] = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, hidden)
            index = stop
        return out

    def _logits(self, sample):
        object_id = self._object("text/output_head")
        vocab = int(self.artifact.object(object_id).shape[0])
        row_bytes = self.cfg.hidden * 2
        vector = np.asarray(sample, dtype=np.float64)
        permutation = self.layout.output_head_rows(self.cfg)
        logits = np.empty(vocab, dtype=np.float64)
        chunk = 8192
        for start in range(0, vocab, chunk):
            count = min(chunk, vocab - start)
            if permutation is None:
                raw = self.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
                weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, self.cfg.hidden)
            else:
                weights = self._read_rows(object_id, permutation[start : start + count], row_bytes)
            logits[start : start + count] = weights.astype(np.float64) @ vector
        return logits

    def position_logits(self, samples):
        """Logits for every position of ``samples`` (diagnostic helper)."""
        object_id = self._object("text/output_head")
        vocab = int(self.artifact.object(object_id).shape[0])
        row_bytes = self.cfg.hidden * 2
        values = np.asarray(samples, dtype=np.float64)
        tokens = values.shape[0]
        logits = np.empty((tokens, vocab), dtype=np.float64)
        permutation = self.layout.output_head_rows(self.cfg)
        chunk = 8192
        for start in range(0, vocab, chunk):
            count = min(chunk, vocab - start)
            if permutation is None:
                raw = self.artifact.read_object(object_id, start * row_bytes, count * row_bytes)
                weights = bf16_from_u16(np.frombuffer(raw, dtype="<u2")).reshape(count, self.cfg.hidden)
            else:
                weights = self._read_rows(object_id, permutation[start : start + count], row_bytes)
            logits[:, start : start + count] = values @ weights.astype(np.float64).T
        return logits

    def _stream_perm(self, value):
        perm = SEM["hc_stream_perm"]
        if perm == "identity" or value is None:
            return value
        array = np.asarray(value)
        if array.ndim < 3 or array.shape[-2] != self.cfg.hc_streams:
            return value
        if perm == "rot1":
            order = np.roll(np.arange(self.cfg.hc_streams), 1)
        elif perm == "pair_swap":
            order = np.array([1, 0, 3, 2])
        elif perm == "reverse":
            order = np.array([3, 2, 1, 0])
        else:
            return value
        return np.asarray(array)[..., order, :]

    def ple_rows(self, ids):
        multipliers, vocab_sizes, offsets = self.artifact.ple_hash_controls()
        return ple_hash_rows(
            list(ids), (self.cfg.eos_token_id, self.cfg.eos_token_id), multipliers, vocab_sizes, offsets,
            self.cfg.eos_token_id,
        )

    def _gdn(self, layer, hidden, weights, state):
        tokens = hidden.shape[0]
        q = dense(hidden, weights["q"])
        k = dense(hidden, weights["k"])
        v = dense(hidden, weights["v"])
        z = dense(hidden, weights["z"])
        g, beta = gdn_gating(hidden, weights["a"], weights["b"], weights["a_log"], weights["dt_bias"])
        conv_in = np.empty((tokens, self.cfg.gdn_conv_channels), dtype=np.float64)
        conv_in[:, 0 : self.cfg.gdn_key_width] = q
        conv_in[:, self.cfg.gdn_key_width : 2 * self.cfg.gdn_key_width] = k
        conv_in[:, 2 * self.cfg.gdn_key_width :] = v
        conv = causal_conv_silu(conv_in, weights["conv"], state.history)
        q_conv = conv[:, 0 : self.cfg.gdn_key_width].reshape(tokens, self.cfg.gdn_k_heads, self.cfg.gdn_head_dim)
        k_conv = conv[:, self.cfg.gdn_key_width : 2 * self.cfg.gdn_key_width].reshape(tokens, self.cfg.gdn_k_heads, self.cfg.gdn_head_dim)
        v_conv = conv[:, 2 * self.cfg.gdn_key_width :].reshape(tokens, self.cfg.gdn_v_heads, self.cfg.gdn_head_dim)
        z_view = z.reshape(tokens, self.cfg.gdn_v_heads, self.cfg.gdn_head_dim)
        out = gated_delta_net(q_conv, k_conv, v_conv, g, beta, state.recurrent, self.cfg.gdn_scale, self.cfg.rms_epsilon)
        normalized = gated_rmsnorm(out, weights["norm"], z_view, self.cfg.rms_epsilon)
        return dense(normalized.reshape(tokens, self.cfg.gdn_value_width), weights["out"])

    def _qsa(self, layer, hidden, weights, tensor_capture=None):
        tokens = hidden.shape[0]
        query = dense(hidden, weights["query"]).reshape(tokens, self.cfg.q_heads, self.cfg.head_dim)
        gate = dense(hidden, weights["gate"]).reshape(tokens, self.cfg.q_heads, self.cfg.head_dim)
        projected = dense(hidden, weights["kv"])
        key = projected[:, 0 : self.cfg.kv_width].reshape(tokens, self.cfg.kv_heads, self.cfg.head_dim)
        value = projected[:, self.cfg.kv_width : 2 * self.cfg.kv_width].reshape(tokens, self.cfg.kv_heads, self.cfg.head_dim)
        index_query = projected[:, 2 * self.cfg.kv_width : 2 * self.cfg.kv_width + self.cfg.index_query_width].reshape(
            tokens, self.cfg.index_heads, self.cfg.index_head_dim
        )
        index_key = projected[:, 2 * self.cfg.kv_width + self.cfg.index_query_width :].reshape(
            tokens, self.cfg.index_kv_heads, self.cfg.index_head_dim
        )
        positions = np.arange(tokens, dtype=np.int64)
        rope_positions = positions + SEM["rope_pos_offset"]
        if SEM["rope_head_pos"] == "index_axis":
            index_positions = np.stack(
                [rope_positions // (2 ** head) for head in range(self.cfg.index_heads)], axis=1
            )
        else:
            index_positions = rope_positions
        query = qsa_norm_rope(query, weights["query_norm"], rope_positions, self.cfg.rms_epsilon, self.cfg.rope_theta, self.cfg.rotary_dim)
        index_query = qsa_norm_rope(index_query, weights["index_query_norm"], index_positions, self.cfg.rms_epsilon, self.cfg.rope_theta, self.cfg.rotary_dim)
        key = qsa_norm_rope(key, weights["key_norm"], rope_positions, self.cfg.rms_epsilon, self.cfg.rope_theta, self.cfg.rotary_dim)
        value_cache = f16(value)
        blocks = tokens // self.cfg.index_compress
        index_blocks = np.zeros((blocks, self.cfg.index_head_dim), dtype=np.float32)
        for block in range(blocks):
            start = block * self.cfg.index_compress
            raw = index_key[start : start + self.cfg.index_compress, 0, :]
            index_blocks[block] = qsa_index_block(
                raw, weights["index_key_norm"], start, self.cfg.rms_epsilon, self.cfg.rope_theta, self.cfg.rotary_dim
            )
        if tensor_capture is not None:
            tensor_capture["index_query"] = index_query
            tensor_capture["index_keys"] = index_blocks
        selections = qsa_indexer(index_query, index_blocks, positions, self.cfg.index_block_budget)
        attention = qsa_sparse_gqa(
            query, gate, key, value_cache, selections, positions, self.cfg.index_compress, self.cfg.head_dim
        )
        return dense(attention.reshape(tokens, self.cfg.query_width), weights["out"]), selections

    def _moe(self, layer, hidden, weights):
        tokens = hidden.shape[0]
        ids, coefficients, shared = topk_route(hidden, weights["control"], self.cfg.experts_per_token)
        xf = np.asarray(hidden, dtype=np.float64)
        projection = xf @ np.asarray(weights["shared_up"], dtype=np.float64).T
        activation = silu(projection[:, : self.cfg.shared_intermediate]) * projection[:, self.cfg.shared_intermediate :]
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
            token_index = chosen // self.cfg.experts_per_token
            coefficient = flat_coefficients[chosen]
            gate_up = self.store.nvfp4(weights["gate_object"], expert).astype(np.float64)
            expert_down = self.store.nvfp4(weights["down_object"], expert).astype(np.float64)
            projected = xf[token_index] @ gate_up.T
            activated = silu(projected[:, : self.cfg.expert_intermediate]) * projected[:, self.cfg.expert_intermediate :]
            total[token_index] += coefficient[:, None] * (activated @ expert_down.T)
            start = end
        return bf16(total)

    def _ple(self, residual, weights, rows, state):
        tokens = residual.shape[0]
        row_tables = {}
        row_values = np.empty((tokens, self.cfg.ple_heads, self.cfg.ple_head_width), dtype=np.float32)
        for token in range(tokens):
            for head in range(self.cfg.ple_heads):
                global_row = int(rows[token, head])
                shard = global_row // self.cfg.ple_shard_rows
                shard_row = global_row % self.cfg.ple_shard_rows
                key = (shard, shard_row)
                value = row_tables.get(key)
                if value is None:
                    value = self.store.ple_row(weights["table"][shard], shard_row)
                    row_tables[key] = value
                row_values[token, head] = value
        embedding = bf16(row_values.reshape(tokens, self.cfg.hidden))
        projected_key = dense(embedding, weights["key"])
        projected_value = dense(embedding, weights["value"])
        gated, normalized_conv = ple_gate(
            residual,
            projected_key,
            projected_value,
            weights["norm_key"],
            weights["norm_query"],
            weights["norm_conv"],
            self.cfg.rms_epsilon,
        )
        updated, normalized = ple_conv_add_norm(
            state.history,
            normalized_conv,
            residual,
            gated,
            weights["conv"],
            self._hc(self.cfg.ple_layer, "attention_hc")["norm"],
            self.cfg.rms_epsilon,
        )
        return updated.reshape(tokens, self.cfg.hc_streams, self.cfg.hidden), normalized.reshape(
            tokens, self.cfg.hc_streams, self.cfg.hidden
        )

    def forward(self, ids, capture=False, capture_tensors=False):
        tokens = list(ids)
        state = ModelState.fresh()
        embedding = self._embedding_rows(tokens)
        residual, normalized = hc_expand_norm(
            embedding, self._hc(0, "attention_hc")["norm"], self.cfg.rms_epsilon
        )
        residual = self._stream_perm(residual)
        normalized = self._stream_perm(normalized)
        captured = {"ple_rows": self.ple_rows(tokens), "qsa": {}}
        tensors = {} if capture_tensors else None
        for layer in range(self.cfg.layers):
            if tensors is not None:
                tensors[(layer, "residual_in")] = residual
                tensors[(layer, "normalized_in")] = normalized
            attention_hc = self._hc(layer, "attention_hc")
            block_input, injection = hc_mix(normalized, attention_hc["down"], attention_hc["up"])
            if tensors is not None:
                tensors[(layer, "attention_block")] = block_input
                tensors[(layer, "attention_injection")] = injection
            if self.cfg.is_qsa_layer(layer):
                layer_tensors = {} if tensors is not None else None
                mixer, selections = self._qsa(
                    layer, block_input, self._qsa_weights(layer), layer_tensors
                )
                if tensors is not None:
                    tensors[(layer, "index_query")] = layer_tensors["index_query"]
                    tensors[(layer, "index_keys")] = layer_tensors["index_keys"]
                if capture:
                    captured["qsa"][layer] = selections
            else:
                mixer = self._gdn(layer, block_input, self._gdn_weights(layer), state.gdn[layer])
            if tensors is not None:
                tensors[(layer, "mixer")] = mixer
            moe_hc = self._hc(layer, "moe_hc")
            stage, normalized = hc_combine_norm(
                residual, mixer, injection, moe_hc["norm"], self.cfg.rms_epsilon
            )
            normalized = self._stream_perm(normalized)
            if tensors is not None:
                tensors[(layer, "stage")] = stage
                tensors[(layer, "normalized_moe")] = normalized
            block_input, injection = hc_mix(normalized, moe_hc["down"], moe_hc["up"])
            if tensors is not None:
                tensors[(layer, "moe_block")] = block_input
                tensors[(layer, "moe_injection")] = injection
            moe_out = self._moe(layer, block_input, self._moe_weights(layer))
            if SEM["ablate"] == "moe":
                moe_out = np.zeros_like(np.asarray(moe_out, dtype=np.float64))
            if tensors is not None:
                tensors[(layer, "moe")] = moe_out
            if layer + 1 == self.cfg.ple_layer:
                residual = hc_combine(stage, moe_out, injection)
                residual, normalized = self._ple(residual, self._ple_weights(), captured["ple_rows"], state.ple)
                residual = self._stream_perm(residual)
                normalized = self._stream_perm(normalized)
            else:
                next_norm = (
                    self._hc(layer + 1, "attention_hc")["norm"]
                    if layer + 1 < self.cfg.layers
                    else self._tail_norm()
                )
                residual, normalized = hc_combine_norm(
                    stage, moe_out, injection, next_norm, self.cfg.rms_epsilon
                )
                residual = self._stream_perm(residual)
                normalized = self._stream_perm(normalized)
            if tensors is not None:
                tensors[(layer, "residual_out")] = residual
                tensors[(layer, "normalized_out")] = normalized
        final_hc = self._final_hc()
        sample = hc_final_mix(normalized[-1:], final_hc["down"], final_hc["up"])[0]
        logits = self._logits(sample)
        captured["sample"] = sample
        captured["logits"] = logits
        if tensors is not None:
            captured["tensors"] = tensors
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
