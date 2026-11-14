# Qwen3.8-Flash-Next artifact reference

This reference is the semantic and geometric authority for the `qwen3.8-flash-next` target
package. It records what the registered `.ninfer` artifact of the Qwen3.8-Flash-Next /
Qwen4Exp checkpoint family contains, how its parameter names map onto weight roles, and the
mathematics that the package must reproduce. Generic container framing is defined in
[`artifact-container.md`](artifact-container.md), numeric semantics in
[`tensor-formats.md`](tensor-formats.md) and byte packing in
[`storage-layouts.md`](storage-layouts.md).

The artifact is the upstream v3 container `qwen3_8_flash_next.ninfer`
(108,445,200,388 B, split into the entry file and `part-0001..part-0003`). This reference
describes that artifact only; no conversion pipeline is owned by this repository and no artifact
byte is rewritten here.

The upstream semantics are the authority for every formula below. The cited paths are the
upstream NInfer repository paths of the Qwen4Exp implementation, with the line range that
carries the formula; the same files were fetched into the working session for transcription.

## 1. Identity, inventory and format census

### 1.1 Identity

```text
model_id   = qwen3.8-flash-next
weights_id = nvfp4
```

The reader synthesizes this identity from the v3 directory metadata name; no filename or object
name selects it (`src/artifact/reader.cpp`, `kContainerV3Identities`). The package's own
registry key is `qwen3_8_flash_next`.

### 1.2 Object and format census

The directory lists 1319 objects: 1312 tensors and 7 raw-byte resources. The format census over
the tensor objects is:

| Format | Layout | Objects |
|---|---|---:|
| `bf16` | `contiguous_le_v1` | 916 |
| `fp32` | `contiguous_le_v1` | 73 |
| `nf4_g16_e4m3fn` | `row_paged_g16_4k_v1` | 128 |
| `q8_g32_fp16` | `row_split_k128_v1` | 96 |
| `nvfp4` | `batched_block_scale_k16_m128x4_v1` | 96 |
| `fp8_e4m3fn_block128x128_bf16` | `block_scale_m128_k128_v1` | 2 |
| `q4_g64_fp16` | `row_split_k128_v1` | 1 |
| resource | — | 7 |

The conversion report `qwen3_8_flash_next.ninfer.conversion.json` prints `fp32: 72` under its
`formats` census. That census covers the 1311 converted tensors; the artifact directory holds 73
FP32 objects. The difference is one object, `auxiliary/shared/expert_activation_divisors`
(FP32 `[48,2]`), which the converter emits as an auxiliary and not as a converted tensor; the
remaining 72 are the 36 GDN layers' `a_log` and `dt_bias` vectors. **The directory is the
authority**; the conversion report is provenance.

The seven resources are `resource/text/tokenizer.json`, `resource/text/tokenizer_config.json`,
`resource/text/chat_template.jinja`, `resource/text/generation_config.json`,
`resource/text/ple_hash_controls`, `resource/vision/preprocessor_config.json` and
`resource/vision/video_preprocessor_config.json`.

### 1.3 Where these measurements come from

Every measured number in this reference is read out of the artifact's v3 directory JSON:

- `components.text.config` gives the model configuration (section 2);
- `bindings` and `part_bindings` give the parameter names, their objects and their element
  ranges (sections 3 and 4);
- `objects` gives each object's format, layout, shape and byte count (sections 1.2 and 4);
- `uses` gives the operator use edges and their activation policies (section 9.4);
- the `ple_hash_controls` resource carries the PLE hash constants (section 8.1).

The `Reader` exposes the directory as `Reader::v3_directory()`
(`src/artifact/reader.h`), and `tensor_element_interval_view` maps an element range onto the
logical view of its layout (`src/artifact/storage_layouts.cpp`). All 76,242 part ranges of this
artifact are rectangles in their own layout. The one conclusion of this reference that the
directory does not state is the PLE shard-to-global-row assignment; it is marked as a
derivation in section 8.1.

## 2. The text configuration

### 2.1 The complete `components.text.config`

| Key | Value |
|---|---|
| `hidden_size` | 2560 |
| `vocab_size` | 248320 |
| `num_hidden_layers` | 48 |
| `max_position_embeddings` | 262144 |
| `hc_count` / `hc_lowrank` | 4 / 320 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 |
| `linear_conv_kernel_dim` | 4 |
| `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` | 4 / 1 / 128 |
| `indexer_compress_ratio` / `indexer_budget` | 4 / 2048 |
| `num_experts` / `num_experts_per_tok` | 512 / 10 |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 |
| `ple_embed_dim` / `ngram_size` / `heads_per_ngram` / `split_ngram_parts` | 2560 / 3 / 8 / 128 |
| `ple_conv_kernel_size` / `ple_conv_dilation` | 4 / 3 |
| `ple_layer_ids` | `[1]` |
| `ple_shard_shape` | `[2500012, 160]` |
| `architectures` / `model_type` | `Qwen4ExpForCausalLM` / `qwen4_exp_text` |
| `hidden_act` / `output_gate_type` | `silu` / `sigmoid` |
| `tie_word_embeddings` / `attention_bias` | false / false |
| `mamba_ssm_dtype` / `norm_topk_prob` | `float32` / true |
| `rms_norm_eps` | 1e-6 |
| `layer_types` | 36 `linear_attention` + 12 `indexed_attention` |
| `rope_parameters.rope_theta` | 1e7 |
| `rope_parameters.partial_rotary_factor` | 0.25 |
| `rope_parameters.mrope_section` | `[11, 11, 10]` |
| `rope_parameters.mrope_interleaved` | true |
| `rope_parameters.rope_type` | `default` |
| `eos_token_id` | 248044 |
| `image_token_id` / `video_token_id` | 248056 / 248057 |
| `vision_start_token_id` / `vision_end_token_id` | 248053 / 248054 |

The layer kinds follow `layer_types`: layer `l` is an indexed-attention (QSA) layer exactly when
`l % 4 == 3`, so the QSA layers are 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 and the other
36 layers are linear-attention (GDN) layers. The PLE layer is layer 1.

### 2.2 Derived dimensions

```text
residual_width      = hidden * hc_count                    = 2560 * 4  = 10240
query_width         = query_heads * head_dim                = 24 * 256  = 6144
kv_width            = kv_heads * head_dim                   = 2 * 256   = 512
index_query_width   = indexer_n_heads * indexer_head_dim    = 4 * 128   = 512
index_key_width     = indexer_kv_heads * indexer_head_dim   = 1 * 128   = 128
gdn_key_width       = linear_num_key_heads * key_head_dim   = 16 * 128  = 2048
gdn_value_width     = linear_num_value_heads * value_dim    = 48 * 128  = 6144
gdn_conv_channels   = 2 * gdn_key_width + gdn_value_width   = 4096+6144 = 10240
hc_lowrank          = 320
rotary_dim          = head_dim * partial_rotary_factor      = 256 * 0.25 = 64
indexer_block_budget = indexer_budget / indexer_compress_ratio = 2048 / 4 = 512
ple_heads            = (ngram_size - 1) * heads_per_ngram = 2 * 8     = 16
ple_head_width      = ple_embed_dim / ple_heads           = 2560 / 16 = 160
```

Two invariants follow from the configuration and must hold for any consumer: the MRoPE sections
must sum to half the rotary width (`11 + 11 + 10 = 32 = rotary_dim / 2`), and the PLE head
geometry must satisfy `(ngram_size - 1) * heads_per_ngram = ple_embed_dim / ple_shard_shape[1]`
(`2 * 8 = 2560 / 160 = 16`).

**`split_ngram_parts = 128` is not stated by the artifact to be the shard count.** The
configuration value and the measured shard count of section 4.5 are both 128, and no consumer
may depend on more than that: the directory does not declare the relation. This package binds
the table's shards from the artifact's own binding (128 parts), not from the configuration
value, and the configuration value is checked against the measured shard geometry instead.

## 3. Parameter names and weight roles

### 3.1 The name table

The directory's `bindings` and `part_bindings` hold 1786 parameter bindings: `text` 1308,
`vision` 441, `mtp` 36, `proposal` 1. The 1308 `text/*` names are the complete parameter space
of this milestone. Their patterns, with the number of layers each pattern appears in, are:

| Pattern | Count | Role |
|---|---:|---|
| `text/layers/N/attention_hc/{down,inject,norm,up}` | each 48 | per-layer attention HC (down 320x10240, inject 4x10240, norm 10240, up 10240x320) |
| `text/layers/N/moe_hc/{down,inject,norm,up}` | each 48 | per-layer MoE HC |
| `text/layers/N/gdn/{query,key,value,z,a_projection,b_projection,a_log,dt_bias,convolution,norm,output}` | each 36 | GDN layer |
| `text/layers/N/attention/{query,key,value,index_key,index_query,output,output_gate,query_norm,index_query_norm,key_norm,index_key_norm}` | each 12 | QSA layer |
| `text/layers/N/moe/{router,shared_score,shared/gate,shared/up,shared/down,experts/gate,experts/up,experts/down}` | each 48 | MoE layer |
| `text/layers/N/ple/{table,key,value,norm_key,norm_query,norm_conv,convolution}` | each 1 | PLE, layer 1 only |
| `text/{token_embedding,output_head,final_hc/norm,final_hc/down,final_hc/up}` | each 1 | embedding, main head, final HC |

The counts add up to `48*4 + 48*4 + 36*11 + 12*11 + 48*8 + 7 + 5 = 1308`. `mtp/*` (36) and
`proposal/head` (1) belong to the MTP milestone; `vision/*` (441) belongs to the Vision
milestone. The role table of this milestone covers `text/*` only; section 3.3 names the
remaining parameters.

### 3.2 The role mapping

`src/targets/qwen3_8_flash_next/impl/load/roles.h` owns the role enumeration and the
name-to-role mapping. There is exactly one role per `text/*` name pattern, and the role table is
the exact inverse of the artifact's names: `roles::role_of` maps every `text/*` binding to
exactly one role, and `roles::parameter_name` spells the artifact's name for every role and
layer. The test `tests/targets/qwen3_8_flash_next/test_roles.cpp` asserts that bijection
against the real artifact, and pins the layer set of every role family: the GDN roles occupy
the 36 layers with `l % 4 != 3`, the QSA roles the 12 layers with `l % 4 == 3`, the HC and MoE
roles all 48 layers, and the PLE roles layer 1 alone.

The role enumeration is:

```text
TokenEmbedding, OutputHead,
FinalHcNorm, FinalHcDown, FinalHcUp,
AttentionHcNorm, AttentionHcDown, AttentionHcInject, AttentionHcUp,
MoeHcNorm, MoeHcDown, MoeHcInject, MoeHcUp,
GdnQuery, GdnKey, GdnValue, GdnZ,
GdnAProjection, GdnBProjection, GdnALog, GdnDtBias,
GdnConvolution, GdnNorm, GdnOutput,
QsaQuery, QsaKey, QsaValue, QsaIndexQuery, QsaIndexKey,
QsaOutput, QsaOutputGate, QsaQueryNorm, QsaIndexQueryNorm,
QsaKeyNorm, QsaIndexKeyNorm,
MoeRouter, MoeSharedScore, MoeSharedGate, MoeSharedUp, MoeSharedDown,
MoeRoutedGate, MoeRoutedUp, MoeRoutedDown,
PleTable, PleKey, PleValue, PleNormKey, PleNormQuery, PleNormConv, PleConvolution
```

Three roles are global (`TokenEmbedding`, `OutputHead`, `FinalHc*`); the other 47 are
layer-scoped and embed the layer index in their name. The PLE roles exist in one layer only
(layer 1); the role table itself is layer-index agnostic and the caller owns that knowledge.

The role table maps names only. Which object a role is bound to, how the object is sliced
across tensor-parallel ranks and what the runtime tensor looks like are binding decisions and
belong to the package's load plan.

### 3.3 Names outside this milestone

The remaining 478 bindings of the same artifact are deliberately not covered by the role table:

| Namespace | Count | Contents | Milestone |
|---|---:|---|---|
| `mtp/*` | 36 | one MTP layer: `token_embedding`, `output_head`, `fc_embedding`, `fc_hidden`, `pre_fc_embedding_norm`, `pre_fc_hidden_norm`, `final_hc/{norm,down,up}`, `attention_hc/{norm,down,inject,up}`, `moe_hc/{norm,down,inject,up}`, a QSA `attention/*` layer and a `moe/*` layer | M3 |
| `proposal/head` | 1 | the full-vocabulary Q4 draft head, `[248320, 2560]` `q4_g64_fp16` | M3 |
| `vision/*` | 441 | 27 vision blocks (attention, MLP, norms) plus patch embedding, position embedding and merger | M6 |

The MTP QSA layer is the reason the artifact also carries a `[13952, 2560]` BF16 parent. Its
six projections are six row ranges of that parent, in the upstream binding order
`{key, value, index_key, query, gate, index_query}` (upstream
`src/models/qwen4_exp/execution/parameters.cpp:68`):

| Parameter | Rows | Width | Width source |
|---|---|---:|---|
| `key` | `[0,512)` | 512 | `kv_width = kv_heads * head_dim = 2 * 256` |
| `value` | `[512,1024)` | 512 | `kv_width` |
| `index_key` | `[1024,1152)` | 128 | `index_key_width = indexer_kv_heads * indexer_head_dim = 1 * 128` |
| `query` | `[1152,7296)` | 6144 | `query_width = query_heads * head_dim = 24 * 256` |
| `output_gate` | `[7296,13440)` | 6144 | `query_width` |
| `index_query` | `[13440,13952)` | 512 | `index_query_width = indexer_n_heads * indexer_head_dim = 4 * 128` |

The widths sum to `512 + 512 + 128 + 6144 + 6144 + 512 = 13952`, the height of the parent. The
text QSA layer does **not** use that parent; it has its own two parents (section 4.3).

## 4. Geometry of the text parameters

The tables below give, for each parameter, the object that stores it, that object's format,
layout and shape, and the element range the parameter occupies inside the object. Ranges are
half-open `[begin, end)` element intervals in the object's own element order; "rows" gives the
same interval as a row range of a rank-two object.

**Every layer owns its own objects.** The Object column names the object of the **first layer
of the pattern** -- layer 0 for the HC, GDN and MoE families, layer 3 for the QSA family, layer
1 for PLE -- and the object ids grow with the layer: `attention_hc/norm` is `weight/000530` in
layer 0 and `weight/000540` in layer 1; `gdn/query` is `weight/000229` in layer 0 and
`weight/000234` in layer 1; `moe/router` is `weight/000228` in layer 0 and `weight/000233` in
layer 1; the routed expert bank of layer 0 is `weight/000000` / `weight/000001` and that of
layer 1 is `weight/000002` / `weight/000003`; the QSA layer 3 uses `weight/000245` /
`weight/000246` and the QSA layer 7 uses `weight/000265` / `weight/000266`. The `text/*`
bindings reference 954 distinct objects, distributed over the 48 layers. What a pattern shares
across the layers is its **format, layout, shape and element range**; only the object id
changes.

### 4.1 The four-stream HC

| Parameter | Object | Format | Layout | Shape | Range | Rows |
|---|---|---|---|---|---|---|
| `attention_hc/norm` | `weight/000530` | bf16 | contiguous_le_v1 | `[10240]` | whole | — |
| `attention_hc/down` | `weight/000226` | bf16 | contiguous_le_v1 | `[324, 10240]` | `[0, 3276800)` | `[0,320)` |
| `attention_hc/inject` | `weight/000226` | bf16 | contiguous_le_v1 | `[324, 10240]` | `[3276800, 3317760)` | `[320,324)` |
| `attention_hc/up` | `weight/000531` | bf16 | contiguous_le_v1 | `[10240, 320]` | whole | — |
| `moe_hc/norm` | `weight/000532` | bf16 | contiguous_le_v1 | `[10240]` | whole | — |
| `moe_hc/down` | `weight/000227` | bf16 | contiguous_le_v1 | `[324, 10240]` | `[0, 3276800)` | `[0,320)` |
| `moe_hc/inject` | `weight/000227` | bf16 | contiguous_le_v1 | `[324, 10240]` | `[3276800, 3317760)` | `[320,324)` |
| `moe_hc/up` | `weight/000533` | bf16 | contiguous_le_v1 | `[10240, 320]` | whole | — |

The HC "down" and "inject" parameters are two row ranges of one `[324, 10240]` parent: the
down projection is rows `[0,320)` and the injection projection is the four rows `[320,324)`.
The norm weight is the flat `[10240]` stream-major vector and is viewed as `[2560, 4]`, element
`s*2560 + d`. The up projection is `[10240, 320]` with row `s*2560 + d`, matching the channel
index `c = d + 2560*s` of the residual.

`final_hc` has the same shape family without the injection rows: `final_hc/norm` is
`weight/000527` `[10240]`, `final_hc/down` is `weight/000528` `[320, 10240]` and `final_hc/up` is
`weight/000529` `[10240, 320]`, all BF16 with layout `contiguous_le_v1`.

### 4.2 GDN

| Parameter | Object | Format | Layout | Shape | Range | Rows |
|---|---|---|---|---|---|---|
| `gdn/query` | `weight/000229` | q8_g32_fp16 | row_split_k128_v1 | `[16384, 2560]` | `[0, 5242880)` | `[0,2048)` |
| `gdn/key` | `weight/000229` | q8_g32_fp16 | row_split_k128_v1 | `[16384, 2560]` | `[5242880, 10485760)` | `[2048,4096)` |
| `gdn/value` | `weight/000229` | q8_g32_fp16 | row_split_k128_v1 | `[16384, 2560]` | `[10485760, 26214400)` | `[4096,10240)` |
| `gdn/z` | `weight/000229` | q8_g32_fp16 | row_split_k128_v1 | `[16384, 2560]` | `[26214400, 41943040)` | `[10240,16384)` |
| `gdn/a_projection` | `weight/000230` | bf16 | contiguous_le_v1 | `[96, 2560]` | `[0, 122880)` | `[0,48)` |
| `gdn/b_projection` | `weight/000230` | bf16 | contiguous_le_v1 | `[96, 2560]` | `[122880, 245760)` | `[48,96)` |
| `gdn/a_log` | `weight/000537` | fp32 | contiguous_le_v1 | `[48]` | whole | — |
| `gdn/dt_bias` | `weight/000538` | fp32 | contiguous_le_v1 | `[48]` | whole | — |
| `gdn/convolution` | `weight/000536` | bf16 | contiguous_le_v1 | `[4, 10240]` | whole | — |
| `gdn/norm` | `weight/000539` | bf16 | contiguous_le_v1 | `[128]` | whole | — |
| `gdn/output` | `weight/000535` | q8_g32_fp16 | row_split_k128_v1 | `[2560, 6144]` | whole | — |

The four input projections share one `[16384, 2560]` Q8 group-32 row-split parent in the order
`Q | K | V | Z`, whose widths are `2048 | 2048 | 6144 | 6144`. The two control projections
share one `[96, 2560]` BF16 parent in the order `A | B`, 48 rows each. The convolution is
stored tap-major: element `(tap j, channel c)` lives at `j * 10240 + c`, and the runtime
`{10240, 4}` view is only the shape label `causal_conv1d_silu` consumes, because the kernel
reads `weight[j * C + c]`. The GDN norm weight is the per-head-dimension gain `[128]`.

The GDN post-norm is `ops::gated_rmsnorm` with a SiLU gate, which is what `hidden_act = silu`
selects. `output_gate_type = sigmoid` belongs to the QSA attention output gate and does not
change the GDN post-norm.

### 4.3 QSA

| Parameter | Object | Format | Layout | Shape | Range | Rows |
|---|---|---|---|---|---|---|
| `attention/query` | `weight/000245` | q8_g32_fp16 | row_split_k128_v1 | `[12288, 2560]` | `[0, 15728640)` | `[0,6144)` |
| `attention/output_gate` | `weight/000245` | q8_g32_fp16 | row_split_k128_v1 | `[12288, 2560]` | `[15728640, 31457280)` | `[6144,12288)` |
| `attention/key` | `weight/000246` | bf16 | contiguous_le_v1 | `[1664, 2560]` | `[0, 1310720)` | `[0,512)` |
| `attention/value` | `weight/000246` | bf16 | contiguous_le_v1 | `[1664, 2560]` | `[1310720, 2621440)` | `[512,1024)` |
| `attention/index_query` | `weight/000246` | bf16 | contiguous_le_v1 | `[1664, 2560]` | `[2621440, 3932160)` | `[1024,1536)` |
| `attention/index_key` | `weight/000246` | bf16 | contiguous_le_v1 | `[1664, 2560]` | `[3932160, 4259840)` | `[1536,1664)` |
| `attention/output` | `weight/000569` | q8_g32_fp16 | row_split_k128_v1 | `[2560, 6144]` | whole | — |
| `attention/query_norm` | `weight/000570` | bf16 | contiguous_le_v1 | `[256]` | whole | — |
| `attention/key_norm` | `weight/000571` | bf16 | contiguous_le_v1 | `[256]` | whole | — |
| `attention/index_query_norm` | `weight/000572` | bf16 | contiguous_le_v1 | `[128]` | whole | — |
| `attention/index_key_norm` | `weight/000573` | bf16 | contiguous_le_v1 | `[128]` | whole | — |

The text QSA layer has two parents and two layouts: the query and the output gate are the two
halves of one `[12288, 2560]` Q8 group-32 row-split parent (`query` rows `[0,6144)`, gate rows
`[6144,12288)`), while `key`, `value`, `index_query` and `index_key` are four row ranges of one
`[1664, 2560]` BF16 `contiguous_le_v1` parent in the order `K | V | index-Q | index-K`
(`512 | 512 | 512 | 128` rows). The upstream consumer keeps these two representations apart:
`MainQsaProjection` binds `{query, gate}` and `{key, value, index_query, index_key}` as two
separate projections (upstream `src/models/qwen4_exp/execution/parameters.cpp:70-71`).

### 4.4 MoE

| Parameter | Object | Format | Layout | Shape | Range | Rows |
|---|---|---|---|---|---|---|
| `moe/router` | `weight/000228` | bf16 | contiguous_le_v1 | `[1793, 2560]` | `[0, 1310720)` | `[0,512)` |
| `moe/shared_score` | `weight/000228` | bf16 | contiguous_le_v1 | `[1793, 2560]` | `[1310720, 1313280)` | `[512,513)` |
| `moe/shared/gate` | `weight/000228` | bf16 | contiguous_le_v1 | `[1793, 2560]` | `[1313280, 2951680)` | `[513,1153)` |
| `moe/shared/up` | `weight/000228` | bf16 | contiguous_le_v1 | `[1793, 2560]` | `[2951680, 4590080)` | `[1153,1793)` |
| `moe/shared/down` | `weight/000534` | bf16 | contiguous_le_v1 | `[2560, 640]` | whole | — |
| `moe/experts/gate` | `weight/000000` | nvfp4 | batched_block_scale_k16_m128x4_v1 | `[512, 1280, 2560]` | 512 parts | matrix rows `[0,640)` |
| `moe/experts/up` | `weight/000000` | nvfp4 | batched_block_scale_k16_m128x4_v1 | `[512, 1280, 2560]` | 512 parts | matrix rows `[640,1280)` |
| `moe/experts/down` | `weight/000001` | nvfp4 | batched_block_scale_k16_m128x4_v1 | `[512, 2560, 640]` | 512 parts | whole matrix |

The router, the shared logit and the shared expert's gate and up projections are four row
ranges of one `[1793, 2560]` BF16 `contiguous_le_v1` parent: 512 router rows, one shared logit row, and
640+640 shared expert rows. The first 513 rows are the routing control weight `[513, 2560]`
(`E+1` rows: the routed logits, then one independent shared logit); the remaining 1280 rows
are the shared expert's gate_up `[1280, 2560]`. The shared down projection is an independent
`[2560, 640]` object.

The routed expert bank is two batched NVFP4 objects. `experts/gate` and `experts/up` are row
ranges inside each expert's `[1280, 2560]` matrix of the `[512, 1280, 2560]` bank (gate rows
`[0,640)`, up rows `[640,1280)`); `experts/down` is the whole `[2560, 640]` matrix of each
expert of the `[512, 2560, 640]` bank. Each of the three parameters has 512 binding parts, one
per expert, and each part is a matrix range of the batched object.

### 4.5 PLE

| Parameter | Object | Format | Layout | Shape | Range | Rows |
|---|---|---|---|---|---|---|
| `ple/table` | `weight/000098` .. `weight/000225` (128 shards) | nf4_g16_e4m3fn | row_paged_g16_4k_v1 | `[2500012, 160]` | `[0, 400001920)` each | whole shard |
| `ple/key` | `weight/000236` | bf16 | contiguous_le_v1 | `[12800, 2560]` | `[0, 26214400)` | `[0,10240)` |
| `ple/value` | `weight/000236` | bf16 | contiguous_le_v1 | `[12800, 2560]` | `[26214400, 32768000)` | `[10240,12800)` |
| `ple/norm_key` | `weight/000550` | bf16 | contiguous_le_v1 | `[10240]` | whole | — |
| `ple/norm_query` | `weight/000551` | bf16 | contiguous_le_v1 | `[10240]` | whole | — |
| `ple/norm_conv` | `weight/000552` | bf16 | contiguous_le_v1 | `[10240]` | whole | — |
| `ple/convolution` | `weight/000553` | bf16 | contiguous_le_v1 | `[4, 10240]` | whole | — |

The PLE key and value projections are two row ranges of one `[12800, 2560]` BF16 parent: the key
projection is `[10240, 2560]` (the four streams of 2560 features) and the value projection is
`[2560, 2560]` (shared by all four streams). The three norm weights are the `[10240]`
stream-major vectors of the gate, the query and the convolution; the convolution kernel is
stored tap-major over the 10240 channels.

The table is 128 NF4 shards. Each shard object has the full `[2500012, 160]` geometry and
`row_paged_g16_4k_v1` encoding (45 rows per 4096-byte page, 80 low-even NF4 code bytes and ten
E4M3 scale bytes per row). The conversion report maps shard `i` to the source tensor
`ple_embedding.ngram_embedding.shard_i.weight`, so the shards are the ordered list of the
checkpoint's n-gram embedding modules. Section 8.1 derives the global row space that the
hash addresses.

### 4.6 Embedding and main head

| Parameter | Object | Format | Layout | Shape |
|---|---|---|---|---|
| `text/token_embedding` | `weight/000525` | bf16 | contiguous_le_v1 | `[248320, 2560]` |
| `text/output_head` | `weight/000526` | bf16 | contiguous_le_v1 | `[248320, 2560]` |

Both are full-vocabulary BF16 matrices (1,271,398,400 B each). The MTP component reuses the same
two objects (`mtp/token_embedding`, `mtp/output_head`).

## 5. The four-stream gated residual

The upstream contract is `include/ninfer/ops/gated_residual.h` (upstream lines 11-147). The
residual is a `[H, S, M]` BF16 tensor with `H = 2560`, `S = 4` and `M = tokens * batch`; the
channel index is `c = d + H*s`. Every norm gain is an **offset** gain: `1 + w` is not rounded to
BF16 before the multiplication.

### 5.1 `hc_expand_norm` (upstream lines 11-28)

```text
residual[d,s,m]   = embedding[d,m]                       (exact BF16 word copy)
normalized[d,s,m] = embedding[d,m] * (1 + norm_weight[d,s])
                    / sqrt( sum_d embedding[d,m]^2 / H + epsilon )
```

The embedding is replicated across the four streams by word copy, including signed zero. The
norm weight is `[H, S]`, viewed from the flat `[10240]` parameter.

### 5.2 `hc_grouped_rmsnorm` (upstream lines 30-40)

```text
normalized[d,s,m] = residual[d,s,m] * (1 + norm_weight[d,s])
                    / sqrt( sum_d residual[d,s,m]^2 / H + epsilon )
```

Each stream has its own statistics and its own affine weight.

### 5.3 `hc_combine` (upstream lines 42-54)

```text
updated_residual[d,s,m] = residual[d,s,m] + injection[s,m] * block_output[d,m]
```

`block_output` is `[H, M]` and `injection` is FP32 `[S, M]` holding already activated
coefficients in `[0,2]`. The coefficient is applied directly: there is no second sigmoid and no
further scaling at this step.

### 5.4 `hc_combine_norm` (upstream lines 56-67)

`hc_combine` followed by `hc_grouped_rmsnorm` over the **actual BF16 words** of
`updated_residual`, with the next HC's norm weight as the offset gain. The gain `1 + w` is not
rounded to BF16.

The upstream contract declares a second `hc_combine_norm` overload as well (upstream lines
69-76): a unit-weight broadcast-add followed by the same grouped offset RMSNorm, with the
math, the BF16 residual boundary, the layouts and the alias rules of the injection overload at
coefficient 1. It neither reads nor creates an injection tensor.

**This repository does not take that overload.** Every combine of the Flash-Next text trunk
carries an injection: the upstream trunk calls the injection overload at both of its combine
sites (upstream `src/models/qwen4_exp/execution/text.cpp:129` and `:134`), and the PLE path has
its own `ple_conv_add_norm`. The unit-weight overload therefore has no call site, and the
operation contract this repository writes does not include it.

### 5.5 `hc_mix` (upstream lines 78-115)

```text
[a, b]   = down_injection * flatten(normalized)        a: 320, b: 4
u        = SiLU(a / 4)                                  SiLU(v) = v * sigmoid(v)
g        = up * u                                       g: [10240, M]
block_input[d,m]  = sum_s sigmoid(g[s*H + d, m]) * normalized[d,s,m] / 4
injection[s,m]    = 2 * sigmoid(b[s, m] / 4)
```

The down and injection projections are one weight: `down_injection` is the BF16 `[324, 10240]`
parent whose rows `[0,320)` are the down projection and whose rows `[320,324)` are the injection
projection. `up` is the BF16 `[10240, 320]` parent with row `s*2560 + d`. The stream gates are
independent, and `injection` is FP32 with saturation endpoints 0 and 2. The intermediate `a`,
`b`, `u` and `g` are private; they are not observable cast boundaries.

### 5.6 `hc_final_mix` (upstream lines 117-123)

The same formula with the down-only `[320, 10240]` weight: there is no `b` and no injection, and
the result is the BF16 sample hidden `[H, M]`.

## 6. The text trunk order

The trunk order is the upstream `src/models/qwen4_exp/execution/text.cpp` (upstream lines
28-171).

```text
stem:   embedding -> hc_expand_norm(input, layers[0].attention_hc.norm) -> residual, normalized

layer i (0 .. 47):
  1  hc_mix(normalized, layers[i].attention_hc.mix)        -> input, injection
  2  mixer(input)                                           -> output      (GDN or QSA)
  3  hc_combine_norm(residual, output, injection,
                     layers[i].moe_hc.norm)                 -> residual, normalized
  4  hc_mix(normalized, layers[i].moe_hc.mix)               -> input, injection
  5  MoE(input)                                             -> output
  6  combine:
       i + 1 == ple.layer (1)   -> hc_combine(residual, output, injection, residual)
       otherwise                -> hc_combine_norm(residual, output, injection, next_norm)
                                   next_norm = layers[i+1].attention_hc.norm
                                   next_norm = final_hc.norm when i + 1 == layers.size()
  7  when i + 1 == ple.layer: PLE runs (lookup -> gate -> conv+norm),
     and its next HC is layers[ple.layer].attention_hc

finish: hc_final_mix(normalized, final_hc.mix) -> sample_hidden -> main head -> logits
```

Two consequences of this order are worth stating explicitly, because both are easy to get wrong
and neither is visible in a shape assertion:

- The MoE HC's norm weight is applied by step 3, **before** the MoE consumes the mixed input of
  step 4. The residual that the layer's combine writes is therefore normalized with the
  *next* layer's attention HC norm (step 6), not with the current layer's.
- The PLE layer's combine deliberately skips the normalization (step 6, first branch) and the
  PLE supplies the normalization itself, because PLE adds to the residual between the two
  layers.

The MTP conditioning is the four-stream residual **before** the final mixer, not the sample
hidden (`src/models/qwen4_exp/execution/text.cpp:153-165`).

## 7. The QSA mathematics

The upstream contract is `include/ninfer/ops/qsa.h` (upstream lines 11-373). The QSA layer keeps
the complete main K/V; the four-token compression applies to the index key only.

### 7.1 `qsa_prepare` (upstream lines 223-302)

Normalization and rotation:

```text
N_D(x, w) = x * (1 + FP32(w)) / sqrt( sum(x*x) / D + rms_epsilon )      (per head)

R(z, rho) rotates the pairs (i, i+32), i in [0,32), by rho[axis(i)] * theta^(-2i/64)
          axis(i) = i % 3 on the canonical three-axis rho
          R[i]    = z[i] * cos - z[i+32] * sin
          R[i+32] = z[i+32] * cos + z[i] * sin
          dimensions [64, D) retain z
```

A single-axis (A=1) coordinate is broadcast to the three axes. With `p = positions[t,b]` and
`t < valid_columns[b]`:

```text
query[:,h,t,b]       = BF16( R( N_256(projected.query[:,h,t,b], query_norm), rho[t,:,b] ) )
index_query[:,h,t,b] = BF16( R( N_128(projected.index_query[:,h,t,b], index_query_norm), rho ) )
k                   = R( N_256(projected.key[:,h,t,b], key_norm), rho )
v                   = projected.value[:,h,t,b]
```

Cache encodings: a BF16 main cache stores BF16(k) and FP16_RNE(v). For a quantized main cache,
`k` enters the fixed FP32 H256/16 Hadamard and the storage encoder directly from norm and RoPE,
with no intermediate BF16 rounding; INT8-G64 and FP8 transform K only, while NVFP4-G16 and
K8V4 transform both K and V. The attention consumer owns the matching Q transform and any
value-basis inversion.

Supported route gap: the implemented QSA route in this package is the BF16-K / FP16-V cache.
INT8 group-64 is described above as a cache encoding but is not implemented here, so no test
exercises it; the M2 acceptance run uses BF16-K / FP16-V.

Index blocks: with `F = positions[0,b]` and `n = valid_columns[b]`, each newly completed block
`g` in `[floor(F/4), floor((F+n)/4))` writes

```text
u = BF16( mean(raw_index_key[4g : 4g+4]) )
index.K[g] = BF16( R( N_128(u, index_key_norm), rho[4g] ) )
```

Raw keys preceding `F` come from the previous call's canonical tail at `F`; the others come
from the projection. A block that starts before `F` uses the tail's first RoPE position,
otherwise the current call's coordinates. The raw index-key words and the actual coordinates are
copied to the records; A=1 replicates the coordinate to all three record axes. Only complete
blocks are written, and no causal frontier is published. Main token `p` addresses
`table[p/64, table_rows[b]]` row `p%64`; index block `g` addresses the same table at `g/16` row
`g%16`.

### 7.2 `qsa_prepare_kv` (upstream lines 304-320)

The K/V, compressed index-key and raw-record effects of `qsa_prepare` without the query inputs
and outputs. The K normalization and RoPE, the V encoding, the four-token index-key pooling
with its BF16 mean boundary and the cross-call tail and coordinate selection are identical.

### 7.3 `qsa_indexer` (upstream lines 114-156)

For each live `(t,b)` with `p = positions[t,b]` and `C = floor((p+1)/4)`, and `g` in `[0,C)`:

```text
score[g] = sum_{h=0..3} max( 0, sum_{d=0..127} index_query[d,h,t,b] * index.K[d,g] )
```

The common positive `1/sqrt(128)` factor is omitted because only the ranking is observable. The
selection keeps `min(512, C)` highest scores, and exact numeric ties prefer the smaller logical
`g`. Block `g` addresses `keys.records.page_offsets[g/16, table_rows[b]]`, row `g%16` of its
record. The four index heads produce **one shared selection** for all query heads, and the
incomplete token tail is not part of the selection. When `C <= 512` every block is selected
without reading Q, the keys or the page table.

### 7.4 `qsa_sparse_gqa` (upstream lines 173-221)

For each live `(t,b)` with `p = positions[t,b]`, `C = floor((p+1)/4)` and `c = block_counts[t,b]`:

```text
J = { 4g, 4g+1, 4g+2, 4g+3 : g in the first c block_ids }  union  [4C, p+1)

for head h, kvh = floor(h/12):
  score[j]  = dot(query[:,h,t,b], K[:,j,kvh]) / 16
  attention[d,h] = sum_{j in J} softmax(score)[j] * V[d,j,kvh]
  out[d,h,t,b]   = BF16( attention[d,h] * sigmoid(FP32(output_gate[d,h,t,b])) )
```

There is no attention-output BF16 rounding before the gate, and the indexer scores never enter
the logits. Logical token `j` addresses `table[j/64, table_rows[b]]` row `j%64`. An empty `J`
and an inert query produce exact `+0`. For a quantized cache the keys represent `R*K` with
`R = H256/16`, so the logits use `R*query` and decoded keys; NVFP4 and K8V4 values represent
`R*V`, so `R^T` is applied to the fully reduced attention before the gate. A query attends at
most `512*4 + 3` tokens.

### 7.5 `qsa_tail_extract` (upstream lines 322-371)

For row `b`, with `F0 = records.base_positions[b]`, `a = accepted_columns[b]`, `F1 = F0 + a`,
`r = F1 % 4` and `s = F1 - r`:

```text
after.raw_keys[c,j,b] = BF16 +0                                  if j >= r
                         before.raw_keys[c, s+j-(F0-F0%4), b]      if s+j < F0
                         records.raw_keys[c, s+j-F0, b]            otherwise

after.first_rope[:,b] = [0,0,0]                                  if r == 0
                         before.first_rope[:,b]                   if s < F0
                         records.rope_positions[:, s-F0, b]        otherwise
```

`before` is the canonical state at `F0`: its first `F0%4` raw columns hold the unfinished
four-token block in chronological order, the remaining columns are BF16 `+0`, and `first_rope`
holds that block's first token coordinates, or three I32 zeros when `F0 % 4 == 0`. The
extraction is exact word and I32 copying; no floating-point evaluation is involved.

## 8. The PLE mathematics

The upstream contract is `include/ninfer/ops/ple.h` (upstream lines 12-245) and
`src/models/qwen4_exp/ple/hash.cpp` (upstream lines 14-41). The shared domain is
`H = 2560` features per stream, `C = 4` streams and `D = H*C = 10240` channels, with channel
`c = d + H*s`. The PLE norm is the same offset RMSNorm as the HC:

```text
Norm(x, w, eps)[d] = x[d] * (1 + w[d]) / sqrt( sum_i x[i]^2 / H + eps )
```

PLE consumes the **raw** token ids: vision features never replace the ids that feed the hash.

### 8.1 The n-gram hash

The `ple_hash_controls` resource is 280 bytes: three U64 multipliers followed by sixteen U64
vocabulary sizes and sixteen U64 offsets. Measured from the artifact:

```text
multipliers m[0..2] = 23703573157769, 20109073645365, 8052911324071

vocab_sizes[h] = 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
                 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
                 20000161, 20000171

offsets[h]      = 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
                 140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
                 260000955, 280001114, 300001275
```

The hash of one token id `x` with the two preceding ids `tail = {older, previous}` (oldest
first) is:

```text
older  = eos if tail[1] == eos else tail[0]
pair   = U64(x) * m[0] XOR U64(tail[1]) * m[1]
triple = pair XOR U64(older) * m[2]

for h in 0..15:
  word  = int64_bits( pair if h < 8 else triple )
  index = word % vocab_sizes[h]      (positive modulo: a negative remainder adds vocab_sizes[h])
  row[h] = index + offsets[h]
```

The tail advances to `{tail[1], x}` after every token. Heads 0..7 use the pair, heads 8..15 use
the triple, so one token addresses 16 rows. The row space is therefore partitioned among the
heads: head `h` owns the global rows `[offsets[h], offsets[h] + vocab_sizes[h])`.

The sixteen vocabulary sizes sum to 320,001,446 global rows, while the table holds
`128 * 2,500,012 = 320,001,536` rows.

**The shard assignment is an assumption, not a proof.** The artifact directory does
not record the global row range of each shard: every shard's binding claims the whole object
(`[0, 400001920)`), and only the conversion report's `methods` entries show that shard `i`
comes from the source tensor `ple_embedding.ngram_embedding.shard_i.weight`. The assignment
follows from the ordered shard list together with the hash controls: the global row space is
partitioned among the heads by `offsets`, the shards hold the global rows in order, and the 90
rows of difference are the unused tail of the last shard. On that reading, global row `r`
would live in shard `r / 2500012` at row `r % 2500012` of that shard.

Task 8's qualification test pins the shard index against the conversion report's `methods`
entries for all 128 shards, their shape, layout and PLE layer role, and covers rows on both sides
of a shard boundary. It does not prove the ordering itself: the ordered-shard assumption is
carried forward and settled by the reference model's per-token parity in task 11, where a wrong
shard or a wrong row offset selects a plausible but wrong embedding that no shape or range
assertion can see.

### 8.2 The row lookup

`ngram_embedding_lookup` (upstream lines 21-60) concatenates sixteen exact row lookups into a
BF16 embedding `[2560, T, B]`. For each live `(t,b)`, `x` is the token id, `p` the preceding id
and `o` the id before `p`; ids before `t = 0` come from the previous call's tail, and a `p`
equal to `eos` replaces `o` by `eos`. Each head contributes 160 features in head order, so the
concatenation is `[2560, T, B]`. A row that no shard covers contributes exact `+0` words.

The shard pages keep the `row_paged_g16_4k_v1` encoding: 45 rows per 4096-byte page, each row
holding 80 low-even NF4 code bytes followed by ten non-negative E4M3 scale bytes (0..126). The
upstream Op indexes a prepared BF16 word table (`make_nf4_bf16_lut()`) and performs no
floating-point arithmetic, so signed zero is preserved.

### 8.3 `ple_gate` (upstream lines 143-172)

For each active `(t,b)` with `0 <= t < n[b]` and each stream `s`, with `eps = rms_epsilon`:

```text
k_s = Norm(projected.key[:,s,t,b],   norm_key[:,s],   eps)
q_s = Norm(residual[:,s,t,b],        norm_query[:,s], eps)
a_s = sum_d k_s[d] * q_s[d] / sqrt(H)
v_s = sign(a_s) * sqrt( max(|a_s|, 1e-6) )
g_s = 1 / (1 + exp(-v_s))
ideal_U[d,s,t,b]   = g_s * projected.value[d,t,b]
ideal_N[:,s,t,b]   = Norm(U[:,s,t,b], norm_conv[:,s], eps)
```

`sign(0) = 0`, so the gate is 0.5 at zero. The gate floor `1e-6` is independent of `eps`. `U` is
the represented value of the actual BF16 `gated_value` output, and the convolution norm must
consume exactly those words.

### 8.4 `ple_conv_add` and `ple_conv_add_norm` (upstream lines 174-215)

The convolution input is the nine preceding normalized columns plus the current ones:

```text
N[c,k,b] = before.history[b][c, 9+k]                 for -9 <= k < 0
           current.values[c, current.offsets[b]+k, b] for 0 <= k < n[b]

for 0 <= t < n[b], c = d + H*s:
  v[c,t,b]           = sum_{j=0..3} kernel[c,j] * N[c,t-9+3j,b]
  ideal_Y[d,s,t,b]   = residual[d,s,t,b] + gated_value[d,s,t,b] + SiLU(v[c,t,b])
  ideal_Z[:,s,t,b]   = Norm(Y[:,s,t,b], next_hc_norm_weight[:,s], next_hc_epsilon)
```

`SiLU(x) = x / (1 + exp(-x))`. Tap 0 reads `t-9` and tap 3 reads `t`; the dilation is 3 and
there is no bias. The norm entry consumes the actual `updated_residual` words. There is no
public intermediate rounding between the convolution, the SiLU and the residual sum.

### 8.5 `ple_state_extract` (upstream lines 217-243)

For row `b` with `a = accepted_columns[b]`, channel `c` and `j` in `[0,9)`, with `k = a-9+j`:

```text
after.history[b][c,j] = before.history[b][c, 9+k]   if k < 0
                        records.values[c, k, b]    otherwise
```

`a = 0` copies the complete `before` history. The extraction is exact BF16 word copying.

## 9. MoE routing and the expert geometry

### 9.1 The routing control

The upstream `MoeWeights.control` is a BF16 `[E+1, H]` weight: the routed logits, then one
independent shared logit (`include/ninfer/ops/moe.h:110-114`). In this artifact the control is
the first 513 rows of the `[1793, 2560]` parent: rows `[0,512)` are `moe/router` and row 512 is
`moe/shared_score`.

`topk_softmax_route` (upstream lines 131-143) computes, per token, `l = control * x` in BF16,
selects the K = 10 greatest `l[0:E]` descending with ties broken by the lower expert id, and
publishes their ids together with their softmax coefficients. `norm_topk_prob = true` makes
those coefficients the softmax taken over the ten selected logits, so they sum to one. The
shared coefficient is `sigmoid(l[E])`, published separately.

### 9.2 The expert contributions

`F_e(x) = down_e * (activation(gate_e * x, up_e * x))` with the coefficient exactly one
(upstream lines 194-220). The routed contribution is
`sum_j coefficient[j,t] * F_{ids[j,t]}(x[:,t])` over the selected experts. The shared expert
contributes `sigmoid(l[E]) * F_shared(x[:,t])` exactly once, with the same SwiGLU shape as a
routed expert. A top-k selection restricted to a subset is not renormalized.

The shared expert's `gate_up` is the `[1280, 2560]` row range `[513,1793)` of the `[1793, 2560]`
parent (gate rows first, then up rows) and its `down` is the independent `[2560, 640]` object.

### 9.3 The batched NVFP4 expert bank

Both routed banks use `batched_block_scale_k16_m128x4_v1`: a code plane, a scale plane and one
FP32 divisor per matrix. The element order is matrix-major, then row-major inside the matrix,
so the binding ranges of section 4.4 are matrix ranges. The gate_up bank is `[512, 1280, 2560]`
(943,720,448 B) and the down bank is `[512, 2560, 640]` (471,861,248 B). When a rank or a
step selects a subset of experts, the scale tiles and the per-matrix divisor must follow the
expert: a mispaired scale or divisor leaves the totals plausible while each expert is wrong.

### 9.4 The activation policies

The 1205 use edges carry the policy of the consuming operator: 1058 `A16Only`, 144 `AllowA4`
and 3 `AllowA8`. The 144 `AllowA4` edges are exactly the three routed expert parameters of the
48 text MoE layers (`experts/gate`, `experts/up`, `experts/down`); each of them carries the
auxiliary `activation_input_divisor`, a range of the FP32 `[48, 2]` object
`auxiliary/shared/expert_activation_divisors`. The three `AllowA8` edges are the MTP expert
parameters. Everything else in the text namespace is `A16Only`.

## 10. The decode boundary

The repository's typed binding maps each numeric format onto exactly one canonical storage
layout (`src/artifact/typed_binding.cpp:16-38`), and the ops consume the decoded planes of that
layout. Against the artifact's census:

| Artifact format and layout | Status in this repository |
|---|---|
| `bf16` / `contiguous_le_v1` | Usable as is. The BF16 linear path and the elementwise ops consume BF16 natively. |
| `fp32` / `contiguous_le_v1` | Usable as is. The GDN control vectors and the activation divisors are not GEMM operands. |
| `q8_g32_fp16` / `row_split_k128_v1` | Usable as is. The W8 group-32 row-split GEMM exists (`src/ops/linear/w8`). |
| `q4_g64_fp16` / `row_split_k128_v1` | Usable as is. The Q4 group-64 row-split GEMM exists (`src/ops/linear/q4`); this object is the M3 proposal head. |
| `nvfp4` / `batched_block_scale_k16_m128x4_v1` | **Not usable yet.** The repository's NVFP4 path implements the non-batched `block_scale_k16_m128x4_v1` layout; the batched variant (per-matrix scale plane plus per-matrix FP32 divisor) has no kernel and no typed binding. The routed expert bank needs one. |
| `nf4_g16_e4m3fn` / `row_paged_g16_4k_v1` | Decoded. The reader addresses a row (`row_paged_nf4_geometry`, `row_paged_nf4_row_offset`) and the PLE lookup Op (`src/ops/ple/lookup.cu`) yields the BF16 words of the codebook product divided by the stored per-shard FP32 divisor. The table is not resident as a whole: the caller stages the addressed row bytes and the 128 divisors. |
| `fp8_e4m3fn_block128x128_bf16` / `block_scale_m128_k128_v1` | **Not usable yet.** The repository's FP8 GEMM is the row-scaled `FP8_E4M3FN_ROW_BF16S` format; the block-128x128 variant has no GEMM. Both objects of this format are the MTP expert banks (M3). |

The `text/*` parameter that still needs a new decode path is therefore the routed expert bank
(batched NVFP4). The PLE table (NF4) is decoded by `src/ops/ple/lookup.cu`. Every other text
parameter is BF16, FP32 or a Q8 row-split projection, all of which the repository already
decodes.

## 11. Host tier slab geometry

The host tier carves its pinned budget into fixed-size slabs sized from the plane extents of the
planned caches (`src/targets/qwen3_6/impl/state/decoder_state.cpp:37-52`, split 80% block / 20%
snapshot in `src/targets/qwen3_6/impl/runtime/program_impl.h:626-640`). These are flash-next
numbers; the 2.19 MiB / 373 slab / 4485 figures in `docs/maintainer/multi-instance.md` §2.1 belong
to the 27B NVFP4 TP2+MTP-3 profile and do not transfer.

Block slab (one 64-token page, INT8 group-64 KV with the FP16 scale plane). The flash-next Variant
maps `convolution_dim` and `gdn_conv_state_width` onto `gdn_conv_channels` and
`gdn_conv_kernel - 1`; the GDN conv of this checkpoint has no dilation, so the state width is
`kernel - 1` (the dilation 3 belongs to the PLE conv).

| Plane, per rank | Extent | Bytes |
|---|---|---|
| K codes | `256 × 64 × (kv_heads/tp = 1) × 1` | 16,384 |
| V codes | same | 16,384 |
| K scales | `(256/64) × 64 × 1 × 2` | 512 |
| V scales | same | 512 |
| Per QSA layer | | 33,792 |
| Per rank (12 QSA layers) | | 405,504 |

Block slab = `tp × per_rank` = **811,008 B ≈ 0.773 MiB**. A slab is a full-width page, so this total
is tp-independent (`tp = 1` gives the same slab from `67,584 × 12`). The speculative backend pool
contributes 0 today because this identity registers no speculative backend yet (MTP is M3); when it
lands, the backend pool's `per_page_bytes()` adds to `per_rank_block_bytes`.

Snapshot slab (one GDN continuation state image):

| Part, per rank | Extent | Bytes |
|---|---|---|
| Conv history, per layer | `(10240/tp = 5120) × 3 × 2` | 30,720 |
| Recurrent state, per layer | `128 × 128 × (48/tp = 24) × 4` | 1,572,864 |
| Per layer | | 1,603,584 |
| Per rank (36 GDN layers) | | 57,729,024 |

Snapshot slab = `tp × per_rank` = **115,458,048 B ≈ 110.1 MiB**.

At `--host-cache-mib 1024` the split gives 858,993,459 B of block budget → **1059 block slabs**
(≈ 67,776 host prefix tokens) and 214,748,364 B of snapshot budget → **1 snapshot slab**. The
snapshot slab is ~142× the block slab, so a 1024 MiB budget retains a single state image; meaningful
snapshot retention needs a much larger budget (`--host-cache-mib 12288` → 12,704 block slabs,
≈ 813K tokens, and 22 snapshot slabs). The smallest budget that enables the tier at all is one slab
of each kind, which for this geometry is ≈ 551 MiB.

## 12. Provenance

- Artifact: `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`
  (entry file plus `part-0001..part-0003`).
- Conversion report:
  `/home/cmj/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer.conversion.json`
  (`components`, `objects`, `methods`, `formats`).
- Directory readings: `components.text.config`, `bindings`, `part_bindings`, `uses` and
  `objects` of the artifact directory, read through `Reader::v3_directory()`
  (`src/artifact/reader.h`).
- PLE hash constants: the 280-byte `resource/text/ple_hash_controls` payload of the artifact.
- Upstream semantics: the Qwen4Exp files of the upstream NInfer repository, cited inline with
  their line ranges; the formulas above are transcribed from them and the Op contracts of this
  repository must be written to match.
- Role table and its test: `src/targets/qwen3_8_flash_next/impl/load/roles.{h,cpp}` and
  `tests/targets/qwen3_8_flash_next/test_roles.cpp`.


## 13. M2 acceptance configuration

The M2 text acceptance run is a test configuration, not a product load configuration: the public
Engine route always loads the full 48 layers.

- The test binds the reduced role set — the first four layers with all 512 experts, the full
  embedding and the full main head — and materializes only those objects on device. Objects
  outside that set are validated only, which is the binder's existing placement contract.
- The reduced stack is a prefix truncation of the real graph, so its last loaded layer (layer 3)
  must normalize with the layer the full graph would have used next, `text/layers/4/attention_hc/norm`.
  That single tensor is therefore part of the reduced role set, and the final mixer
  (`text/final_hc/{down,up}`) still produces the sample hidden. A reference model reproduces the
  same rule: layers `[0, N)` plus the next layer's attention HC norm, then `final_hc.mix`.
- The PLE table is not materialized as a device tensor. `ops::PleNf4Table` carries no per-shard
  device base, so the lookup keeps the table on disk and stages only the rows a call addresses,
  as §8.2 describes. The text state derives the 16 row addresses of each forward with
  `ple_hash_tokens`, fetches those encoded rows through the target's file-backed row source, and
  stages them together with the 128 divisor words. Materializing all 128 shards as one device
  table would require changing the Op contract, which this milestone does not do.
- Greedy parity compares the produced token with `NINFER_FLASH_NEXT_REFERENCE_TOKEN`. The M2
  mathematical oracle is `tools/reference/qwen3_8_flash_next`: it decodes the stored codes, scales
  and divisors (including the per-shard PLE divisors and the per-expert NVFP4 divisors) into a
  naive FP64/FP32 evaluation, uses BF16-K / FP16-V QSA cache semantics, and reproduces the
  observable BF16/FP32 storage boundaries. The 8-token configuration produces token `20560` with
  top-two logit margin `0.086052638`.
- `tests/targets/qwen3_8_flash_next/test_reference_parity.py` checks the greedy token, the
  sixteen PLE hash row numbers of every token, and the selected QSA block set of every token.
  The acceptance criteria are:
  - Short prompt (8 tokens): the greedy token must equal the reference token exactly, and the
    top-two logit margin must be at least `1e-2`. Measured: token `20560`, margin `0.086052638`.
  - Long prompt: the operator-level evidence is exact per-token block-set equality when the
    reference indexer is fed the same `index_query` and `index_keys` as the production kernel.
    Measured at 2100 tokens (`C = 525`, 512 selected): `2100/2100` tokens equal. The model-level
    evidence is exact per-token set equality at 2052 tokens (`C = 513`). At 2100 tokens a
    divergent model-level block set is acceptable only on a token whose truncated score gap
    (rank 512 versus rank 513) is below the measured maximum score perturbation `2.307295`; the
    run must report the number of divergent pruned tokens and the reference boundary gap `δ`.
    Measured: `16` divergent of `49` pruned tokens, every divergent token has a boundary gap in
    `[1.43e-2, 7.21e-1]`, and `δ = 0.7208509`. The greedy token still equals the reference
    (`225013` at 2052 tokens, margin `0.00173172358`).
  - The 2100-token model-level divergence is a conditioning result, not an implementation
    defect. Section 5 rounds `residual` and `normalized` to BF16 words, and `hc_combine_norm`
    amplifies a one-ulp difference when a per-stream RMS is near zero; the smallest measured
    layer-0 per-stream RMS is `1.6761e-3` (stream 0, token 98). The production kernel's FP32
    accumulation and BF16 outputs therefore do not reproduce the reference FP64 evaluation
    bit-for-bit, and the QSA top-512 boundary is discontinuous; these are implementation-profile
    differences, not semantic differences.
- The tokenizer is outside M2: the test feeds token ids directly.
