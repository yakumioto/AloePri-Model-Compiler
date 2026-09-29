# Architecture adapter boundary

The registry performs exact-match detection. v0.1 registers one
`LlamaDenseAdapter`; zero matches return `UnsupportedArchitecture`, and more
than one registered match returns `AmbiguousArchitecture`. There is no generic
fallback based on directory names or substrings.

The adapter recognizes `config.json` with `model_type = "llama"` and validates
the configured dense schema: layer count, hidden size, attention and
key/value-head dimensions, intermediate size, embedding, layer norms, Q/K/V/O,
MLP gate/up/down, final norm, and (when untied) `lm_head.weight`. Grouped-query
attention is represented by the configured key/value head count rather than
assuming Q/K/V have equal widths.

Validation is fail-closed. `hidden_size`, `num_hidden_layers`,
`num_attention_heads`, `intermediate_size` and `vocab_size` are required; a
config that omits any of them is `UnsupportedArchitecture` rather than a model
that silently skips every schema check. Only the two fallbacks the Llama config
defines are applied: `num_key_value_heads` defaults to `num_attention_heads`,
and `head_dim` defaults to `hidden_size / num_attention_heads`. Attention
dimensions must be non-zero, the key/value head count must not exceed the head
count, and the head count must divide `hidden_size`; the resulting head
dimension must multiply back to `hidden_size`.

When `tie_word_embeddings` is true and the physical checkpoint omits
`lm_head.weight`, the model spec records an alias to the embedding only. The
identity plan keeps the physical inventory unchanged and never synthesizes an
output tensor. Extra physical tensors are copied transparently as physical
inventory entries; they are not interpreted as architecture semantics.
