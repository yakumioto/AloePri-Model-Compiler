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

When `tie_word_embeddings` is true and the physical checkpoint omits
`lm_head.weight`, the model spec records an alias to the embedding only. The
identity plan keeps the physical inventory unchanged and never synthesizes an
output tensor. Extra physical tensors are copied transparently as physical
inventory entries; they are not interpreted as architecture semantics.
