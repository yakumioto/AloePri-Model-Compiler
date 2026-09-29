# Artifact format

An input directory must contain `config.json` and exactly one active weight
layout:

- `model.safetensors` for a single file; or
- `model.safetensors.index.json` with a `weight_map` and every referenced
  shard present beneath the artifact root.

The discovery layer rejects `.bin` fallback, duplicate JSON keys, absolute or
parent-traversing shard paths, duplicate physical tensor names, missing or
extra index entries, conflicting shard metadata, header truncation, overflow,
negative/reversed/out-of-bounds ranges, overlaps, holes, and shape/dtype byte
length mismatches. Header and index parsing is bounded and the source config
bytes are preserved exactly.

Index entries are compared by resolved shard identity rather than by raw text,
so the spellings Hugging Face itself treats as equivalent (`x`,
`./x`, `sub/x`, `./sub/x`) all resolve to the same shard and a legitimate model
is never rejected; an entry that resolves to a different shard is still
reported. An index whose `total_size` disagrees with the payload is rejected.

Output uses the standard safetensors header and `__metadata__.format = "pt"`.
A small output is `model.safetensors`; multiple payload shards use
`model-00001-of-000NN.safetensors` plus a new HF index. The shard target is a
payload-byte limit. A tensor larger than that target gets an oversize shard and
is still streamed in bounded chunks. Tensor names, shapes, dtypes, and payload
bytes are preserved; offsets and shard assignment may change.

The output directory also contains the preserved `config.json`, an allowlisted
set of HF sidecars when present, and `aloepri.json`. The manifest is v1,
records the Identity method, source and plan digests, per-tensor BLAKE3 values,
and `standard_hf_checkpoint: true`. No secret-like files are copied.
