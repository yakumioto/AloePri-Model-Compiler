# Artifact format

An input directory must contain `config.json` and exactly one active weight
layout:

- `model.safetensors` for a single file; or
- `model.safetensors.index.json` with a `weight_map` and every referenced shard
  present beneath the artifact root.

The discovery layer rejects `.bin` fallback, duplicate JSON keys, absolute or
parent-traversing shard paths, duplicate physical tensor names, missing or
extra index entries, conflicting shard metadata, header truncation, overflow,
negative/reversed/out-of-bounds ranges, overlaps, holes, and shape/dtype byte
length mismatches. Header and index parsing is bounded and source config bytes
are preserved exactly.

Output uses the standard safetensors header and `__metadata__.format = "pt"`.
A small output is `model.safetensors`; multiple payload shards use
`model-00001-of-000NN.safetensors` plus a new HF index. Tensor names, shapes,
dtypes, and payload bytes are preserved; offsets and shard assignment may
change. The output directory also contains the preserved `config.json`, an
allowlisted set of HF sidecars when present, and `aloepri.json`.

Identity writes manifest/checkpoint version 1 with a null legacy
`secret_key_id`. AloePri token writes version 2 and a non-null `secret_id`.
Both versions record method, source fingerprint, plan hash, layout, config and
sidecar digests, and output tensor digests. The token manifest never contains
the permutation, inverse permutation, nonce, or a recoverable seed. The Client
Secret is written only to the caller-selected external path with exclusive
creation and restrictive permissions.
