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
`model-00001-of-000NN.safetensors` plus a new HF index. Offsets and shard
assignment may change because they are planned from the **output** descriptors,
not the source. `config.json` is preserved verbatim and always keeps describing
the *logical* architecture; it never records the physical, transformed
dimensions. The output directory also contains an allowlisted set of HF sidecars
when present, and `aloepri.json`.

## Serialized schema versions

| Object | Version | Notes |
| --- | --- | --- |
| `TransformPlan` | 3 | source *and* output inventories, runtime contract, output-based layout |
| `aloepri.json` manifest | 3 | embeds the non-secret plan and a layout hash |
| `checkpoint.json` | 3 | adds the runtime contract to the resume binding |

Legacy artifacts stay readable but are never upgraded automatically:

| Object | Behaviour |
| --- | --- |
| v1 identity / v2 token artifacts | read-only structural verification under `verification_scope = legacy_v1_v2`; no v3 plan integrity is implied |
| v1/v2 checkpoint | rejected by a v3 execution before staging is created |
| published v3 artifact + a different plan | the `--resume` fast path is refused; the output is reported as already existing |

A schema bump is explicit. Recomputing an identity or token plan under v3
produces a different plan hash than v1/v2 did; that is an intended compatibility
change, not hash equality.

## Manifest, checkpoint and hash binding

A v3 `aloepri.json` records the method, architecture, source fingerprint, plan
hash, layout hash, runtime contract, config and sidecar digests, and one digest
per output tensor. It also embeds the full non-secret `TransformPlan`, so a
verifier can recompute the plan/layout relationship instead of trusting a bare
hash string. The outer fields are a compatibility projection: `Manifest::read`
rejects any artifact whose outer method, version, architecture, fingerprint,
plan hash, layout, secret id, runtime contract or `standard_hf_checkpoint` flag
disagrees with the embedded plan.

A v3 `checkpoint.json` binds the same contract—source fingerprint, method,
runtime contract, plan hash, layout hash, and applicable `secret_id`—plus the
ordered completed-operation prefix with each finished tensor's digest. A resume
is validated before the staging directory, the output lock, the writer, or any
shard is touched. Changing the expansion/physical size, runtime contract,
method, source fingerprint, shard layout, or secret therefore fails closed and
leaves staging byte-identical.

None of manifest, checkpoint or plan contains the permutation, inverse
permutation, nonce, or any key material. The Client Secret is written only to
the caller-selected external path with exclusive creation and restrictive
permissions.
