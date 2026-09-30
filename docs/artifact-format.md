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
| KeyMat plan / manifest / checkpoint | 4 | explicit `aloepri-keymat/0.1` binding and exact_covariant runtime |

KeyMat does not upgrade identity, token or diagnostic v3 artifacts. Optional
KeyMat fields are omitted from their canonical encodings and hashes; v3 resume
remains supported. A v4 method/version mismatch is rejected.

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

## Private KeyMat bundle v1

KeyMat uses small `KeyMatSecretV1` JSON plus same-directory `key-material.bin`.
The binary format is `keymat-f64-le-v1`: canonical row-major little-endian F64
P followed by Q, exactly `16*d*D` bytes. Large matrices are never JSON arrays.
The JSON binds method/source/d/h, canonical lambda bits, algorithm/RNG versions,
master seed, nullspace cutoff and separate P/Q digests. A domain-separated
BLAKE3 commitment binds all of these as `secret_id`; Rust and Python verify the
same bytes, not separately regenerated approximations.

Both files use exclusive creation, restrictive permissions and fsync; paths
must be outside source/output/staging and cannot contain symlinks. Generation
and full plan preflight precede persistence; binary is synced before JSON and
both before model publication. This is not a cross-path atomic transaction:
a failed write may leave private orphan material. Fresh/resume reject incomplete
or corrupt bundles and never replace them automatically.

The public v4 plan contains only the binding/parameters and linear operation
roles. Manifest/checkpoint/shards contain no P/Q, private seed, material path or
key tensor. Parameter/PQ identity changes affect secret_id and plan hash; resume
also binds source, method, runtime and layout. Old v3 canonical fields/hashes
and resume behavior remain unchanged.
