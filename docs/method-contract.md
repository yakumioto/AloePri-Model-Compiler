# Method contract

The product CLI exposes two method contracts:

```json
{"id":"identity","version":"0.1"}
{"id":"aloepri-token","version":"0.1"}
```

A third contract, `{"id":"expand-test","version":"0.1"}`, exists only so the
test suite can exercise a real shape-changing pipeline. It is never selectable
from the CLI and is not an AloePri algorithm.

## Runtime contract: standard versus custom

Each plan declares a runtime contract. Identity and `aloepri-token` use the
`huggingface` runtime with `standard_hf_checkpoint = true`, no expansion, and
`physical_dimensions == logical_dimensions`: their outputs remain standard HF
checkpoints and the missing tied `lm_head.weight` is restored by the loader. A
shape-changing method declares a non-standard runtime with an explicit
expansion size; `physical.hidden_size == logical.hidden_size + 2 * expansion`.
An artifact built by such a method is **not** loadable by vanilla Transformers,
and no manifest, report or document may claim otherwise.

The runtime contract is validated against the operations, not merely hashed: a
padding width must correspond to the declared physical/logical difference, and
a standard method may not declare non-standard dimensions or an expansion.

## Identity

Identity accepts only `OutputDType::Preserve` and one worker. Every source
tensor maps to exactly one copy operation and one output tensor with the same
name, shape, dtype and byte length. The operation reads source bytes through a
bounded `TensorReader` and writes them through the compiler-owned sink. It does
not decode, cast, reorder, transpose, or numerically modify bytes.

## AloePri token

`aloepri-token` supports the tested dense Llama schema and F32/F16/BF16
vocabulary matrices. It creates an external Client Secret v1 and marks the
`model.embed_tokens.weight` operation, plus a physical `lm_head.weight` when
present, as row permutations. For `tau: original -> obfuscated`:

```text
E'[tau(i)] = E[i]
H'[tau(i)] = H[i]
```

The permutation keeps the output topology identical to the source (same name,
shape, dtype, length). A tied checkpoint with no physical `lm_head.weight` keeps
that inventory; the Hugging Face loader restores the logical alias. A tied
checkpoint containing both tensors must contain identical bytes before
transformation. Unsupported biases, quantized vocabulary weights, and other
architectures fail closed.

The executor still proves the embedding really changed by comparing a digest of
the bytes it emits against the source, but the authoritative completion digest
comes from the sink.

The Client Secret contains `version`, method, `secret_id`, `vocab_size`,
`source_fingerprint`, `binding_nonce`, `token_permutation`, and
`inverse_token_permutation`. `secret_id` is a BLAKE3 commitment over the
versioned method, source fingerprint, vocabulary size, nonce, and both
permutations. Loading validates the full bijection, inverse relationship,
source binding, and commitment.

## Verification layering

Structural verification and model semantic verification are separate:

- `structure_valid` — the container parses and is internally consistent.
- `manifest_verified` / `artifact_valid` — a manifest was found and matched.
- `plan_verified` — a v3 artifact's embedded plan, layout, physical shards and
  index were recomputed and agree.
- `standard_hf_checkpoint` — the contract's declaration, reported verbatim. It
  is not the result of loading the artifact with Transformers.
- `semantic_verification` — always `not_run` here; mathematical inference
  equivalence is the separate client-side Python demo.

A raw directory with no manifest is verified as container-only: it is reported
with `standard_hf_checkpoint = null` and runtime compatibility unknown, never as
a loadable standard checkpoint. Legacy v1/v2 manifests verify under
`verification_scope = legacy_v1_v2` and are not credited with v3 plan integrity.

## Checkpoint and resume

A v3 checkpoint records the source fingerprint, method, runtime contract, plan
hash, output layout hash, public secret binding, and the ordered
completed-operation prefix with per-tensor digests. Resume recomputes the plan
and validates the full contract before staging, locking, or writing. A v1/v2
checkpoint, or any change to the expansion/physical size, runtime contract,
method, source fingerprint, shard layout or secret, is refused with staging left
untouched. A token resume reads the existing Client Secret and never generates
or overwrites one; a distinct but individually valid secret cannot resume
another secret's artifact.

The candidate artifact is verified from disk before publication. Publication
uses a fixed lock file and Linux `renameat2(RENAME_NOREPLACE)` on the same
filesystem; an existing output is an error.
