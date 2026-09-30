# Method contract

The product CLI exposes three method contracts:

```json
{"id":"identity","version":"0.1"}
{"id":"aloepri-token","version":"0.1"}
{"id":"aloepri-keymat","version":"0.1"}
```

A diagnostic contract, `{"id":"expand-test","version":"0.1"}`, exists only so the
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

## AloePri KeyMat (experimental Phase 3B)

`aloepri-keymat/0.1` uses schema v4 and F32-only dense Llama, without bias or
quantization. Algorithm `algorithm1-v1` constructs in F64:

```text
B = U + lambda V; E = E1 E2; F = F1 F2
CF = 0; EN = 0; Z Z^T = I_D
P = [B C E] Z                  # d x D
Q = Z^T [B^-1; F; N]           # D x d
PQ = BB^-1 + CF + EN = I_d     # not QP=I_D
```

U/Z use QR with R-diagonal sign correction, Gaussian substreams are fixed by
`chacha20-rand0.9-normal0.5-v1`, and complete right nullspaces use bounded SVD
with cutoff `max(1e-10,1e-10*sigma_max)`. Wide matrices are zero-row padded before
SVD so the thin decomposition does not discard the nullspace. Singular B,
nonconvergence, non-finite keys or PQ error above 1e-5 fail closed without
resampling. Diagnostics include max/mean PQ error, Frobenius/spectral norms,
P condition estimate and B condition estimate.

With PyTorch row-vector weights: embedding uses EP; head/q/k/v/gate/up use
WQ^T; o/down use P^TW. Norm weights remain logical `[d]`. A tied missing head
has its own physical `EQ^T` producer, never an alias to EP. Runtime id/version
are `aloepri/1`, with `norm_mode=exact_covariant`,
`kv_cache_format=standard_projection_v1`, and `standard_hf_checkpoint=false`.
Only physical hidden changes; config, heads/head_dim, RoPE and intermediate
size remain logical. No Algorithm 2, paper RMSNorm kappa, FFN intermediate
permutation/scaling, noise or token composition is implemented.

The diagnostic norm is `RMSNorm_d(zQ)P`, with F64 mappings and original F32
Llama norm. It is not the paper's norm or a security claim. The standalone
loader accepts only transformed artifact and external Secret, validates their
contracts/digests and never calls `from_pretrained` on transformed weights.
G1–G4 equivalence is a separate fail-closed harness: absolute 1e-5, all specified
nodes and full/prefill/decode logits; generation runs only after prior gates
pass. Structural verification never supplies these numerical results.

### Internal precision experiment (not the default runtime)

`F64AccumLinear` is the bounded diagnostic candidate
`f32-state-f64-linear-acc-v1`: require F32 input/weight and no bias, perform
`F.linear(input.double(), weight_chunk.double())`, cast each result back to
F32 and reject non-finite output. A fixed 1024-output-row chunk bounds temporary
F64 weights; no F64 parameters, persistent buffers, residuals or shadow model
are introduced. Strict state_dict keys and physical tensor bits are preserved.

Only the diagnostic first-layer down-only and gate/up/down evaluations use this
class. All three fixed sequences must pass every original first-block node
before considering a full-model integration. They did not pass, so the loader
still installs ordinary F32 Linear and formal candidate G2/G3/G4 remain not_run.
This experiment adds no artifact/runtime-contract field or schema version;
SOURCE oracle use remains isolated from transformed-only inference.

Error attribution retains elementwise telescoping vectors, source-baseline F32
rounding, storage/output rounding and same-input native arithmetic differences.
Maxima of separate components are not additive contributions or lower bounds.
Higher arithmetic precision is not automatically equivalence to the original
F32 baseline; neither unchanged encoding nor improved local errors satisfy AC10.

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
