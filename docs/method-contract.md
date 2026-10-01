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
quantization. Both supported Algorithm 1 versions construct in F64:

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

`algorithm1-v1` remains the default and preserves its original draws, bytes and
identity semantics. `algorithm1-balanced-null-v2` is a fixed engineering variant:
only Gaussian C/N nullspace coefficients change from std=1 to std=d^-1/2;
U/V/E/F/Z, domains, raw RNG samples, cutoff, h/lambda/master seed stay unchanged.
There is no free scale parameter, search or FFN-neuron scaling. Since CF=EN=0,
PQ cancellation remains; Gram PSD ordering implies nonincreasing spectral norms,
not a G2 error bound or guaranteed condition-ratio improvement. d=1 may produce
identical matrix bytes but still has distinct algorithm/Secret identities.
The changed sampling distribution does not claim unchanged paper privacy metrics.

The fixed d576/h32/lambda0.3 trial reduced the P/Q spectral product from about
761.97 to 2.73482, with PQ max about7.1e-15. It still failed the necessary
native-F32 first-block gate on prompt2: FFN=1.1917904516e-5 and
block=1.3106583758e-5. Other two prompts passing locally and improved key norms
are not Phase3B acceptance. Formal full-model G2/G3/G4 were not run; no second
scale/seed/dtype strategy was tried, and native runtime remains unchanged.

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

The historical v1 first-layer down-only/gate-up-down evaluations failed and
remain isolated. Native remains the loader default. The distinct same-v2
`balanced-v2-all-boundary-f64-acc-v1` profile is an explicit opt-in, accepted only
for balanced-v2 artifacts: install all q/k/v/o and gate/up/down roles plus head
(7L+1), using this same kernel and no other precision changes. Unknown, legacy,
mixed or partially installed profiles are rejected; actual module inventory,
geometry, CPU/F32 state and original Parameter references are checked. Same
profile installation is idempotent after full validation, with no fallback.

Installation occurs after strict weight assignment/eval and adds only ordinary
nonpersistent metadata. The artifact/Secret/transform identity and state keys
remain unchanged. All 211 real-model roles were installed, but only the first
seven executed in the bounded trial: head and later-layer installation is not
numerical evidence for those modules. Candidate prompt1/prompt2 first-FFN errors
were 1.0797817197e-5 / 1.3192825580e-5, both above 1e-5. The necessary local gate
failed; full candidate G2/G3/G4 remain not_run, with no additional profile search.
This experiment adds no artifact/runtime-contract field or schema version;
SOURCE oracle use remains isolated from transformed-only inference.

Error attribution retains elementwise telescoping vectors, source-baseline F32
rounding, storage/output rounding and same-input native arithmetic differences.
Maxima of separate components are not additive contributions or lower bounds.
Higher arithmetic precision is not automatically equivalence to the original
F32 baseline; neither unchanged encoding nor improved local errors satisfy AC10.

### Bounded floating semantics audit (references are not profiles)

`keymat_float_semantics_audit.py` fixes four layer0 paths: A is the only native
F32 SOURCE acceptance target; B keeps its F32 boundaries and changes seven
logical Linear arithmetic calls to temporary F64; C is the actual failed
same-v2 all-boundary runtime; D keeps C's artifact embedding/keys/norm and F32
boundaries but uses seven private ideal F64 coefficients derived from SOURCE.
B/D are independent first-block references using the pinned Llama forward
prefix, not supported execution profiles or legal artifacts. They have no
normal profile metadata, head/logits/generation path or full-model FP64 shadow.
Normal A/C objects, modules, Parameters, state and buffers are checked unchanged.

At every required node, separately recover C/D as double@Q, then reconstruct
C-A=(B-A)+(D-B)+(C-D) in F64. The audit reports maxima/means/L2/signed means and
all three signed terms at the *actual C* worst coordinate. Independent maxima
are never additive causes or lower bounds. Projection diagnostics are compared
directly, without Q. Private error vectors stay outside models in exclusive0600
files, not public reports.

Actual vector reconstruction error was0. Ideal D still differs from A beyond
1e-5 at prompt1/2 FFN/block, while B itself crosses1e-5 at prompt1 block and
prompt2 FFN/block. At prompt2 C's worst FFN coordinate, signed terms were about
-5.72205e-6, -7.41227e-6 and -5.85119e-8. Removing Linear weight encoding alone
therefore does not explain away the observed failure; native-arithmetic
fidelity and expanded coordinate/boundary propagation need future evidence.
No compensation/profile/family/baseline change is implemented by this audit.

Audit exit0 means checks completed only: audit_completed/diagnostic_only=true,
ac10_satisfied=false and formalG2/G3/G4=not_run. Reference comparisons within a
threshold are not AC10 evidence; A remains the original native F32 target.

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
