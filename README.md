# AloePri Model Compiler

A local Rust compiler for Hugging Face safetensors artifacts. The compiler
supports byte-preserving `identity`, `aloepri-token` vocabulary permutation,
and experimental F32 `aloepri-keymat` hidden expansion for dense Llama checkpoints.

## Scope

The compiler discovers local `config.json` plus a single `model.safetensors` or
an HF `model.safetensors.index.json` and its shards. It validates tensor ranges
and metadata, fingerprints source bytes with BLAKE3, builds a streaming plan,
writes standard safetensors output, verifies the result from disk, and publishes
it atomically.

Operation inputs and outputs are decoupled, so a method is no longer assumed to
produce the same tensor names, shapes, dtypes or byte lengths that it consumed.
Layout, headers, offsets and shard assignment are planned from the output
descriptors. `identity` preserves bytes exactly; `aloepri-token` permutes
vocabulary rows while keeping the tensor topology; tied checkpoints keep the
missing physical `lm_head.weight` alias.

Every artifact declares a runtime contract. Identity and token outputs remain
standard Hugging Face checkpoints (`standard_hf_checkpoint = true`). A
non-standard artifact declares `standard_hf_checkpoint = false` and is reported
as requiring its own runtime: `aloepri verify` never presents it as loadable by
vanilla Transformers, and it performs structural verification only.

The compiler does not download models, implement paper Attention/FFN/RoPE
transforms or quantization, or claim cryptographic privacy or weight secrecy.
The Client Secret is an external file and must be supplied to the client-side
inference demo.

## Build and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo build --release --workspace --locked
```

The toolchain is pinned in `rust-toolchain.toml`.

## Acceptance checks

`tests/reference/identity_oracle.py` is an independent byte-level oracle. It
compares the config bytes, tensor names, dtypes, shapes and payloads of an input
and an output artifact, optionally cross-checks per-tensor BLAKE3 digests, and
with `--transformers-load` reads every output shard through `safetensors` and
loads the artifact with a locally-only Transformers instance, asserting the
loading report has no unexpected, mismatched or error entries.

```bash
HF_HUB_OFFLINE=1 python tests/reference/identity_oracle.py \
  --source INPUT --output OUTPUT --transformers-load
```

`scripts/large_model_smoke.py` generates synthetic artifacts whose payload far
exceeds `--memory-limit`, runs the release binary as a separate process and
reports each process peak RSS; the assertion is that peak RSS does not grow with
the model payload.

```bash
python scripts/large_model_smoke.py --binary target/release/aloepri \
  --work-dir ./smoke --memory-limit 32MiB --sizes 128MiB,512MiB
```

## CLI

```bash
aloepri inspect MODEL
aloepri plan MODEL --identity --memory-limit 32MiB --max-shard-size 32MiB
aloepri transform MODEL --output OUTPUT --identity \
  --memory-limit 32MiB --max-shard-size 32MiB [--resume]
aloepri transform MODEL --output OUTPUT --method aloepri-token \
  --secret-output ./client-secret.json \
  --memory-limit 32MiB --max-shard-size 32MiB [--resume]
aloepri verify MODEL
```

`--method identity` is equivalent to `--identity`; the token method requires a
new external `--secret-output` path. A resumed token transform reads and
validates the existing Secret and never generates a replacement.

`aloepri verify` prints a layered report: structure, manifest, versioned plan
verification, the declared `standard_hf_checkpoint` flag, `runtime_required`,
and `semantic_verification: not_run`. A non-standard artifact verifies
structurally and is reported as requiring its runtime; it is never described as
vanilla-HF loadable. An artifact with no manifest is container-only, with
runtime compatibility reported as unknown.

`--memory-limit` applies to the compiler-managed metadata and I/O working set;
it is not an operating-system RSS cap. A tensor is never split across output
shards. A tensor larger than `--max-shard-size` is placed in an oversize shard
and copied in chunks.

Transform writes a versioned manifest and checkpoint into a private staging
directory, verifies the candidate from disk, and publishes it with a Linux
`renameat2(RENAME_NOREPLACE)` operation. Existing output is never overwritten.
Resume refuses any checkpoint whose source fingerprint, method, runtime
contract, layout, plan hash or secret does not match the current plan, before
staging is touched. Identity/token/diagnostic use schema v3; KeyMat uses v4.
Legacy v1/v2 artifacts remain readable for structural verification only and are
never resumed or upgraded.

## Token permutation demo

Install the pinned demo dependencies from `demo/requirements.txt`, prepare a
local Hugging Face SmolLM2-135M checkout, run the Rust transform, then execute:

```bash
HF_HUB_OFFLINE=1 python demo/token_permutation_demo.py \
  --source ./SmolLM2-135M \
  --obfuscated ./SmolLM2-135M-aloepri \
  --secret ./client-secret.json \
  --prompt 'Once upon a time' \
  --max-new-tokens 16
```

The demo independently checks the Secret commitment, source/output tensor
relationships, tied embedding loading, teacher-forced logits, and restored
greedy token IDs before printing `Baseline`, `AloePri`, and `Equivalent: true`.
See `docs/architecture.md`, `docs/artifact-format.md`, and
`docs/method-contract.md` for the stable contracts.

## Experimental KeyMat correctness runtime

`aloepri-keymat/0.1` implements Algorithm 1 in F64 and transforms **F32-only**
dense Llama weights into `D=d+2h`. `h` must be positive and even; lambda must be
finite and nonnegative. F16/BF16/quantized weights, biases and unknown tensor
variants are rejected. The logical config stays unchanged. In particular, a
missing tied head is materialized independently: embedding `EP` and head
`EQ^T` cannot share physical storage.

`demo/aloepri_runtime.py` is an **exact_covariant diagnostic correctness oracle**,
not the final AloePri paper runtime or a privacy guarantee. Only RMSNorm
explicitly uses `z -> zQ -> RMSNorm_d -> P`; residuals remain D-dimensional,
Attention/RoPE/SwiGLU remain logical, and KV cache is standard projection-space.
Loading requires transformed artifact + external Secret; never load this
artifact with `AutoModelForCausalLM.from_pretrained()`.

Install `demo/requirements.txt`. Prepare the fixed local model revision in
`demo/hidden_expansion_fixture.json` as a separate F32 source (this is not
compiler dtype conversion), then run:

```bash
HF_HUB_OFFLINE=1 python demo/prepare_f32_fixture.py \
  --source LOCAL_SMOLLM2_SNAPSHOT --output SOURCE_F32 --report export.json
mkdir PRIVATE_DIR
aloepri transform SOURCE_F32 --output OUTPUT_KEYMAT \
  --method aloepri-keymat --expansion-size 32 --keymat-lambda 0.3 \
  --keymat-fixture-seed 20260930 --secret-output PRIVATE_DIR/keymat-secret.json \
  --memory-limit 256MiB --max-shard-size 32MiB
aloepri verify OUTPUT_KEYMAT
HF_HUB_OFFLINE=1 python demo/hidden_expansion_demo.py \
  --source SOURCE_F32 --artifact OUTPUT_KEYMAT \
  --secret PRIVATE_DIR/keymat-secret.json \
  --fixture demo/hidden_expansion_fixture.json --report gates.json
HF_HUB_OFFLINE=1 python demo/aloepri_runtime.py \
  --artifact OUTPUT_KEYMAT --secret PRIVATE_DIR/keymat-secret.json \
  --token-ids 6403 1980 253 655 --max-new-tokens 0
python scripts/keymat_memory_smoke.py \
  --binary target/release/aloepri --report memory.json
python -m unittest discover -s tests/python -p 'test_*.py'
```

The fixture seed is for reproducible testing only; omit it for OS-random private
production material. Private JSON and `key-material.bin` must both live outside
source/output/staging. Neither is included in the published artifact. Reuse the
same bundle and parameters with `--resume`; incomplete bundles are never
repaired or overwritten automatically.

The harness freezes finite + **max_abs<=1e-5**, with no relative-tolerance escape.
It compares every specified layer node, full/prefill/cached decode logits and
independent greedy sequences. Failed upstream gates leave downstream gates
`not_run`; a G2/G3 failure is not evidence that Phase 3B is complete. Standalone
runtime inference is diagnostic, not an equivalence claim. Do not alter fixture
parameters or thresholds to make a failed report pass.

### Bounded numerical diagnostics — still blocked

The original native-runtime failure at commit `8fdd8c9` is retained. The fixed
three-sequence fixture first fails at `layer.0.ffn_output` (max errors
7.1695e-5 / 8.3798e-5 / 6.9545e-5). G1/structure passing does not repair G2.

`demo/keymat_numerical_diagnostics.py` stops after the first block and checks the
complete gate/up/down weights, SwiGLU hooks and same-input counterfactuals. The
F64 SOURCE/theoretical-weight references are **diagnostic_only**, never runtime
substitutes or formal gate evidence. Error vectors stay in a private 0600 file;
the public JSON contains only summaries, identities and code/environment hashes.

```bash
HF_HUB_OFFLINE=1 python demo/keymat_numerical_diagnostics.py \
  --source SOURCE_F32 --artifact OUTPUT_KEYMAT \
  --secret PRIVATE_DIR/keymat-secret.json \
  --fixture demo/hidden_expansion_fixture.json --failure-report ORIGINAL_GATES_JSON \
  --report numerical-diagnostics.json \
  --private-error-vectors PRIVATE_DIR/diagnostic-errors.pt
HF_HUB_OFFLINE=1 python demo/keymat_numerical_diagnostics.py \
  --source SOURCE_F32 --artifact OUTPUT_KEYMAT \
  --secret PRIVATE_DIR/keymat-secret.json \
  --fixture demo/hidden_expansion_fixture.json --failure-report ORIGINAL_GATES_JSON \
  --diagnostic-input numerical-diagnostics.json --evaluate-candidate \
  --report candidate-local.json
```

The single approved candidate, `F64AccumLinear`, keeps parameters/state, inputs
and outputs F32 and uses only temporary F64 calculation. Output rows are
chunked at a fixed 1024, including head-sized test cases; it never stores a
persistent F64 shadow model. Reports count temporary tensors and time separately
from cumulative process RSS (which includes baseline/runtime/oracles).

Native, down-only, and gate/up/down controls use the same artifact and fixture.
The all-FFN candidate still fails locally: FFN max errors
6.5052e-5 / 8.0413e-5 / 5.4939e-5 exceed 1e-5. Its largest per-Linear F64 temporary
working set is 8073216 bytes (8134656 including F32 outputs/scratch), not a
compiler memory-limit claim. This is not a production optimization.

**Default loader/normal inference remains native F32 Linear.** The candidate is
not installed across the model: its prerequisite local gate failed, so candidate
full-model G2/G3/G4 are `not_run`. No source, Secret, artifact/schema, norm,
fixture or tolerance was changed. These results neither prove all F32 strategies
impossible nor authorize relaxing the gate; further changes require a new plan.

### Versioned balanced-null construction — still blocked

The single fixed `algorithm1-balanced-null-v2` trial changes only C/N nullspace
coefficient std from1 to d^-1/2, retaining raw substreams/bases and the same
source, h/lambda/master seed and input fixture. The original v1 defaults and
failed artifacts are retained. New P/Q/Secret/plan/artifact identities are
intentional; do not reuse or overwrite the old private directory or staging.
Both versions use Secret schema1, KeyMat artifact schema4, method0.1 and native
F32 runtime. This is a sampling-distribution engineering variant, not a claim
of unchanged reference distribution or paper privacy guarantees.

```bash
mkdir PRIVATE_BALANCED_DIR
aloepri transform SOURCE_F32 --output OUTPUT_BALANCED \
  --method aloepri-keymat --keymat-algorithm algorithm1-balanced-null-v2 \
  --expansion-size 32 --keymat-lambda 0.3 --keymat-fixture-seed 20260930 \
  --secret-output PRIVATE_BALANCED_DIR/secret.json \
  --memory-limit 256MiB --max-shard-size 32MiB
aloepri verify OUTPUT_BALANCED
HF_HUB_OFFLINE=1 python demo/keymat_numerical_diagnostics.py \
  --source SOURCE_F32 --artifact OUTPUT_BALANCED \
  --secret PRIVATE_BALANCED_DIR/secret.json \
  --fixture demo/hidden_expansion_fixture.json --failure-report ORIGINAL_GATES_JSON \
  --construction-trial --control-secret PRIVATE_DIR/keymat-secret.json \
  --control-diagnostics numerical-diagnostics.json --report balanced-first-block.json \
  --private-error-vectors PRIVATE_BALANCED_DIR/diagnostic-errors.pt
```

The trial has a separate strict identity check; it is not the v1 same-Secret
replay and does not weaken its validation. Only after all original first-block
nodes pass for all three fixed sequences may the formal full-model harness run.
Fresh omitted algorithm still means v1; omitted resume algorithm means the
stored Secret's actual version. Explicit mismatch and v1↔v2 interrupted/published
resume fail before touching staging/locks, including equal-layout cases.

Actual v2 P/Q spectral norms were about1.61916/1.68904 (product2.73482),
PQ max about7.1e-15, and complete first-FFN F32 encoding oracle error0/ULP0.
The first two sequences passed locally, but prompt2 still failed
`layer.0.ffn_output` at1.1917904516e-5 and block output at1.3106583758e-5.
The once-only trial stopped: full-model v2 G2/G3/G4 are not_run. No scale/seed
search, F64 model/state/residual or failed accumulation candidate was installed.
Phase3B remains incomplete; lower key norms and structure/unit passes do not
replace the unchanged absolute1e-5 acceptance gate.

### Explicit same-v2 all-boundary profile — still blocked

`load_runtime(artifact, secret, *, linear_profile=...)` defaults to
`f32-state-native-linear-v1`. The only new opt-in is
`balanced-v2-all-boundary-f64-acc-v1`, restricted to existing balanced-v2
artifacts. It reuses temporary F64 Linear arithmetic with fixed1024 row chunks
and F32 Parameter/state/input/output/residual; norm, Attention internals, RoPE,
SwiGLU and cache remain unchanged. There is no artifact regeneration, new key
family, persistent F64 shadow or silent fallback.

The shared installer checks all `7*layers+1` roles, exact geometry, CPU/F32 and
bias=None, preserving original Parameter objects/state keys/head independence.
It prepares every replacement before mutation and rejects unknown/v1/mixed/
partial/misreported profiles. A repeated same-profile install validates before
returning. Reports derive profile from actual module inventory, not the flag.

```bash
HF_HUB_OFFLINE=1 python demo/keymat_numerical_diagnostics.py \
  --execution-profile-trial --linear-profile balanced-v2-all-boundary-f64-acc-v1 \
  --source SOURCE_F32 --artifact EXISTING_V2_ARTIFACT \
  --secret EXISTING_V2_PRIVATE/secret.json --fixture demo/hidden_expansion_fixture.json \
  --reference-trial balanced-first-block.json --report profile-first-block.json

# Run only if the above necessary local gate passes for every sequence/node:
HF_HUB_OFFLINE=1 python demo/hidden_expansion_demo.py \
  --linear-profile balanced-v2-all-boundary-f64-acc-v1 \
  --source SOURCE_F32 --artifact EXISTING_V2_ARTIFACT \
  --secret EXISTING_V2_PRIVATE/secret.json --fixture demo/hidden_expansion_fixture.json \
  --report profile-gates.json
```

The native control replayed every original v2 first-block node. All211 roles
were installed in the candidate, but only layer0's seven roles executed; head
and later-layer installation is not proof of numerical equivalence. Prompt0
passed locally (max7.28893e-6), while prompt1/prompt2 failed FFN at
1.0797817197e-5 / 1.3192825580e-5 and block output at
1.0934948847e-5 / 1.4176833531e-5. The fixed profile stopped: full-model candidate
G2/G3/G4 are not_run. No search or threshold/baseline change followed.

Measured per-Linear peak temporary F64 tensors were8073216 bytes
(8134656 including F32 output/scratch), largest at layer0 down. CPU/role calls
and cumulative process RSS are separately reported; these are not Compiler
memory-limit or independent summed peaks. The source-free subprocess validated
all211 installed roles, F32/D640 first-block outputs and first-layer projection
cache [1,3,4,64], without SOURCE or Llama.from_pretrained. It did not execute
head/later layers or generation. This remains a diagnostic experiment, not
production/paper support or Phase3B completion.

### Bounded floating semantics audit — evidence only

No new repair profile is selected. `demo/keymat_float_semantics_audit.py` runs
only the fixed three `prompt+continuation` layer0 prefixes and preserves normal
runtime/installer/harness/Compiler/Secret/fixture behavior:

- A: original native CPU/F32 SOURCE, always the only acceptance target.
- B: independent SOURCE first-block copy, seven temporary-F64 Linear calls,
  with original F32 embedding/norm/attention/nonlinearity/residual boundaries.
- C: actual same-v2 all-boundary runtime, strictly replayed from latest evidence.
- D: independent first-block reference with the same actual artifact embedding,
  keys/norm/config/F32 boundaries as C; seven ideal private F64 coefficients
  from SOURCE/P/Q. D is SOURCE-dependent and not a legal F32 artifact/profile.

The references reuse the pinned HF Llama prefix/DecoderLayer/MLP forward and
stop after layer0. Only two first-block copies and seven ideal matrices exist;
no whole-model double shadow is created. B/D cannot be formal baselines or
normal profile models. Original A/C state, Parameter/module/buffer identities
are checked before/after reference execution; their weights are not replaced.

```bash
mkdir PRIVATE_AUDIT_DIR
HF_HUB_OFFLINE=1 python demo/keymat_float_semantics_audit.py \
  --source SOURCE_F32 --artifact EXISTING_V2_ARTIFACT \
  --secret EXISTING_V2_PRIVATE/secret.json \
  --fixture demo/hidden_expansion_fixture.json \
  --reference-profile profile-first-block.json \
  --private-error-vectors PRIVATE_AUDIT_DIR/errors.pt \
  --report float-semantics-audit.json
```

Each D/C residual is separately recovered by double@Q before subtraction.
Elementwise C-A=(B-A)+(D-B)+(C-D) reconstruction was0; signed terms are reported
at C's actual worst coordinate, not by adding unrelated scalar maxima.
Prompt1 C's worst FFN coordinate [0,1,308] decomposed into about
+9.53674e-6 +7.63195e-7 +4.97879e-7 = +1.07978e-5.
Prompt2 [0,0,247] decomposed into
-5.72205e-6 -7.41227e-6 -5.85119e-8 = -1.31928e-5.
Ideal D still exceeded1e-5 against A for both failing prompts; logical B itself
also exceeded1e-5 at prompt1 block and prompt2 FFN/block. Thus more accurate
coefficients alone are not a verified repair, and more accurate real arithmetic
is not automatically fidelity to the original native F32 execution.

Exit0 reports audit_completed=true/diagnostic_only=true/ac10_satisfied=false;
formal G2/G3/G4 remain not_run. The 0600/exclusive private vectors and ideal
coefficients are not uploaded. This audit ends at supported/unknown evidence
for a future plan: no baseline/threshold/dtype/profile/keys/compensation change,
no head/logits/cache/generation, and no claim that every F32 strategy is impossible.

### Native-input FFN-only necessary precheck — no new supported profile

The one fixed precheck keeps the same v2 native Attention prefix and uses its
actual D/F32 z_norm and post-attention residual z1. Original MLP recomputation
and F32 residual addition bitmatch the native trace. A separate clone of the
actual artifact MLP (F32 parameters/activation) wraps only gate/up/down with the
existing temporary-F64 arithmetic. It does not read SOURCE weights or baseline
hidden states as candidate data, nor install wrappers in the actual model.
The diagnostic candidate computes ffn_D32=cloned_mlp(z_norm_D32) and
block_D32=z1_D32+ffn_D32. All eight original required nodes/three full fixed
sequences use the original A/Q comparison; first-six prefix nodes are unchanged.

```bash
HF_HUB_OFFLINE=1 python demo/keymat_ffn_same_input_trial.py \
  --source SOURCE_F32 --artifact EXISTING_V2_ARTIFACT \
  --secret EXISTING_V2_PRIVATE/secret.json --fixture demo/hidden_expansion_fixture.json \
  --reference-audit float-semantics-audit.json --report ffn-same-input-trial.json
```

Original models, Parameter/state/buffer/profile metadata and cloned weight bits
were unchanged. Every original MLP/recomputed residual matched native trace
bits. Prompt0/prompt1 candidate block max errors were7.1053110062e-6 and
8.4815633912e-6; prompt2 still failed FFN/block at1.1115812466e-5 /
1.2282493927e-5, both at [0,0,247]. The fixed precheck exited1 and stopped.
Thus the proposed `balanced-v2-native-attn-head-f64-ffn-v1` was **not** added to
SUPPORTED_LINEAR_PROFILES/normal loader, no per-role production implementation
or full-model G2/G3/G4 was run, and no further scope/chunk/keys/dtype/baseline
search was attempted. The precheck is diagnostic_only, not a SOURCE-dependent
runtime or a formal equivalence claim. Temporary FFN F64 peak was8073216 bytes
(working8134656); process RSS includes both models/oracles/allocator and is not
the Compiler memory-limit. Phase3B remains incomplete.
