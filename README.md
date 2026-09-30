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
