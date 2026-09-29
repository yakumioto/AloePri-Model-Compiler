# AloePri Model Compiler

A local Rust compiler for Hugging Face safetensors artifacts. The compiler
supports the byte-preserving `identity` method and the first weight-changing
`aloepri-token` vocabulary permutation for dense Llama checkpoints.

## Scope

The compiler discovers local `config.json` plus a single `model.safetensors` or
an HF `model.safetensors.index.json` and its shards. It validates tensor ranges
and metadata, fingerprints source bytes with BLAKE3, builds a streaming plan,
writes standard safetensors output, verifies the result from disk, and publishes
it atomically. `aloepri-token` changes vocabulary rows while preserving the
physical tensor inventory; tied checkpoints keep the missing physical
`lm_head.weight` alias.

The compiler does not download models, implement Attention/FFN/RoPE transforms,
quantization, hidden-state obfuscation, or claim cryptographic weight secrecy.
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

`--memory-limit` applies to the compiler-managed metadata and I/O working set;
it is not an operating-system RSS cap. A tensor is never split across output
shards. A tensor larger than `--max-shard-size` is placed in an oversize shard
and copied in chunks.

Transform writes a versioned manifest and checkpoint into a private staging
directory, verifies the candidate from disk, and publishes it with a Linux
`renameat2(RENAME_NOREPLACE)` operation. Existing output is never overwritten.

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
