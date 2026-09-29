# AloePri Model Compiler

A local Rust compiler for identity-preserving Hugging Face safetensors artifacts.

## v0.1 scope

v0.1 discovers local `config.json` plus a single `model.safetensors` or an HF
`model.safetensors.index.json` and its shards. It validates tensor ranges and
metadata, fingerprints source bytes with BLAKE3, builds a deterministic identity
plan, copies payloads in bounded chunks, writes standard safetensors output, and
verifies the result from disk. The first architecture adapter is the tested
Llama dense/GQA schema used by SmolLM-compatible checkpoints.

The compiler does not download models, run inference, or implement AloePri
math, Client Secrets, KDFs, quantization, or non-Identity transforms.

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
aloepri verify MODEL
```

`--memory-limit` applies to the compiler-managed metadata and I/O working set;
it is not an operating-system RSS cap. A tensor is never split across output
shards. A tensor larger than `--max-shard-size` is placed in an oversize shard
and copied in chunks.

Transform writes a versioned manifest and checkpoint into a private staging
directory, verifies the candidate from disk, and publishes it with a Linux
`renameat2(RENAME_NOREPLACE)` operation. Existing output is never overwritten.
See `docs/architecture.md`, `docs/artifact-format.md`, and
`docs/method-contract.md` for the stable v0.1 boundaries.
