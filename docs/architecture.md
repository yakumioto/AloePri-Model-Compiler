# Architecture

The workspace has six crates with one-way dependencies:

```text
aloepri-cli -> aloepri-artifact -> aloepri-core
            -> aloepri-architecture -> aloepri-core
            -> aloepri-transform -> aloepri-core
            -> aloepri-secret -> aloepri-core
```

`aloepri-core` owns units, errors, model and adapter contracts, plans, the
managed memory budget, and the generic streaming executor. `aloepri-artifact`
owns HF discovery, safetensors headers, bounded readers, fingerprints, output
layout, writer, checkpoint, manifest, atomic publishing, and verification.
`aloepri-architecture` contains the Llama dense adapter. `aloepri-transform`
owns Identity copying and token row-permutation execution without knowing
architecture-specific tensor discovery. `aloepri-secret` owns Client Secret v1
schema, OS-random permutation generation, commitment validation, and external
file persistence.

## Compiler façade and injection contract

`aloepri_core::Compiler<B, R, E>` is constructed from three injected
implementations and exposes:

```rust
Compiler::new(backend, registry, executor)
    .inspect(&Path)                       -> InspectionReport
    .plan(&Path, &TransformConfig)        -> TransformPlan
    .transform(&TransformRequest)         -> TransformReport
    .verify(&Path)                        -> VerificationOutcome
```

The compiler owns ordering, staging, checkpointing, publication, and disk
verification. The architecture adapter creates a plan whose operations are
marked `Copy` or `TokenPermutation` with an abstract vocabulary role. The token
executor receives an already validated inverse permutation and wraps the
existing `TensorReader`; it never materializes a whole tensor or writes a
second pipeline. The artifact writer continues to validate shape, dtype, and
length and streams each transformed tensor through the same output layout.

## Tied vocabulary weights

The adapter validates the complete dense Llama schema. For tied models where
only `model.embed_tokens.weight` is physical, the plan transforms that tensor
only and preserves the logical `lm_head.weight` alias. If both tied tensors are
physical, their metadata and bytes must match before both receive the same row
mapping. Token mode rejects unsupported output bias and non-floating
vocabulary dtypes; Identity retains its original passthrough behavior.

## Secret and publication lifecycle

The CLI opens the source and computes its fingerprint before generating a
Client Secret. A fresh token run exclusively creates and syncs the selected
external Secret file, then invokes the same `Compiler::transform` pipeline.
Resume reads and validates that file and never regenerates it. The compiler
stores only the public `secret_id` in plan, checkpoint, and manifest; the
permutations and nonce remain in the external Secret and process memory.
Secret persistence and model publication are separate paths: the guarantee is
that a visible new model cannot appear before the requested Secret is synced,
not a cross-path transaction.

## Resume detection and memory

Interruption leaves shard files in staging even though the index is written only
after every operation, so resume is chosen from what staging holds. The
compiler validates the checkpoint contract—including method and `secret_id`—
before creating the staging directory, taking the lock, or opening a writer.
Token plan metadata accounts for the vocabulary mapping in its managed memory
estimate; row reads and writes remain bounded by the existing budget.

The memory limit covers compiler-managed metadata, I/O buffers, headers, and
scratch state. Runtime and allocator overhead and the OS page cache are
outside that number; the project does not claim a hard RSS limit.
