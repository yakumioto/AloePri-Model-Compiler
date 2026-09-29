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
`aloepri-architecture` contains the Llama dense adapter only. The adapter
validates configuration and tensor schema but never performs numerical work.
`aloepri-transform` owns Identity execution and knows neither model names nor
architecture-specific naming rules. `aloepri-secret` records the explicit
v0.1 boundary and never emits a key or seed.

## Compiler façade and injection contract

`aloepri-core::Compiler<B, R, E>` is constructed from three injected
implementations and exposes the four v0.1 APIs:

```rust
Compiler::new(backend, registry, executor)
    .inspect(&Path)                       -> InspectionReport
    .plan(&Path, &TransformConfig)        -> TransformPlan
    .transform(&TransformRequest)         -> TransformReport
    .verify(&Path)                        -> VerificationOutcome
```

- `B: ArtifactBackend` carries every artifact-format capability: `open`,
  `plan_output`, `staging_has_output`, `create_writer`, `resume_writer`,
  `prepare_staging`, `load_checkpoint`, `store_checkpoint`, `finalize`,
  `output_matches_plan`, `publish`, `lock_output`, `verify`. `HfBackend` in
  `aloepri-artifact` is the v0.1 implementation. Keeping these behind the trait
  is what prevents a `core <-> artifact` dependency cycle.
- `R: ArchitectureRegistry` detects and selects the adapter; the Llama dense
  adapter is the only registered one.
- `E: TransformExecutor` performs one operation. `StreamingExecutor` in core is
  method-agnostic; `IdentityExecutor` in `aloepri-transform` enforces the
  identity contract and delegates the mechanical copy.

The compiler orders operations, decides where to stage, persists checkpoints,
publishes atomically and validates resume state. The executor receives only the
final plan and injected reader/writer interfaces, runs one operation at a time
and never materializes a whole tensor.

The compiler creates a semantic `PlanDraft`, then combines it with source
fingerprint and artifact output layout to produce a validated `TransformPlan`.

## Resume detection

Interruption leaves the planned shard files in staging even though the index is
written only after every operation, so the resume path is chosen from the
output layout (`ArtifactBackend::staging_has_output`) rather than from a
single-file name. A resumed run re-validates every existing shard header and
length, re-hashes the checkpointed prefix and refuses a mismatch.

## Memory limit

The memory limit is a contract for compiler-managed metadata, I/O buffers,
headers, and scratch state. Runtime and allocator overhead and the OS page
cache are outside that number; the project does not claim a hard RSS limit.
`scripts/large_model_smoke.py` measures the Rust process peak RSS across
payload sizes to show it does not scale with the model.
