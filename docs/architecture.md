# Architecture

The workspace has six crates with one-way dependencies:

```text
aloepri-cli -> aloepri-artifact -> aloepri-core
            -> aloepri-architecture -> aloepri-core
            -> aloepri-transform -> aloepri-core
            -> aloepri-secret -> aloepri-core
```

`aloepri-core` owns units, errors, model and adapter contracts, plans, the
managed memory budget, the sink/executor boundary, and the generic streaming
executor. `aloepri-artifact` owns HF discovery, safetensors headers, bounded
readers, fingerprints, output layout, writer, checkpoint, manifest, atomic
publishing, and verification. `aloepri-architecture` contains the Llama dense
adapter. `aloepri-transform` owns Identity copying and token row-permutation
execution without knowing architecture-specific tensor discovery.
`aloepri-secret` owns Client Secret v1 schema, OS-random permutation
generation, commitment validation, and external file persistence.

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
verification. The pipeline is deliberately output-driven:

1. open the source, fingerprint it, and let the adapter produce a `PlanDraft`
   (runtime contract plus one operation per output);
2. derive the output inventory from the operations and plan the physical layout
   from those **output** descriptors, before any file exists;
3. validate the layout, then construct and validate the version-3 plan;
4. execute each operation into a compiler-owned output sink, syncing and
   checkpointing only after the sink reports a complete write.

## Input/output decoupling

`Operation` no longer assumes one identical input and output. It carries
`inputs: Vec<OperationInput>` (N source descriptors) and exactly one
`output: OperationOutput`, where the output descriptor has no source location.
The source inventory and output inventory are separate; they need not share
names, shapes, dtypes or byte lengths. Generic structural validation only
requires that every input resolves to a real source, every output has exactly
one producer, and each descriptor is internally consistent; method-specific
rules (name/shape preservation for identity and token, padding geometry for the
diagnostic method) live in the method validator.

Two copies of a source are still rejected, as are duplicate or unproduced
outputs, unknown inputs, forged input metadata, forward/self/unknown
dependencies, and descriptor/layout mismatches.

## Writer, sinks and bounded memory

Executors never create files, plan shards, write manifests or report
completion. The compiler calls `TensorWriter::begin_tensor` to obtain a bounded
`TensorSink`, hands the executor only a writable borrow, and calls
`TensorSink::finish` itself. A sink counts bytes actually written, hashes
exactly those bytes, rejects an over-long write as a whole before any byte
reaches the file, and refuses to finish short. Completion is therefore derived
from the sink, not from an executor's self-report.

Readers and writers stream in chunks bounded by the remaining memory budget and
capped at 4 MiB, so a tensor—or a single row—larger than the working buffer
still transforms without being materialized. No production tensor path calls
`read_to_end`.

The memory limit covers a layered estimate: plan metadata, input buffer, output
buffer, transform scratch, and method state. It is an estimate of the managed
working set, not an operating-system RSS cap; runtime and allocator overhead and
the OS page cache sit outside it.

## Runtime contract

Every plan carries a `RuntimeContract` describing whether the artifact is a
`standard_hf_checkpoint`, separated logical and physical dimensions, and an
optional expansion size. Identity and token declare the `huggingface` runtime
with `standard = true` and `physical == logical`. A shape-changing method
declares a non-standard runtime; `config.json` continues to describe the
logical model. The runtime contract participates in the plan hash, the
manifest, the checkpoint, and resume validation.

## Tied vocabulary weights

The adapter validates the complete dense Llama schema. For tied models where
only `model.embed_tokens.weight` is physical, the plan transforms that tensor
only and preserves the logical `lm_head.weight` alias. If both tied tensors are
physical, their metadata and bytes must match before both receive the same row
mapping. Token mode rejects unsupported output bias and non-floating vocabulary
dtypes; Identity retains its original passthrough behavior.

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

## Resume detection

Interruption leaves shard files in staging even though the index is written only
after every operation, so resume is chosen from what staging holds. The
compiler validates the full checkpoint contract—including method, runtime
contract, layout and `secret_id`—before creating the staging directory, taking
the lock, or opening a writer.
