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

The compiler creates a semantic `PlanDraft`, then combines it with source
fingerprint and artifact output layout to produce a validated `TransformPlan`.
The executor receives only the final plan and injected artifact reader/writer
interfaces. It runs one operation at a time and never materializes a whole
tensor.

The memory limit is a contract for compiler-managed metadata, I/O buffers,
headers, and scratch state. Runtime and allocator overhead and the OS page
cache are outside that number; the project does not claim a hard RSS limit.
