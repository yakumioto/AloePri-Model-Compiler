# Method contract

The only v0.1 method is:

```json
{"id":"identity","version":"0.1"}
```

Identity accepts only `OutputDType::Preserve` and one worker. Every physical
input tensor maps to exactly one operation and one output tensor. The operation
reads source bytes through a bounded `TensorReader`, writes them through the
injected writer, and updates a BLAKE3 digest. It does not decode, cast,
reorder, transpose, or numerically modify bytes.

A checkpoint is schema v1. It records source fingerprint, plan hash, output
layout hash, method contract, explicit absence of `secret_key_id`, and the
ordered completed-operation prefix with expected digests. Resume recomputes the
source and plan, validates output headers and lengths, re-hashes completed
ranges, and rejects a mismatch. Each completed operation is synced before its
checkpoint is atomically replaced.

The candidate artifact is verified from disk before publication. Publication
uses a fixed lock file and Linux `renameat2(RENAME_NOREPLACE)` on the same
filesystem; an existing output is an error. A failed run leaves its owned
staging directory for explicit `--resume` and never exposes it as the final
artifact. Because the index and the single-file name appear only after every
operation, resume is detected from the planned output layout: a staging
directory holding any planned shard file is resumed, so a sharded run that was
interrupted mid-execution is recoverable rather than requiring manual cleanup.
