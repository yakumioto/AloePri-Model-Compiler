# Method contract

The compiler supports two method contracts:

```json
{"id":"identity","version":"0.1"}
{"id":"aloepri-token","version":"0.1"}
```

## Identity

Identity accepts only `OutputDType::Preserve` and one worker. Every physical
input tensor maps to exactly one copy operation and one output tensor. The
operation reads source bytes through a bounded `TensorReader`, writes them
through the injected writer, and updates a BLAKE3 digest. It does not decode,
cast, reorder, transpose, or numerically modify bytes. Identity manifests and
checkpoints remain schema v1.

## AloePri token

`aloepri-token` supports the tested dense Llama schema and F32/F16/BF16
vocabulary matrices. It creates an external Client Secret v1 and marks the
`model.embed_tokens.weight` operation, plus a physical `lm_head.weight` when
present, as row permutations. For `tau: original -> obfuscated`:

```text
E'[tau(i)] = E[i]
H'[tau(i)] = H[i]
```

A tied checkpoint with no physical `lm_head.weight` keeps that inventory; the
Hugging Face loader restores the logical alias. A tied checkpoint containing
both tensors must contain identical bytes before transformation. Unsupported
biases, quantized vocabulary weights, and other architectures fail closed.
Token manifests and checkpoints are schema v2 and carry only the public
`secret_id`, source fingerprint, method, and plan hash. The permutation,
inverse, nonce, and any seed never enter the model directory or compiler
checkpoint.

The Client Secret contains `version`, method, `secret_id`, `vocab_size`,
`source_fingerprint`, `binding_nonce`, `token_permutation`, and
`inverse_token_permutation`. `secret_id` is a BLAKE3 commitment over the
versioned method, source fingerprint, vocabulary size, nonce, and both
permutations. Loading validates the full bijection, inverse relationship,
source binding, and commitment.

## Checkpoint and resume

A checkpoint records source fingerprint, plan hash, output layout hash, method,
public secret binding, and the ordered completed-operation prefix. Resume
recomputes the plan and validates the contract before staging, locking, or
writing. A token resume reads the existing Client Secret and never generates or
overwrites one. The specified Secret file is persisted and synced before the
compiler can publish a new model; failure may leave that external file and
private staging for an explicit resume. There is no cross-path transaction.

The candidate artifact is verified from disk before publication. Publication
uses a fixed lock file and Linux `renameat2(RENAME_NOREPLACE)` on the same
filesystem; an existing output is an error. The client-side Python demo is the
separate inference/equivalence check; ordinary artifact verification does not
claim mathematical inference equivalence.
