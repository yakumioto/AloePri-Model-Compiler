#!/usr/bin/env python3
"""Verify and run an AloePri token-permutation model locally."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, NoReturn

SIDECARS = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
SECRET_KEYS = {
    "version",
    "method",
    "secret_id",
    "vocab_size",
    "source_fingerprint",
    "binding_nonce",
    "token_permutation",
    "inverse_token_permutation",
}


class DuplicateJsonKey(ValueError):
    pass


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def parse_json(data: str | bytes, label: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=reject_duplicate_pairs)
    except DuplicateJsonKey as error:
        fail(f"duplicate JSON key in {label}: {error}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {label}: {error}")


def load_json(path: Path) -> Any:
    try:
        data = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    return parse_json(data, str(path))


def hex_bytes(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        fail(f"{label} must be 64 hexadecimal characters")
    try:
        decoded = bytes.fromhex(value)
    except ValueError:
        fail(f"{label} is not hexadecimal")
    if value != value.lower():
        fail(f"{label} must use lowercase hexadecimal")
    return decoded


def checked_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer")
    return value


def commitment(secret: dict[str, Any], source_fingerprint: bytes, nonce: bytes) -> str:
    try:
        import blake3
    except ImportError as error:
        fail(f"the blake3 Python package is required: {error}")
    hasher = blake3.blake3()
    hasher.update(b"aloepri-client-secret-v1")
    hasher.update(struct.pack("<I", secret["version"]))
    for value in (secret["method"]["id"], secret["method"]["version"]):
        encoded = value.encode("utf-8")
        hasher.update(struct.pack("<I", len(encoded)))
        hasher.update(encoded)
    hasher.update(source_fingerprint)
    hasher.update(struct.pack("<Q", secret["vocab_size"]))
    hasher.update(nonce)
    for values in (secret["token_permutation"], secret["inverse_token_permutation"]):
        for value in values:
            hasher.update(struct.pack("<I", value))
    return hasher.hexdigest()


def load_secret(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != SECRET_KEYS:
        fail("client secret has an invalid field set")
    if checked_integer(value["version"], "secret version") != 1:
        fail("unsupported client secret version")
    method = value["method"]
    if method != {"id": "aloepri-token", "version": "0.1"}:
        fail("client secret method must be aloepri-token/0.1")
    vocab_size = checked_integer(value["vocab_size"], "vocab_size")
    if not 2 <= vocab_size <= 2**32 - 1:
        fail("vocab_size must be between 2 and u32::MAX")
    source_fingerprint = hex_bytes(value["source_fingerprint"], "source_fingerprint")
    nonce = hex_bytes(value["binding_nonce"], "binding_nonce")
    secret_id = value["secret_id"]
    if not isinstance(secret_id, str) or len(secret_id) != 64 or secret_id != secret_id.lower():
        fail("secret_id must be lowercase hexadecimal")
    try:
        bytes.fromhex(secret_id)
    except ValueError:
        fail("secret_id is not hexadecimal")
    permutation = value["token_permutation"]
    inverse = value["inverse_token_permutation"]
    if not isinstance(permutation, list) or not isinstance(inverse, list):
        fail("permutations must be arrays")
    if len(permutation) != vocab_size or len(inverse) != vocab_size:
        fail("permutation length does not match vocab_size")
    permutation = [checked_integer(item, "token permutation item") for item in permutation]
    inverse = [checked_integer(item, "inverse permutation item") for item in inverse]
    if any(item >= vocab_size for item in permutation + inverse):
        fail("permutation contains an out-of-range token id")
    if len(set(permutation)) != vocab_size or len(set(inverse)) != vocab_size:
        fail("permutations must be bijections")
    expected_inverse = [0] * vocab_size
    for original, obfuscated in enumerate(permutation):
        expected_inverse[obfuscated] = original
    if expected_inverse != inverse:
        fail("inverse permutation does not match token_permutation")
    if commitment(value, source_fingerprint, nonce) != secret_id:
        fail("client secret commitment does not match its contents")
    value["token_permutation"] = permutation
    value["inverse_token_permutation"] = inverse
    return value


def read_artifact(root: Path) -> tuple[bytes, dict[str, tuple[str, list[int], bytes]]]:
    config = (root / "config.json").read_bytes()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path)
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            fail(f"invalid weight map in {index_path}")
        shard_names = sorted(set(index["weight_map"].values()))
        expected = index["weight_map"]
    else:
        shard_names = ["model.safetensors"]
        expected = None
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for shard_name in shard_names:
        if not isinstance(shard_name, str) or Path(shard_name).is_absolute() or ".." in Path(shard_name).parts:
            fail(f"invalid shard path {shard_name!r}")
        raw = (root / shard_name).read_bytes()
        if len(raw) < 8:
            fail(f"truncated safetensors shard {shard_name}")
        header_length = int.from_bytes(raw[:8], "little")
        if 8 + header_length > len(raw):
            fail(f"truncated safetensors header {shard_name}")
        header = json.loads(raw[8 : 8 + header_length].decode("utf-8").rstrip(" "))
        payload = raw[8 + header_length :]
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                fail(f"duplicate tensor {name}")
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
                fail(f"invalid metadata for {name}")
            start, end = offsets
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(payload):
                fail(f"invalid offsets for {name}")
            tensors[name] = (dtype, shape, payload[start:end])
            if expected is not None and expected.get(name) != shard_name:
                fail(f"index points {name} to the wrong shard")
    if expected is not None and set(expected) != set(tensors):
        fail("index tensor set differs from physical tensor set")
    return config, tensors


def hash_bytes(hasher: Any, label: bytes, value: bytes) -> None:
    hasher.update(struct.pack("<Q", len(label)))
    hasher.update(label)
    hasher.update(struct.pack("<Q", len(value)))
    hasher.update(value)


def artifact_fingerprint(root: Path, config: bytes, tensors: dict[str, tuple[str, list[int], bytes]]) -> str:
    try:
        import blake3
    except ImportError as error:
        fail(f"the blake3 Python package is required: {error}")
    hasher = blake3.blake3()
    hasher.update(b"aloepri-artifact-fingerprint-v1")
    hash_bytes(hasher, b"config", config)
    for name in sorted(SIDECARS):
        path = root / name
        if path.is_file():
            hash_bytes(hasher, name.encode(), path.read_bytes())
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        hash_bytes(hasher, b"tensor-name", name.encode())
        hash_bytes(hasher, b"tensor-dtype", dtype.encode())
        hash_bytes(hasher, b"tensor-shape", json.dumps(shape, separators=(",", ":")).encode())
        hash_bytes(hasher, b"tensor-length", struct.pack("<Q", len(payload)))
        hasher.update(payload)
    return hasher.hexdigest()


def verify_weight_relationship(
    source: dict[str, tuple[str, list[int], bytes]],
    obfuscated: dict[str, tuple[str, list[int], bytes]],
    permutation: list[int],
) -> None:
    if set(source) != set(obfuscated):
        fail("source and obfuscated physical tensor inventories differ")
    for name in source:
        source_dtype, source_shape, source_bytes = source[name]
        output_dtype, output_shape, output_bytes = obfuscated[name]
        if (source_dtype, source_shape) != (output_dtype, output_shape):
            fail(f"tensor metadata differs for {name}")
        if name not in {"model.embed_tokens.weight", "lm_head.weight"}:
            if source_bytes != output_bytes:
                fail(f"non-vocabulary tensor changed: {name}")
            continue
        if len(source_shape) != 2 or source_shape[0] != len(permutation):
            fail(f"invalid vocabulary shape for {name}")
        width = len(source_bytes) // source_shape[0]
        if width * source_shape[0] != len(source_bytes):
            fail(f"invalid vocabulary payload for {name}")
        for original, obfuscated_id in enumerate(permutation):
            source_row = source_bytes[original * width : (original + 1) * width]
            output_row = output_bytes[obfuscated_id * width : (obfuscated_id + 1) * width]
            if source_row != output_row:
                fail(f"permutation relationship failed for {name} row {original}")
    if source["model.embed_tokens.weight"][2] == obfuscated["model.embed_tokens.weight"][2]:
        fail("embedding payload did not change")


def verify_safetensors(root: Path, expected_count: int) -> None:
    try:
        from safetensors import safe_open
    except ImportError as error:
        fail(f"install the demo dependencies before reading safetensors: {error}")
    opened = 0
    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            opened += len(handle.keys())
    if opened != expected_count:
        fail(f"safe_open exposed {opened} tensors in {root}, expected {expected_count}")


UNSAFE_LOADING_KEYS = {
    "_attn_implementation",
    "_attn_implementation_internal",
    "attn_implementation",
    "attn_implementation_internal",
    "kernels",
    "kernel",
    "use_kernels",
}


def _reject_unsafe_loading_fields(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in UNSAFE_LOADING_KEYS or key.startswith("_attn_"):
                fail(f"unsafe model loading field {path}.{key}")
            _reject_unsafe_loading_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unsafe_loading_fields(child, f"{path}[{index}]")


def load_local_llama_config(root: Path) -> dict[str, Any]:
    value = load_json(root / "config.json")
    if not isinstance(value, dict) or value.get("model_type") != "llama":
        fail("local model config must be a Llama configuration")
    _reject_unsafe_loading_fields(value)
    return value


def load_local_llama_model(root: Path, model_class: Any, torch_module: Any) -> tuple[Any, dict[str, Any]]:
    config_data = load_local_llama_config(root)
    config_class = getattr(model_class, "config_class", None)
    if config_class is None or not hasattr(config_class, "from_dict"):
        fail("Llama model class does not expose a safe config constructor")
    config = config_class.from_dict(config_data)
    model, info = model_class.from_pretrained(
        str(root),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
        dtype=torch_module.float32,
        attn_implementation="eager",
    )
    missing = set(info.get("missing_keys", []))
    allowed_missing = {"lm_head.weight"} if model.config.tie_word_embeddings else set()
    if (
        missing - allowed_missing
        or info.get("unexpected_keys")
        or info.get("mismatched_keys")
        or info.get("error_msgs")
    ):
        fail(f"unexpected local model loading report for {root}: {info}")
    model.eval()
    model.to("cpu")
    return model, info


def load_transformers_models(source: Path, obfuscated: Path) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM
    except ImportError as error:
        fail(f"install the demo dependencies before running inference: {error}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    loaded = [
        load_local_llama_model(root, LlamaForCausalLM, torch)[0]
        for root in (source, obfuscated)
    ]
    if getattr(loaded[0].config, "tie_word_embeddings", False):
        if loaded[0].lm_head.weight.data_ptr() != loaded[0].model.embed_tokens.weight.data_ptr():
            fail("source tied embedding was not restored by Transformers")
        if loaded[1].lm_head.weight.data_ptr() != loaded[1].model.embed_tokens.weight.data_ptr():
            fail("obfuscated tied embedding was not restored by Transformers")
    return tokenizer, loaded[0], loaded[1]


def logits_for(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def greedy(model: Any, input_ids: Any, attention_mask: Any, permutation: list[int], max_new_tokens: int, obfuscated: bool) -> Any:
    import torch

    sequence = input_ids.clone()
    mask = attention_mask.clone()
    permutation_tensor = torch.tensor(permutation, dtype=torch.long)
    eos = getattr(model.config, "eos_token_id", None)
    eos_values = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits = logits_for(model, sequence, mask)[:, -1, :]
            if obfuscated:
                original_logits = logits.index_select(-1, permutation_tensor)
                next_original = torch.argmax(original_logits, dim=-1)
                next_token = permutation_tensor.index_select(0, next_original)
            else:
                next_token = torch.argmax(logits, dim=-1)
            sequence = torch.cat((sequence, next_token[:, None]), dim=1)
            mask = torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)), dim=1)
            original_token = int(next_token[0])
            if obfuscated:
                original_token = next_original.item()
            if original_token in eos_values:
                break
    if obfuscated:
        inverse = [0] * len(permutation)
        for original, obfuscated_id in enumerate(permutation):
            inverse[obfuscated_id] = original
        sequence = torch.tensor(
            [[inverse[int(token)] for token in sequence[0]]], dtype=sequence.dtype
        )
    return sequence


def validate_artifact_binding(
    secret: dict[str, Any],
    source_config: dict[str, Any],
    obfuscated_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if source_config != obfuscated_config:
        fail("source and obfuscated config.json values differ")
    if (
        source_config.get("vocab_size") != secret["vocab_size"]
        or obfuscated_config.get("vocab_size") != secret["vocab_size"]
    ):
        fail("client secret vocab_size does not match model config")
    if manifest.get("artifact_version") != 2 or manifest.get("method") != {
        "id": "aloepri-token",
        "version": "0.1",
    }:
        fail("obfuscated artifact is not an aloepri-token manifest")
    if manifest.get("secret_id") != secret["secret_id"]:
        fail("manifest secret_id does not match client secret")
    if manifest.get("source_fingerprint") != secret["source_fingerprint"]:
        fail("manifest source fingerprint does not match client secret")


def run(args: argparse.Namespace) -> bool:
    secret = load_secret(args.secret)
    source_config, source_tensors = read_artifact(args.source)
    obfuscated_config, obfuscated_tensors = read_artifact(args.obfuscated)
    source_config_value = parse_json(source_config, str(args.source / "config.json"))
    obfuscated_config_value = parse_json(obfuscated_config, str(args.obfuscated / "config.json"))
    manifest = load_json(args.obfuscated / "aloepri.json")
    if source_config != obfuscated_config:
        fail("source and obfuscated config.json bytes differ")
    if not isinstance(source_config_value, dict) or not isinstance(obfuscated_config_value, dict):
        fail("model config.json must contain objects")
    if not isinstance(manifest, dict):
        fail("obfuscated manifest must contain an object")
    validate_artifact_binding(
        secret,
        source_config_value,
        obfuscated_config_value,
        manifest,
    )
    if artifact_fingerprint(args.source, source_config, source_tensors) != secret["source_fingerprint"]:
        fail("client secret source fingerprint does not match source artifact")
    verify_weight_relationship(source_tensors, obfuscated_tensors, secret["token_permutation"])
    verify_safetensors(args.source, len(source_tensors))
    verify_safetensors(args.obfuscated, len(obfuscated_tensors))

    tokenizer, baseline_model, obfuscated_model = load_transformers_models(args.source, args.obfuscated)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.numel() == 0:
        fail("prompt tokenization produced no tokens")
    if int(input_ids.min()) < 0 or int(input_ids.max()) >= secret["vocab_size"]:
        fail("tokenizer produced an id outside the client secret vocabulary")
    import torch

    permutation_tensor = torch.tensor(secret["token_permutation"], dtype=torch.long)
    obfuscated_input = permutation_tensor.index_select(-1, input_ids.reshape(-1)).reshape(input_ids.shape)
    with torch.inference_mode():
        baseline_logits = logits_for(baseline_model, input_ids, attention_mask)
        obfuscated_logits = logits_for(obfuscated_model, obfuscated_input, attention_mask)
    aligned_logits = obfuscated_logits.index_select(-1, permutation_tensor)
    if not torch.isfinite(baseline_logits).all() or not torch.isfinite(aligned_logits).all():
        fail("model logits contain non-finite values")
    max_error = float((baseline_logits - aligned_logits).abs().max())
    logits_equivalent = bool(torch.allclose(baseline_logits, aligned_logits, rtol=1e-4, atol=1e-5))
    baseline_tokens = greedy(baseline_model, input_ids, attention_mask, secret["token_permutation"], args.max_new_tokens, False)
    obfuscated_tokens = greedy(
        obfuscated_model,
        obfuscated_input,
        attention_mask,
        secret["token_permutation"],
        args.max_new_tokens,
        True,
    )
    token_equivalent = torch.equal(baseline_tokens, obfuscated_tokens)
    baseline_text = tokenizer.decode(baseline_tokens[0].tolist(), skip_special_tokens=True)
    obfuscated_text = tokenizer.decode(obfuscated_tokens[0].tolist(), skip_special_tokens=True)
    equivalent = logits_equivalent and token_equivalent
    print(f"Prompt:\n{args.prompt}\n")
    print(f"Baseline:\n{baseline_text}\n")
    print(f"AloePri:\n{obfuscated_text}\n")
    print(f"Equivalent: {str(equivalent).lower()}")
    print(f"Logits max absolute error: {max_error:.8g}")
    if not equivalent:
        print(f"Logits equivalent: {logits_equivalent}; token IDs equivalent: {token_equivalent}", file=sys.stderr)
    return equivalent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--obfuscated", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    try:
        return 0 if run(args) else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
