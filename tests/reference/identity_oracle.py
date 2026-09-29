#!/usr/bin/env python3
"""Independent byte-level oracle for an Identity safetensors output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SIDECARS = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def load_model(root: Path) -> tuple[bytes, dict[str, tuple[str, list[int], bytes]]]:
    config = (root / "config.json").read_bytes()
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
        expected = index["weight_map"]
    else:
        shards = ["model.safetensors"]
        expected = None

    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for shard_name in shards:
        path = root / shard_name
        raw = path.read_bytes()
        header_length = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + header_length].decode("utf-8").rstrip(" "))
        payload = raw[8 + header_length :]
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            start, end = metadata["data_offsets"]
            tensors[name] = (metadata["dtype"], metadata["shape"], payload[start:end])
            if expected is not None and expected.get(name) != shard_name:
                raise AssertionError(f"index points {name} to the wrong shard")
    if expected is not None and set(expected) != set(tensors):
        raise AssertionError("index tensor set differs from physical tensor set")
    return config, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transformers-load", action="store_true")
    args = parser.parse_args()

    source_config, source = load_model(args.source)
    output_config, output = load_model(args.output)
    assert source_config == output_config, "config.json bytes differ"
    assert set(source) == set(output), "tensor names differ"
    for name in sorted(source):
        source_dtype, source_shape, source_bytes = source[name]
        output_dtype, output_shape, output_bytes = output[name]
        assert source_dtype == output_dtype, f"dtype differs for {name}"
        assert source_shape == output_shape, f"shape differs for {name}"
        assert source_bytes == output_bytes, f"payload differs for {name}"
    for name in SIDECARS:
        source_sidecar = args.source / name
        output_sidecar = args.output / name
        if source_sidecar.exists():
            assert output_sidecar.read_bytes() == source_sidecar.read_bytes(), f"sidecar differs: {name}"
        else:
            assert not output_sidecar.exists(), f"unexpected sidecar: {name}"

    total_bytes = sum(len(source[name][2]) for name in source)
    print(f"compared {len(source)} tensors, {total_bytes} payload bytes, byte-for-byte equal")

    try:
        import blake3
    except ImportError:
        print("per-tensor BLAKE3 skipped: install the `blake3` package to cross-check digests")
    else:
        for name in sorted(source):
            source_digest = blake3.blake3(source[name][2]).hexdigest()
            output_digest = blake3.blake3(output[name][2]).hexdigest()
            assert source_digest == output_digest, f"BLAKE3 differs for {name}"
        aggregate = blake3.blake3(
            b"".join(blake3.blake3(source[n][2]).digest() for n in sorted(source))
        ).hexdigest()
        print(f"per-tensor BLAKE3 equal for {len(source)} tensors; aggregate {aggregate}")

    if args.transformers_load:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        opened = 0
        for shard in sorted(args.output.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                opened += len(handle.keys())
        assert opened == len(output), "safe_open did not expose every tensor"
        print(f"safe_open read {opened} tensors from {len(list(args.output.glob('*.safetensors')))} shards")

        _model, info = AutoModelForCausalLM.from_pretrained(
            str(args.output),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
        )
        missing = set(info.get("missing_keys", []))
        unexpected = set(info.get("unexpected_keys", []))
        mismatched = info.get("mismatched_keys", [])
        errors = info.get("error_msgs", [])
        # The checkpoint stores no `lm_head.weight` because the config ties it to
        # the embedding; the framework materialises it from the alias.
        allowed_missing = {"lm_head.weight"}
        assert not unexpected, f"unexpected keys: {sorted(unexpected)}"
        assert not mismatched, f"mismatched keys: {mismatched}"
        assert not errors, f"loading errors: {errors}"
        assert missing <= allowed_missing, f"unexpected missing keys: {sorted(missing - allowed_missing)}"
        print(
            "transformers local load: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)} "
            f"mismatched={mismatched} errors={errors}"
        )
        tied = getattr(_model.config, "tie_word_embeddings", False)
        assert tied is True, "expected tied word embeddings in the loaded config"
        assert _model.lm_head.weight.data_ptr() == _model.model.embed_tokens.weight.data_ptr(), (
            "tied lm_head does not share storage with the embedding"
        )
        print("tied lm_head shares storage with model.embed_tokens.weight")


if __name__ == "__main__":
    main()
