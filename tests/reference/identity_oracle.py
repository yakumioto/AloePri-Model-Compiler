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

    try:
        import blake3
    except ImportError:
        blake3 = None
    if blake3 is not None:
        print(f"verified {len(source)} tensors; source/output BLAKE3: {blake3.blake3(b''.join(source[n][2] for n in sorted(source))).hexdigest()}")
    else:
        print(f"verified {len(source)} tensors byte-for-byte (install blake3 for digest output)")

    if args.transformers_load:
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM

        for shard in sorted(args.output.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu"):
                pass
        AutoModelForCausalLM.from_pretrained(
            str(args.output),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )


if __name__ == "__main__":
    main()
