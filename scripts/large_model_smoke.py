#!/usr/bin/env python3
"""Large-model smoke test for the managed memory budget.

Generates synthetic safetensors artifacts whose total payload far exceeds the
compiler's `--memory-limit`, runs the release binary as a separate process for
each size, and reports the peak RSS of the Rust process. The claim under test is
that the peak RSS does not grow with the model payload.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

HEADER_PADDING = 8
TIME_BINARY = "/usr/bin/time"

# A complete, minimal Llama dense schema (hidden=1, one layer, one head) so the
# generated artifact passes the architecture adapter's schema validation. The
# bulk payload rides along in passthrough tensors.
CONFIG = (
    '{"model_type":"llama","hidden_size":1,"num_hidden_layers":1,'
    '"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":1,'
    '"vocab_size":1,"tie_word_embeddings":true}'
)
SCHEMA_TENSORS = (
    ("model.embed_tokens.weight", [1, 1]),
    ("model.layers.0.input_layernorm.weight", [1]),
    ("model.layers.0.mlp.down_proj.weight", [1, 1]),
    ("model.layers.0.mlp.gate_proj.weight", [1, 1]),
    ("model.layers.0.mlp.up_proj.weight", [1, 1]),
    ("model.layers.0.post_attention_layernorm.weight", [1]),
    ("model.layers.0.self_attn.k_proj.weight", [1, 1]),
    ("model.layers.0.self_attn.o_proj.weight", [1, 1]),
    ("model.layers.0.self_attn.q_proj.weight", [1, 1]),
    ("model.layers.0.self_attn.v_proj.weight", [1, 1]),
    ("model.norm.weight", [1]),
)


def write_model(root: Path, tensor_bytes: int, tensor_count: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(CONFIG, encoding="utf-8")

    per_tensor = tensor_bytes // tensor_count
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in SCHEMA_TENSORS:
        end = offset + 1
        header[name] = {"dtype": "U8", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    for index in range(tensor_count):
        end = offset + per_tensor
        header[f"passthrough.{index}"] = {
            "dtype": "U8",
            "shape": [per_tensor],
            "data_offsets": [offset, end],
        }
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((-len(header_bytes)) % HEADER_PADDING)

    chunk = bytes(range(256)) * (1024 * 1024 // 256)
    with (root / "model.safetensors").open("wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, "little"))
        handle.write(header_bytes)
        handle.write(bytes(len(SCHEMA_TENSORS)))
        remaining = tensor_bytes
        while remaining > 0:
            size = min(remaining, len(chunk))
            handle.write(chunk[:size])
            remaining -= size
    return len(header_bytes) + 8 + len(SCHEMA_TENSORS) + tensor_bytes


def run(binary: Path, args: list[str], work: Path) -> float:
    """Run the binary and return its own peak RSS in MiB."""
    command = [str(binary), *args]
    if Path(TIME_BINARY).exists():
        completed = subprocess.run(
            [TIME_BINARY, "-v", *command],
            cwd=work,
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr)
            raise SystemExit(f"command failed: {' '.join(args)}")
        for line in completed.stderr.splitlines():
            if "Maximum resident set size" in line:
                return int(line.split(":", 1)[1].strip()) / 1024.0
        raise SystemExit("could not read peak RSS from /usr/bin/time")
    completed = subprocess.run(
        command, cwd=work, capture_output=True, text=True, env={**os.environ}
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"command failed: {' '.join(args)}")
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--memory-limit", default="32MiB")
    parser.add_argument("--sizes", default="128MiB,512MiB")
    parser.add_argument("--tensor-count", type=int, default=16)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    for spec in args.sizes.split(","):
        size = parse_size(spec.strip())
        label = spec.strip()
        model = args.work_dir / f"input-{label}"
        output = args.work_dir / f"output-{label}"
        total = write_model(model, size, args.tensor_count)
        peak = run(
            args.binary,
            [
                "transform",
                str(model),
                "--output",
                str(output),
                "--identity",
                "--memory-limit",
                args.memory_limit,
                "--max-shard-size",
                args.memory_limit,
            ],
            args.work_dir,
        )
        measurements.append((label, total, peak))
        print(f"{label}: payload={size} file={total} peak_rss={peak:.1f}MiB")

    ok = True
    for _, total, peak in measurements:
        if peak >= total / (1024 * 1024):
            ok = False
    spread = max(peak for _, _, peak in measurements) - min(
        peak for _, _, peak in measurements
    )
    print(f"peak RSS spread across sizes: {spread:.1f}MiB")
    if spread > 16:
        ok = False
    print("RESULT:", "ok" if ok else "unexpected growth")
    raise SystemExit(0 if ok else 1)


def parse_size(value: str) -> int:
    for suffix, scale in (
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
        ("B", 1),
    ):
        if value.endswith(suffix):
            return int(value[: -len(suffix)]) * scale
    return int(value)


if __name__ == "__main__":
    main()
