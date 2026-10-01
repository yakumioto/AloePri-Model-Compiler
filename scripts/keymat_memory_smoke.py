import argparse
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def write_source(root, vocab, intermediate):
    root.mkdir()
    config = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": intermediate, "vocab_size": vocab, "tie_word_embeddings": True}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    specs = {"model.embed_tokens.weight": [vocab, 8], "model.norm.weight": [8], "model.layers.0.input_layernorm.weight": [8], "model.layers.0.post_attention_layernorm.weight": [8]}
    for suffix, shape in {"self_attn.q_proj": [8, 8], "self_attn.k_proj": [4, 8], "self_attn.v_proj": [4, 8], "self_attn.o_proj": [8, 8], "mlp.gate_proj": [intermediate, 8], "mlp.up_proj": [intermediate, 8], "mlp.down_proj": [8, intermediate]}.items():
        specs[f"model.layers.0.{suffix}.weight"] = shape
    header, offset = {}, 0
    for name, shape in sorted(specs.items()):
        length = math.prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with (root / "model.safetensors").open("xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name, shape in sorted(specs.items()):
            chunk = struct.pack("<f", 1.0 if "norm" in name else 0.01) * 16384
            remaining = math.prod(shape) * 4
            while remaining:
                current = min(remaining, len(chunk))
                handle.write(chunk[:current])
                remaining -= current
    return offset


def run(binary, root, vocab, intermediate):
    source, output, secret = root / "source", root / "output", root / "secret.json"
    payload = write_source(source, vocab, intermediate)
    rss_file = root / "rss.txt"
    result = subprocess.run(["/usr/bin/time", "-f", "%M", "-o", str(rss_file), str(binary), "transform", str(source), "--output", str(output), "--method", "aloepri-keymat", "--expansion-size", "2", "--keymat-lambda", "0.3", "--keymat-fixture-seed", "42", "--secret-output", str(secret), "--memory-limit", "4MiB", "--max-shard-size", "8MiB"], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    verified = subprocess.run([str(binary), "verify", str(output)], capture_output=True, text=True, check=True)
    verification = json.loads(verified.stdout)
    assert verification["structure_valid"] and verification["plan_verified"]
    assert report["memory_estimate"]["method_state_bytes"] == 16 * 8 * 12
    assert report["memory_estimate"]["peak_bytes"] <= 4 * 1024**2
    return {"source_payload_bytes": payload, "output_payload_bytes": report["transform"]["payload_bytes"], "peak_rss_kib": int(rss_file.read_text().strip()), "memory_estimate": report["memory_estimate"], "generation_peak_bytes": report["generation_peak_bytes"], "structure_valid": verification["structure_valid"]}


def main():
    parser = argparse.ArgumentParser(description="Measure RSS of actual F32 right/left KeyMat transforms")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        small, large = root / "small", root / "large"
        small.mkdir()
        large.mkdir()
        results = [run(args.binary.resolve(), small, 32768, 8192), run(args.binary.resolve(), large, 1048576, 262144)]
    growth = results[1]["peak_rss_kib"] - results[0]["peak_rss_kib"]
    passed = growth <= 32 * 1024
    report = {"d": 8, "h": 2, "D": 12, "memory_limit_bytes": 4 * 1024**2, "rss_growth_kib": growth, "rss_growth_limit_kib": 32 * 1024, "bounded_rss": passed, "runs": results, "note": "RSS is an OS measurement, not the managed MemoryBudget"}
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
