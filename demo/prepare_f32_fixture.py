import argparse
import json
import shutil
from pathlib import Path

import torch
import transformers
from transformers import LlamaForCausalLM
from safetensors import safe_open
from token_permutation_demo import load_local_llama_model


def main():
    parser = argparse.ArgumentParser(description="Export a local baseline as a separate F32 source fixture")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fixture output already exists")
    torch.set_num_threads(1)
    model, _ = load_local_llama_model(args.source, LlamaForCausalLM, torch)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
        source = args.source / name
        if source.is_file():
            shutil.copyfile(source, args.output / name)
    inventory = {}
    for shard in args.output.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                if tensor.get_dtype() != "F32":
                    raise ValueError("export is not F32")
                inventory[name] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
    args.report.write_text(json.dumps({"torch": torch.__version__, "transformers": transformers.__version__, "config": json.loads((args.output / "config.json").read_text()), "inventory": inventory}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
