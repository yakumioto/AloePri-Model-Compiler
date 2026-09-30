import argparse
import json
import math
import struct
from pathlib import Path

import blake3
import torch
import transformers
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from token_permutation_demo import load_json, load_local_llama_config

METHOD = {"id": "aloepri-keymat", "version": "0.1"}
ALGORITHM = "algorithm1-v1"
RNG = "chacha20-rand0.9-normal0.5-v1"
TOLERANCE = 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def ordered(value, fields):
    names = fields.split()
    require(set(value) <= set(names), "unknown canonical contract field")
    return {name: value[name] for name in names if name in value}


def canonical_layout(layout):
    value = ordered(dict(layout, index=layout.get("index")), "shards tensors index")
    value["shards"] = [ordered(shard, "id filename payload_length file_length header") for shard in value["shards"]]
    value["tensors"] = [ordered(tensor, "name shape dtype byte_length shard offset") for tensor in value["tensors"]]
    return value


def canonical_plan(plan):
    # Mirror serde struct order, independently of JSON input member order.
    value = ordered(plan, "version source_fingerprint source_inventory output_inventory method runtime_contract secret_id keymat_binding architecture operations output_layout memory_estimate plan_hash")
    def descriptor(raw):
        result = ordered(raw, "name shape dtype byte_length location")
        if "location" in result:
            result["location"] = ordered(result["location"], "shard offset length")
        return result
    value["source_inventory"] = [descriptor(tensor) for tensor in value["source_inventory"]]
    value["output_inventory"] = [descriptor(tensor) for tensor in value["output_inventory"]]
    value["method"] = ordered(value["method"], "id version")
    runtime = ordered(value["runtime_contract"], "id version standard_hf_checkpoint architecture logical_dimensions physical_dimensions expansion_size norm_mode kv_cache_format")
    for field in ["logical_dimensions", "physical_dimensions"]:
        runtime[field] = dict(sorted(runtime[field].items()))
    value["runtime_contract"] = runtime
    value["keymat_binding"] = ordered(value["keymat_binding"], "secret_id source_fingerprint hidden_size expansion_size physical_hidden_size lambda_bits algorithm rng")
    operations = []
    for operation in value["operations"]:
        result = ordered(operation, "id kind inputs output memory_requirement dependencies")
        if result.get("kind") == "copy":
            del result["kind"]
        result["inputs"] = [{"descriptor": descriptor(item["descriptor"])} for item in result["inputs"]]
        result["output"] = {"descriptor": descriptor(result["output"]["descriptor"])}
        operations.append(result)
    value["operations"] = operations
    value["output_layout"] = canonical_layout(value["output_layout"])
    value["memory_estimate"] = ordered(value["memory_estimate"], "metadata_bytes input_buffer_bytes output_buffer_bytes transform_scratch_bytes method_state_bytes peak_bytes")
    value["plan_hash"] = blake3.blake3(b"excluded").hexdigest()
    return value


def child_file(root, name):
    require(isinstance(name, str) and name and Path(name).name == name and name not in {".", ".."}, "unsafe artifact filename")
    path = root / name
    require(not path.is_symlink() and path.is_file(), "artifact file must be a regular non-symlink file")
    return path


def check_hex(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), "invalid KeyMat identity")


def load_keymat(secret_path):
    require(not any(p.is_symlink() for p in [secret_path, *secret_path.parents]), "Secret path contains a symlink")
    require(secret_path.stat().st_size <= 65536, "Secret JSON is too large")
    secret = load_json(secret_path)
    expected_fields = {"version", "method", "secret_id", "source_fingerprint", "hidden_size", "expansion_size", "lambda_bits", "algorithm", "rng", "master_seed", "nullspace_cutoff", "material_format", "material_file", "p_digest", "q_digest"}
    require(set(secret) == expected_fields, "unknown or missing Secret fields")
    require(secret["version"] == 1 and secret["method"] == METHOD and secret["algorithm"] == ALGORITHM and secret["rng"] == RNG, "unsupported Secret version")
    d, h = secret["hidden_size"], secret["expansion_size"]
    require(type(d) is int and type(h) is int and d > 0 and h > 0 and h % 2 == 0, "invalid KeyMat dimensions")
    big_d = d + 2 * h
    bits = secret["lambda_bits"]
    require(type(bits) is int and 0 <= bits < 2**64, "invalid lambda identity")
    lam = struct.unpack("<d", struct.pack("<Q", bits))[0]
    require(math.isfinite(lam) and lam >= 0 and not (lam == 0 and bits != 0), "invalid lambda")
    require(secret["nullspace_cutoff"] == 1e-10 and secret["material_format"] == "keymat-f64-le-v1" and secret["material_file"] == "key-material.bin", "unsupported KeyMat material parameters")
    for name in ["secret_id", "source_fingerprint", "master_seed", "p_digest", "q_digest"]:
        check_hex(secret[name])
    commitment = blake3.blake3(b"aloepri-keymat-secret-v1")
    commitment.update(struct.pack("<I", 1))
    for value in [METHOD["id"], METHOD["version"], secret["source_fingerprint"], ALGORITHM, RNG, secret["master_seed"], secret["material_format"], secret["material_file"], secret["p_digest"], secret["q_digest"]]:
        encoded = value.encode("utf-8")
        commitment.update(struct.pack("<I", len(encoded)))
        commitment.update(encoded)
    for value in [d, h, bits, struct.unpack("<Q", struct.pack("<d", 1e-10))[0]]:
        commitment.update(struct.pack("<Q", value))
    require(commitment.hexdigest() == secret["secret_id"], "Secret commitment mismatch")
    material = child_file(secret_path.parent, secret["material_file"])
    require(material.stat().st_size == 16 * d * big_d, "KeyMat binary length mismatch")
    data = material.read_bytes()
    split = 8 * d * big_d
    require(blake3.blake3(data[:split]).hexdigest() == secret["p_digest"] and blake3.blake3(data[split:]).hexdigest() == secret["q_digest"], "KeyMat digest mismatch")
    values = torch.frombuffer(bytearray(data), dtype=torch.float64)
    p = values[:d * big_d].view(d, big_d)
    q = values[d * big_d:].view(big_d, d)
    require(torch.isfinite(values).all().item(), "non-finite KeyMat")
    error = (p @ q - torch.eye(d, dtype=torch.float64)).abs()
    require(torch.isfinite(error).all().item() and error.max().item() <= TOLERANCE, "G1 KeyMat identity failed")
    ps, qs = torch.linalg.svdvals(p), torch.linalg.svdvals(q)
    diagnostics = {"max_abs_pq_error": error.max().item(), "mean_abs_pq_error": error.mean().item(), "finite_p": True, "finite_q": True, "p_frobenius_norm": p.norm().item(), "q_frobenius_norm": q.norm().item(), "p_spectral_norm": ps.max().item(), "q_spectral_norm": qs.max().item(), "p_condition_estimate": (ps.max() / ps.min()).item()}
    binding = {"secret_id": secret["secret_id"], "source_fingerprint": secret["source_fingerprint"], "hidden_size": d, "expansion_size": h, "physical_hidden_size": big_d, "lambda_bits": bits, "algorithm": ALGORITHM, "rng": RNG}
    return binding, p, q, diagnostics


def tensor_specs(config, big_d):
    d, v, m = config.hidden_size, config.vocab_size, config.intermediate_size
    require(d > 0 and v > 0 and m > 0 and config.num_hidden_layers > 0 and not config.attention_bias and not config.mlp_bias, "unsupported logical Llama dimensions/bias")
    require(config.num_key_value_heads > 0 and config.num_attention_heads % config.num_key_value_heads == 0 and config.num_attention_heads * config.head_dim == d, "invalid logical attention geometry")
    specs = {"model.embed_tokens.weight": (v, big_d), "lm_head.weight": (v, big_d), "model.norm.weight": (d,)}
    for i in range(config.num_hidden_layers):
        for name, out in [("self_attn.q_proj", d), ("self_attn.k_proj", config.num_key_value_heads * config.head_dim), ("self_attn.v_proj", config.num_key_value_heads * config.head_dim), ("mlp.gate_proj", m), ("mlp.up_proj", m)]:
            specs[f"model.layers.{i}.{name}.weight"] = (out, big_d)
        for name, size in [("self_attn.o_proj", d), ("mlp.down_proj", m)]:
            specs[f"model.layers.{i}.{name}.weight"] = (big_d, size)
        for name in ["input_layernorm", "post_attention_layernorm"]:
            specs[f"model.layers.{i}.{name}.weight"] = (d,)
    return specs


class ExactCovariantNorm(LlamaRMSNorm):
    def __init__(self, d, eps, p, q):
        super().__init__(d, eps)
        self.register_buffer("_p", p, persistent=False)
        self.register_buffer("_q", q, persistent=False)

    def forward(self, z):
        require(z.shape[-1] == self._q.shape[0] and z.dtype == torch.float32, "expected expanded F32 residual")
        logical = (z.double() @ self._q).float()
        normalized = super().forward(logical)
        return (normalized.double() @ self._p).float()


def load_runtime(artifact_root, secret_path):
    require(transformers.__version__ == "5.5.0" and torch.__version__.split("+")[0] == "2.9.1", "runtime requires the pinned demo dependencies")
    root, secret_path = Path(artifact_root).resolve(), Path(secret_path)
    require(not secret_path.resolve().is_relative_to(root), "Secret must be outside the artifact")
    manifest = load_json(child_file(root, "aloepri.json"))
    binding, p, q, diagnostics = load_keymat(secret_path)
    require(manifest.get("artifact_version") == 4 and manifest.get("method") == METHOD and manifest.get("secret_key_id") is None, "not a KeyMat v4 artifact")
    require(manifest.get("standard_hf_checkpoint") is False, "KeyMat must not claim standard HF support")
    plan, runtime = manifest["plan"], manifest["runtime_contract"]
    require(plan["version"] == 4 and plan["method"] == METHOD and plan.get("keymat_binding") == binding, "KeyMat plan binding mismatch")
    for field in ["architecture", "source_fingerprint", "plan_hash", "secret_id", "output_layout", "runtime_contract"]:
        require(plan[field] == manifest[field], f"manifest/plan {field} mismatch")
    require(manifest["source_fingerprint"] == binding["source_fingerprint"] and manifest["secret_id"] == binding["secret_id"], "wrong runtime Secret")
    require(runtime["id"] == "aloepri" and runtime["version"] == "1" and runtime["architecture"] == "llama" and manifest["architecture"] == "llama" and runtime["standard_hf_checkpoint"] is False and runtime["norm_mode"] == "exact_covariant" and runtime["kv_cache_format"] == "standard_projection_v1", "unsupported runtime contract")
    require(runtime["expansion_size"] == binding["expansion_size"] and runtime["logical_dimensions"]["hidden_size"] == binding["hidden_size"], "runtime logical dimensions mismatch")
    expected_physical = dict(runtime["logical_dimensions"], hidden_size=binding["physical_hidden_size"])
    require(runtime["physical_dimensions"] == expected_physical, "runtime physical dimensions mismatch")
    require(blake3.blake3(canonical(canonical_plan(plan))).hexdigest() == plan["plan_hash"], "plan hash mismatch")
    require(blake3.blake3(canonical(canonical_layout(plan["output_layout"]))).hexdigest() == manifest["layout_hash"], "layout hash mismatch")
    config_data = load_local_llama_config(root)
    require(blake3.blake3((root / "config.json").read_bytes()).hexdigest() == manifest["config_blake3"], "logical config digest mismatch")
    for name, value in runtime["logical_dimensions"].items():
        require(config_data.get(name) == value, "logical config dimension mismatch")
    for name, digest in manifest["sidecar_blake3"].items():
        require(blake3.blake3(child_file(root, name).read_bytes()).hexdigest() == digest, "sidecar digest mismatch")
    config = LlamaConfig.from_dict(config_data)
    config._attn_implementation = "eager"
    big_d, d = binding["physical_hidden_size"], binding["hidden_size"]
    require(config.num_hidden_layers * 9 + 3 == len(plan["output_inventory"]), "logical layer count disagrees with inventory")
    specs = tensor_specs(config, big_d)
    source_names = [t["name"] for t in plan["source_inventory"]]
    require(len(source_names) == len(set(source_names)) and set(source_names) in [set(specs), set(specs) - {"lm_head.weight"}], "invalid source inventory")
    require("lm_head.weight" in source_names or config.tie_word_embeddings, "untied source cannot omit head")
    require(len(plan["operations"]) == len(specs), "invalid operation count")
    produced, operation_ids = set(), set()
    for operation in plan["operations"]:
        name = operation["output"]["descriptor"]["name"]
        require(name in specs and name not in produced and operation["id"] not in operation_ids, "invalid operation producer")
        require(all(dependency in operation_ids for dependency in operation["dependencies"]), "invalid operation dependencies")
        produced.add(name)
        operation_ids.add(operation["id"])
        require(len(operation["inputs"]) == 1, "KeyMat requires one input")
        descriptor = operation["inputs"][0]["descriptor"]
        expected_name = "model.embed_tokens.weight" if name == "lm_head.weight" and "lm_head.weight" not in source_names else name
        require(descriptor["name"] == expected_name and descriptor["dtype"] == "F32", "invalid KeyMat source")
        require(descriptor in plan["source_inventory"], "input/source metadata mismatch")
        if "norm" in name:
            expected_kind = "copy"
        elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            expected_kind = {"keymat_left": {"role": "OutputPTranspose"}}
        else:
            role = "EmbeddingP" if name == "model.embed_tokens.weight" else ("HeadQTranspose" if name == "lm_head.weight" else "InputQTranspose")
            expected_kind = {"keymat_right": {"role": role}}
        require(operation.get("kind", "copy") == expected_kind and tuple(operation["output"]["descriptor"]["shape"]) == specs[name], "invalid KeyMat operation geometry/role")
        logical_shape = list(specs[name])
        if isinstance(expected_kind, dict):
            logical_shape[0 if "keymat_left" in expected_kind else 1] = d
        require(descriptor["shape"] == logical_shape, "invalid logical source shape")
    for inventory in [plan["output_inventory"], plan["output_layout"]["tensors"], manifest["tensors"]]:
        require(len(inventory) == len(specs) and {t["name"] for t in inventory} == set(specs), "incomplete tensor inventory")
        for tensor in inventory:
            require(tuple(tensor["shape"]) == specs[tensor["name"]] and tensor["dtype"] == "F32" and tensor["byte_length"] == math.prod(tensor["shape"]) * 4, "physical shape/dtype mismatch")
    expected_digests = {t["name"]: t["blake3"] for t in manifest["tensors"]}
    weight_map = {}
    layout = plan["output_layout"]
    require(len({s["filename"] for s in layout["shards"]}) == len(layout["shards"]), "duplicate shards")
    for tensor in layout["tensors"]:
        weight_map[tensor["name"]] = layout["shards"][tensor["shard"]]["filename"]
    if layout.get("index") is not None:
        index = load_json(child_file(root, "model.safetensors.index.json"))
        require(index["weight_map"] == weight_map, "artifact index mismatch")
    states = {}
    for shard in layout["shards"]:
        path = child_file(root, shard["filename"])
        require(path.stat().st_size == shard["file_length"], "shard size mismatch")
        with safe_open(path, framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == {k for k, v in weight_map.items() if v == shard["filename"]}, "shard inventory mismatch")
            for name in handle.keys():
                require(name not in states, "duplicate tensor")
                tensor = handle.get_tensor(name)
                require(tuple(tensor.shape) == specs[name] and tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(), "invalid physical tensor")
                require(blake3.blake3(tensor.numpy().tobytes()).hexdigest() == expected_digests[name], "tensor digest mismatch")
                states[name] = tensor
    require(set(states) == set(specs), "missing physical weights")
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
        model.model.embed_tokens = torch.nn.Embedding(config.vocab_size, big_d, config.pad_token_id)
        model.lm_head = torch.nn.Linear(big_d, config.vocab_size, bias=False)
        for layer in model.model.layers:
            for name in ["q_proj", "k_proj", "v_proj"]:
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, torch.nn.Linear(big_d, original.out_features, bias=False))
            layer.self_attn.o_proj = torch.nn.Linear(layer.self_attn.o_proj.in_features, big_d, bias=False)
            for name in ["gate_proj", "up_proj"]:
                setattr(layer.mlp, name, torch.nn.Linear(big_d, config.intermediate_size, bias=False))
            layer.mlp.down_proj = torch.nn.Linear(config.intermediate_size, big_d, bias=False)
            layer.input_layernorm = ExactCovariantNorm(d, config.rms_norm_eps, p, q)
            layer.post_attention_layernorm = ExactCovariantNorm(d, config.rms_norm_eps, p, q)
        model.model.norm = ExactCovariantNorm(d, config.rms_norm_eps, p, q)
    model.model.rotary_emb = LlamaRotaryEmbedding(config, device="cpu")
    model.load_state_dict(states, strict=True, assign=True)
    require(model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "physical embedding/head must not be retied")
    model.eval()
    return model, q, diagnostics


@torch.inference_mode()
def greedy(model, prompt_ids, max_new_tokens, eos_ids):
    require(prompt_ids and max_new_tokens >= 0, "greedy requires a nonempty prompt and nonnegative token count")
    eos = set(eos_ids if isinstance(eos_ids, list) else ([] if eos_ids is None else [eos_ids]))
    generated = list(prompt_ids)
    cache = None
    for _ in range(max_new_tokens):
        ids = generated if cache is None else generated[-1:]
        mask = torch.ones((1, len(generated)), dtype=torch.long)
        result = model(input_ids=torch.tensor([ids]), attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        token = result.logits[0, -1].argmax().item()
        generated.append(token)
        if token in eos:
            break
    return generated


def main():
    parser = argparse.ArgumentParser(description="Diagnostic exact_covariant runtime, not the AloePri paper runtime")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--token-ids", type=int, nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _, diagnostics = load_runtime(args.artifact, args.secret)
    tokenizer = None
    if args.token_ids:
        ids = args.token_ids
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.artifact, local_files_only=True, trust_remote_code=False)
        ids = tokenizer.encode(args.prompt)
    require(ids, "empty prompt")
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits
    tokens = greedy(model, ids, args.max_new_tokens, model.config.eos_token_id) if args.max_new_tokens else ids
    print(json.dumps({"mode": "exact_covariant", "keymat": diagnostics, "logits_shape": list(logits.shape), "finite_logits": torch.isfinite(logits).all().item(), "token_ids": tokens, "text": tokenizer.decode(tokens) if tokenizer else None}))


if __name__ == "__main__":
    main()
