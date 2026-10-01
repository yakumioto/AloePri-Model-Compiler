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
from signed_null_linear import signed_null_support, SignedColumnBlockedLinear, SignedRowBlockedLinear

METHOD = {"id": "aloepri-keymat", "version": "0.1"}
ALGORITHM = "algorithm1-v1"
BALANCED_ALGORITHM = "algorithm1-balanced-null-v2"
SIGNED_ALGORITHM = "algorithm1-signed-null-v3"
SUPPORTED_ALGORITHMS = {ALGORITHM, BALANCED_ALGORITHM, SIGNED_ALGORITHM}
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
    algorithm = secret["algorithm"]
    require(secret["version"] == 1 and secret["method"] == METHOD and algorithm in SUPPORTED_ALGORITHMS and secret["rng"] == RNG, "unsupported Secret version")
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
    for value in [METHOD["id"], METHOD["version"], secret["source_fingerprint"], algorithm, RNG, secret["master_seed"], secret["material_format"], secret["material_file"], secret["p_digest"], secret["q_digest"]]:
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
    binding = {"secret_id": secret["secret_id"], "source_fingerprint": secret["source_fingerprint"], "hidden_size": d, "expansion_size": h, "physical_hidden_size": big_d, "lambda_bits": bits, "algorithm": algorithm, "rng": RNG}
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


F64_LINEAR_ROW_CHUNK = 1024
NATIVE_LINEAR_PROFILE = "f32-state-native-linear-v1"
ALL_BOUNDARY_LINEAR_PROFILE = "balanced-v2-all-boundary-f64-acc-v1"
SIGNED_LINEAR_PROFILE = "signed-null-f32-blocked-v1"
SUPPORTED_LINEAR_PROFILES = {NATIVE_LINEAR_PROFILE, ALL_BOUNDARY_LINEAR_PROFILE, SIGNED_LINEAR_PROFILE}


class F64AccumLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=torch.float32):
        require(not bias and dtype == torch.float32, "candidate Linear requires F32 parameters and no bias")
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)
        self.peak_f64_temporary_bytes = 0
        self.peak_working_bytes = 0
        self.elapsed_seconds = 0.0
        self.calls = 0

    @classmethod
    def from_linear(cls, module):
        require(module.bias is None and module.weight.dtype == torch.float32, "candidate Linear requires stored F32 weights")
        with torch.device("meta"):
            replacement = cls(module.in_features, module.out_features)
        replacement.weight = module.weight
        replacement.train(module.training)
        return replacement

    @torch.inference_mode()
    def forward(self, inputs):
        import time
        require(inputs.dtype == self.weight.dtype == torch.float32 and self.bias is None, "candidate Linear forbids dtype fallback/bias")
        require(inputs.shape[-1] == self.in_features and inputs.device == self.weight.device, "candidate Linear input geometry/device mismatch")
        started = time.perf_counter()
        shape = (*inputs.shape[:-1], self.out_features)
        output = torch.empty(shape, dtype=torch.float32, device=inputs.device)
        inputs64 = inputs.double()
        positions = inputs.numel() // self.in_features
        for start in range(0, self.out_features, F64_LINEAR_ROW_CHUNK):
            end = min(start + F64_LINEAR_ROW_CHUNK, self.out_features)
            weight64 = self.weight[start:end].double()
            result64 = torch.nn.functional.linear(inputs64, weight64, None)
            require(torch.isfinite(result64).all().item(), "non-finite candidate Linear result")
            result32 = result64.float()
            require(torch.isfinite(result32).all().item(), "non-finite candidate F32 output")
            output[..., start:end].copy_(result32)
            f64_bytes = inputs64.numel() * 8 + weight64.numel() * 8 + result64.numel() * 8
            working_bytes = f64_bytes + (positions * self.out_features + result32.numel()) * 4
            self.peak_f64_temporary_bytes = max(self.peak_f64_temporary_bytes, f64_bytes)
            self.peak_working_bytes = max(self.peak_working_bytes, working_bytes)
            del weight64, result64, result32
        require(output.dtype == torch.float32 and output.shape == shape, "candidate Linear changed output boundary")
        self.calls += 1
        self.elapsed_seconds += time.perf_counter() - started
        return output


def _linear_targets(model, big_d):
    config = model.config
    require(len(model.model.layers) == config.num_hidden_layers, "profile layer count mismatch")
    specs = tensor_specs(config, big_d)
    targets = []
    for index, layer in enumerate(model.model.layers):
        for section, members in [("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                                 ("mlp", ["gate_proj", "up_proj", "down_proj"])]:
            parent = getattr(layer, section)
            for member in members:
                name = f"model.layers.{index}.{section}.{member}"
                out_features, in_features = specs[name + ".weight"]
                targets.append((name, parent, member, in_features, out_features))
    targets.append(("lm_head", model, "lm_head", big_d, config.vocab_size))
    require(len(targets) == 7 * config.num_hidden_layers + 1, "incomplete profile target inventory")
    return targets


def _profile_implementation(profile, member):
    if profile == NATIVE_LINEAR_PROFILE: return torch.nn.Linear
    if profile == ALL_BOUNDARY_LINEAR_PROFILE: return F64AccumLinear
    require(profile == SIGNED_LINEAR_PROFILE, "unknown role profile")
    return SignedRowBlockedLinear if member in ["o_proj", "down_proj"] else SignedColumnBlockedLinear


def _validate_linear_targets(targets, profile):
    for name, parent, member, in_features, out_features in targets:
        module = getattr(parent, member)
        require(type(module) is _profile_implementation(profile, member), f"mixed/partial Linear profile at {name}")
        require(module.bias is None and module.weight.dtype == torch.float32 and module.weight.device.type == "cpu", f"profile requires CPU/F32/no-bias at {name}")
        require(module.in_features == in_features and module.out_features == out_features and tuple(module.weight.shape) == (out_features, in_features), f"profile geometry mismatch at {name}")


def linear_profile_summary(model):
    require(hasattr(model, "_aloepri_linear_profile"), "runtime profile metadata is missing")
    profile = model._aloepri_linear_profile
    require(profile in SUPPORTED_LINEAR_PROFILES, "unknown installed Linear profile")
    if profile == ALL_BOUNDARY_LINEAR_PROFILE:
        require(model._aloepri_construction_algorithm == BALANCED_ALGORITHM, "all-boundary profile requires balanced-v2")
    targets = _linear_targets(model, model._aloepri_physical_hidden_size)
    _validate_linear_targets(targets, profile)
    if profile == SIGNED_LINEAR_PROFILE:
        require(model._aloepri_construction_algorithm == SIGNED_ALGORITHM, "signed-null profile requires v3")
        active, _ = signed_null_support(model.model.norm._p, model.model.norm._q)
        for _, parent, member, _, _ in targets:
            module = getattr(parent, member)
            require(module.d == model.config.hidden_size and torch.equal(module._active_p, active), "signed-null role support mismatch")
    _validate_linear_targets(targets, profile)
    require(all(getattr(parent, member).weight is original for (_, parent, member, _, _), original in zip(targets, model._aloepri_linear_parameters)), "profile changed Parameter objects")
    require(len(model._aloepri_linear_parameters) == len(targets), "profile Parameter inventory mismatch")
    require(all(value.dtype == torch.float32 and value.device.type == "cpu" for value in model.parameters()), "non-F32/CPU model parameters")
    require(all(value.dtype == torch.float32 for value in model.state_dict().values()), "non-F32 persistent model state")
    require(model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "profile retied physical head")
    roles = [{"name": name, "in_features": width, "out_features": height,
              "implementation": type(getattr(parent, member)).__name__, "parameter_dtype": "torch.float32"}
             for name, parent, member, width, height in targets]
    return {"execution_profile": profile, "construction_profile": model._aloepri_construction_algorithm,
            "installed_role_count": len(roles), "wrapped_role_count": len(roles) if profile != NATIVE_LINEAR_PROFILE else 0,
            "parameter_objects_preserved": True, "parameters_and_state_f32": True, "physical_head_independent": True,
            "physical_hidden_size": model._aloepri_physical_hidden_size, "roles": roles}


def install_linear_profile(model, algorithm, big_d, *, linear_profile=NATIVE_LINEAR_PROFILE):
    require(linear_profile in SUPPORTED_LINEAR_PROFILES, "unknown Linear execution profile")
    require(algorithm in SUPPORTED_ALGORITHMS, "unknown profile construction algorithm")
    if linear_profile == ALL_BOUNDARY_LINEAR_PROFILE:
        require(algorithm == BALANCED_ALGORITHM, "all-boundary profile requires balanced-v2 artifact")
    if linear_profile == SIGNED_LINEAR_PROFILE:
        require(algorithm == SIGNED_ALGORITHM, "signed-null profile requires v3 construction")
    targets = _linear_targets(model, big_d)
    if hasattr(model, "_aloepri_linear_profile"):
        current = linear_profile_summary(model)
        require(current["construction_profile"] == algorithm and current["physical_hidden_size"] == big_d, "profile identity metadata mismatch")
        if current["execution_profile"] == linear_profile:
            return current
        require(current["execution_profile"] == NATIVE_LINEAR_PROFILE and linear_profile != NATIVE_LINEAR_PROFILE, "profile cannot silently switch/fallback")
    _validate_linear_targets(targets, NATIVE_LINEAR_PROFILE)
    originals = tuple(getattr(parent, member).weight for _, parent, member, _, _ in targets)
    state_keys = tuple(model.state_dict())
    if linear_profile == SIGNED_LINEAR_PROFILE:
        active, _ = signed_null_support(model.model.norm._p, model.model.norm._q)
        replacements = [(parent, member, _profile_implementation(linear_profile, member)(getattr(parent, member).weight, model.config.hidden_size, active))
                        for _, parent, member, _, _ in targets]
    else:
        replacements = [(parent, member, F64AccumLinear.from_linear(getattr(parent, member)))
                        for _, parent, member, _, _ in targets] if linear_profile == ALL_BOUNDARY_LINEAR_PROFILE else []
    for parent, member, replacement in replacements:
        replacement.train(getattr(parent, member).training)
        setattr(parent, member, replacement)
    model._aloepri_linear_profile = linear_profile
    model._aloepri_construction_algorithm = algorithm
    model._aloepri_physical_hidden_size = big_d
    model._aloepri_linear_parameters = originals
    require(tuple(model.state_dict()) == state_keys, "profile changed state_dict keys")
    return linear_profile_summary(model)


def linear_profile_costs(model):
    installed = linear_profile_summary(model)
    costs = []
    for name, parent, member, _, _ in _linear_targets(model, model._aloepri_physical_hidden_size):
        module = getattr(parent, member)
        if type(module) is F64AccumLinear:
            costs.append({"name": name, "calls": module.calls, "elapsed_seconds": module.elapsed_seconds,
                          "peak_f64_temporary_bytes": module.peak_f64_temporary_bytes,
                          "peak_working_bytes": module.peak_working_bytes, "output_row_chunk": F64_LINEAR_ROW_CHUNK})
        elif type(module) in [SignedColumnBlockedLinear, SignedRowBlockedLinear]:
            require(module.core_calls == module.aux_calls, "signed-null skipped auxiliary GEMM")
            costs.append({"name": name, "calls": module.core_calls, "core_gemm_calls": module.core_calls, "aux_gemm_calls": module.aux_calls,
                          "elapsed_seconds": module.elapsed_seconds, "peak_f64_temporary_bytes": 0, "peak_working_bytes": module.peak_f32_temporary_weight_bytes,
                          "peak_f32_core_weight_copy_bytes": module.peak_f32_temporary_weight_bytes, "physical_tile_not_logical_decode": True})
    largest = max(costs, key=lambda row: row["peak_working_bytes"], default=None)
    return {"execution_profile": installed["execution_profile"], "installed_scope_count": installed["installed_role_count"],
            "executed_scope_count": sum(row["calls"] > 0 for row in costs), "executed_roles": [row["name"] for row in costs if row["calls"]],
            "max_temporary_module": largest["name"] if largest else None,
            "peak_f64_temporary_bytes": largest["peak_f64_temporary_bytes"] if largest else 0,
            "peak_working_bytes": max((row["peak_working_bytes"] for row in costs), default=0),
            "total_linear_elapsed_seconds": sum(row["elapsed_seconds"] for row in costs), "roles": costs,
            "memory_note": "maximum simultaneously live per-Linear tensors, not sum of independent peaks or process RSS"}


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


def load_runtime(artifact_root, secret_path, *, linear_profile=NATIVE_LINEAR_PROFILE):
    require(linear_profile in SUPPORTED_LINEAR_PROFILES, "unknown Linear execution profile")
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
    install_linear_profile(model, binding["algorithm"], big_d, linear_profile=linear_profile)
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
    parser.add_argument("--linear-profile", choices=sorted(SUPPORTED_LINEAR_PROFILES), default=NATIVE_LINEAR_PROFILE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, _, diagnostics = load_runtime(args.artifact, args.secret, linear_profile=args.linear_profile)
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
    print(json.dumps({"mode": "exact_covariant", "keymat": diagnostics, "installed_linear_profile": linear_profile_summary(model), "linear_costs": linear_profile_costs(model), "logits_shape": list(logits.shape), "finite_logits": torch.isfinite(logits).all().item(), "token_ids": tokens, "text": tokenizer.decode(tokens) if tokenizer else None}))


if __name__ == "__main__":
    main()
