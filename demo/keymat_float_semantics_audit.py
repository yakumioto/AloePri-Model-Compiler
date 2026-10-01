import argparse
import copy
import json
import os
import resource
import struct
import time
from pathlib import Path

import blake3
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from aloepri_runtime import (ALL_BOUNDARY_LINEAR_PROFILE, BALANCED_ALGORITHM, F64AccumLinear,
    F64_LINEAR_ROW_CHUNK, NATIVE_LINEAR_PROFILE, install_linear_profile, linear_profile_summary,
    load_runtime, require)
from hidden_expansion_demo import compare_nodes, metrics
from keymat_numerical_diagnostics import (difference_chain, pair_summary, provenance, trace_first_block)
from token_permutation_demo import artifact_fingerprint, load_json, load_local_llama_model, read_artifact

ROLE_MEMBERS = [("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
                ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj")]
PATH_DEFINITIONS = {
    "A": {"kind": "native_source_target", "hidden_space": "logical_d", "parameter_storage": "F32_SOURCE",
          "linear_arithmetic": "native_F32", "boundary_dtype": "F32", "admissible_acceptance_target": True},
    "B": {"kind": "diagnostic_only_source_arithmetic_reference", "hidden_space": "logical_d", "parameter_storage": "independent_first_block_F32_SOURCE",
          "linear_arithmetic": "temporary_F64_then_F32", "boundary_dtype": "F32", "admissible_acceptance_target": False},
    "C": {"kind": "actual_failed_v2_runtime", "hidden_space": "physical_D", "parameter_storage": "F32_ARTIFACT",
          "linear_arithmetic": "temporary_F64_then_F32", "boundary_dtype": "F32", "admissible_acceptance_target": False},
    "D": {"kind": "diagnostic_only_ideal_coefficient_reference", "hidden_space": "physical_D", "parameter_storage": "private_F64_theory_coefficients_not_an_artifact",
          "linear_arithmetic": "temporary_F64_then_F32", "boundary_dtype": "F32", "admissible_acceptance_target": False},
}


class IdealCoefficientLinear(torch.nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        require(coefficient.ndim == 2 and coefficient.dtype == torch.float64 and coefficient.device.type == "cpu" and torch.isfinite(coefficient).all().item(), "invalid private ideal coefficient")
        self.register_buffer("_audit_coefficient", coefficient, persistent=False)
        self.out_features, self.in_features = coefficient.shape
        self.peak_working_bytes = 0
        self.calls = 0

    @torch.inference_mode()
    def forward(self, inputs):
        require(inputs.dtype == torch.float32 and inputs.device.type == "cpu" and inputs.shape[-1] == self.in_features, "ideal reference requires F32 input boundary")
        output = torch.empty((*inputs.shape[:-1], self.out_features), dtype=torch.float32)
        inputs64 = inputs.double()
        for start in range(0, self.out_features, F64_LINEAR_ROW_CHUNK):
            end = min(start + F64_LINEAR_ROW_CHUNK, self.out_features)
            coefficient = self._audit_coefficient[start:end]
            result64 = F.linear(inputs64, coefficient, None)
            result32 = result64.float()
            require(torch.isfinite(result64).all().item() and torch.isfinite(result32).all().item(), "non-finite ideal reference output")
            output[..., start:end].copy_(result32)
            self.peak_working_bytes = max(self.peak_working_bytes, inputs64.numel() * 8 + coefficient.numel() * 8 + result64.numel() * 8 + result32.numel() * 4 + output.numel() * 4)
            del result64, result32
        self.calls += 1
        return output


class AuditFirstBlockReference(torch.nn.Module):
    def __init__(self, original, block, kind):
        super().__init__()
        prefix = copy.copy(original.model)
        # Separate the module registry before replacing its layer list.
        prefix._modules = original.model._modules.copy()
        prefix.layers = torch.nn.ModuleList([block])
        self.model = prefix
        self.config = original.config
        self._audit_reference_kind = kind
        self.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def clone_first_block(model):
    block = model.model.layers[0]
    memo = {id(model.config): model.config}
    for norm in [block.input_layernorm, block.post_attention_layernorm]:
        for name in ["_p", "_q"]:
            if hasattr(norm, name):
                value = getattr(norm, name)
                memo[id(value)] = value
    return copy.deepcopy(block, memo)


@torch.inference_mode()
def build_references(source, actual, p, q):
    source_block = clone_first_block(source)
    expanded_block = clone_first_block(actual)
    ideal_bytes = 0
    roles = []
    for section, member in ROLE_MEMBERS:
        logical = getattr(getattr(source.model.layers[0], section), member)
        logical_copy = getattr(getattr(source_block, section), member)
        physical = getattr(getattr(actual.model.layers[0], section), member)
        require(logical.bias is None and physical.bias is None and logical.weight.dtype == physical.weight.dtype == torch.float32, "audit role requires original F32 weights/no bias")
        setattr(getattr(source_block, section), member, F64AccumLinear.from_linear(logical_copy))
        coefficient = p.T @ logical.weight.double() if member in ["o_proj", "down_proj"] else logical.weight.double() @ q.T
        require(coefficient.shape == physical.weight.shape, "ideal reference role direction/shape mismatch")
        ideal_bytes += coefficient.numel() * 8
        setattr(getattr(expanded_block, section), member, IdealCoefficientLinear(coefficient))
        roles.append({"name": f"model.layers.0.{section}.{member}", "ideal_shape": list(coefficient.shape), "coefficient_dtype": "torch.float64"})
    for name in ["input_layernorm", "post_attention_layernorm"]:
        require(torch.equal(getattr(source.model.layers[0], name).weight, getattr(actual.model.layers[0], name).weight), "SOURCE/artifact logical norm bytes differ")
    logical_ref = AuditFirstBlockReference(source, source_block, "B_SOURCE_F64_LINEAR_REFERENCE")
    ideal_ref = AuditFirstBlockReference(actual, expanded_block, "D_IDEAL_F64_COEFFICIENT_REFERENCE")
    require(logical_ref.model.embed_tokens is source.model.embed_tokens and ideal_ref.model.embed_tokens is actual.model.embed_tokens, "reference changed embedding source")
    require(logical_ref.model.rotary_emb is source.model.rotary_emb and ideal_ref.model.rotary_emb is actual.model.rotary_emb, "reference changed RoPE implementation")
    return logical_ref, ideal_ref, {"independent_first_blocks": 2, "no_full_model_shadow": True,
                                  "ideal_coefficient_count": len(roles), "resident_private_ideal_coefficient_bytes": ideal_bytes,
                                  "D_embedding_is_actual_artifact": True, "logical_norm_weights_bitwise_equal": True, "roles": roles}


def normal_snapshot(model):
    return {"parameter_ids": {name: id(value) for name, value in model.named_parameters()},
            "state": {name: blake3.blake3(value.detach().numpy().tobytes()).hexdigest() for name, value in model.state_dict().items()},
            "module_types": {name: type(module).__name__ for name, module in model.named_modules()},
            "buffers": {name: {"id": id(value), "hash": blake3.blake3(value.detach().numpy().tobytes()).hexdigest()} for name, value in model.named_buffers()},
            "profile_metadata": getattr(model, "_aloepri_linear_profile", None)}


def audit_identity(source, artifact, secret_path, fixture_path, reference_path):
    reference = load_json(reference_path)
    require(reference.get("diagnostic_only") is True and reference.get("construction_profile") == BALANCED_ALGORITHM, "audit requires latest balanced-v2 profile evidence")
    require(reference.get("execution_profile") == ALL_BOUNDARY_LINEAR_PROFILE and reference.get("candidate_local_gate_pass") is False, "audit reference is not the fixed failed all-boundary profile")
    require(all(reference.get(field) == "not_run" for field in ["full_model_G2", "G3", "G4"]), "audit reference advanced formal gates")
    require(set(reference.get("profiles", {})) == {NATIVE_LINEAR_PROFILE, ALL_BOUNDARY_LINEAR_PROFILE}, "audit reference has wrong profile set")
    native, candidate = reference["profiles"][NATIVE_LINEAR_PROFILE], reference["profiles"][ALL_BOUNDARY_LINEAR_PROFILE]
    require(native["execution_profile"] == NATIVE_LINEAR_PROFILE and candidate["execution_profile"] == ALL_BOUNDARY_LINEAR_PROFILE and candidate["local_gate_pass"] is False, "audit profile strategies mismatch")
    require(candidate["installed_linear_profile"]["wrapped_role_count"] == candidate["installed_scope_count"], "audit reference has partial installed profile")
    require(any(row["status"] == "fail" for row in candidate["prompts"]), "audit reference has no numerical failure")
    manifest = load_json(artifact / "aloepri.json")
    secret = load_json(secret_path)
    fixture = load_json(fixture_path)
    require(secret["algorithm"] == BALANCED_ALGORITHM and manifest["plan"]["keymat_binding"]["algorithm"] == BALANCED_ALGORITHM, "audit requires unchanged v2 construction")
    identity = reference["identity"]
    for field in ["source_fingerprint", "secret_id", "plan_hash", "layout_hash"]:
        require(manifest[field] == identity[field], "audit changed material identity")
    require(secret["secret_id"] == identity["secret_id"] and secret["source_fingerprint"] == identity["source_fingerprint"], "audit changed Secret")
    require({item["name"]: item["blake3"] for item in manifest["tensors"]} == identity["artifact_tensor_digests"], "audit tensor digests changed")
    require(blake3.blake3(fixture_path.read_bytes()).hexdigest() == identity["fixture_blake3"], "audit fixture bytes changed")
    require(fixture["prompts"] and fixture["continuation_ids"], "audit fixture has empty sequences")
    for profile in [native, candidate]:
        require(len(profile["prompts"]) == len(fixture["prompts"]) and [row["prompt_index"] for row in profile["prompts"]] == list(range(len(fixture["prompts"]))), "audit sequence inventory differs")
    require(secret["expansion_size"] == fixture["h"] and secret["lambda_bits"] == struct.unpack("<Q", struct.pack("<d", fixture["lambda"]))[0], "audit generation parameters changed")
    require(secret["master_seed"] == blake3.blake3(struct.pack("<Q", fixture["fixture_seed"])).hexdigest(), "audit generation seed changed")
    config, tensors = read_artifact(source)
    require(all(tensor[0] == "F32" for tensor in tensors.values()) and artifact_fingerprint(source, config, tensors) == identity["source_fingerprint"], "audit SOURCE/dtype/fingerprint mismatch")
    return fixture, reference, dict(identity, reference_profile_blake3=blake3.blake3(reference_path.read_bytes()).hexdigest())


def assert_replay(comparison, expected):
    require([row["node"] for row in comparison["nodes"]] == [row["node"] for row in expected["nodes"]], "audit replay nodes differ")
    require(all(row["finite"] and old["finite"] and row["expanded_shape"] == old["expanded_shape"] and abs(row["max_abs"] - old["max_abs"]) <= 1e-12 for row, old in zip(comparison["nodes"], expected["nodes"])), "audit replay numeric data differs from reference")


def audit_error_budget(a, b, c_hat, d_hat):
    require(a.shape == b.shape == c_hat.shape == d_hat.shape, "audit vector shape mismatch")
    require(all(torch.isfinite(value).all().item() for value in [a, b, c_hat, d_hat]), "audit vector contains non-finite values")
    stages = {"A_native_target": a.double(), "B_source_arithmetic": b.double(),
              "D_expanded_representation": d_hat.double(), "C_actual_weight_encoding": c_hat.double()}
    chain, components = difference_chain(stages)
    observed = c_hat.double() - a.double()
    require(chain["reconstruction_max_abs"] <= 1e-12, "audit error-vector reconstruction failed")
    position = tuple(chain["total"]["max_error_position"])
    signed = {name: value[position].item() for name, value in components.items()}
    chain["actual_C_worst_coordinate"] = list(position)
    chain["same_coordinate_signed_components"] = signed
    chain["same_coordinate_observed_C_minus_A"] = observed[position].item()
    chain["same_coordinate_sum"] = sum(signed.values())
    chain["same_coordinate_reconstruction_error"] = abs(sum(signed.values()) - observed[position].item())
    chain["comparisons"] = {"A_vs_B": pair_summary(a, b), "A_vs_D": pair_summary(a, d_hat),
                            "B_vs_D": pair_summary(b, d_hat), "C_vs_D": pair_summary(d_hat, c_hat),
                            "A_vs_C": pair_summary(a, c_hat)}
    components["observed_C_minus_A"] = observed
    return chain, components


@torch.inference_mode()
def execute_audit(baseline, actual, q, fixture, reference, private_vectors):
    native_rows = []
    a_traces = []
    require(linear_profile_summary(actual)["execution_profile"] == NATIVE_LINEAR_PROFILE, "audit native control must be unmodified")
    for index, prompt in enumerate(fixture["prompts"]):
        ids = prompt + fixture["continuation_ids"]
        a_nodes, a_internal = trace_first_block(baseline, ids)
        native_nodes, _ = trace_first_block(actual, ids)
        compared = compare_nodes(a_nodes, native_nodes, q)
        assert_replay(compared, reference["profiles"][NATIVE_LINEAR_PROFILE]["prompts"][index])
        native_rows.append(dict(compared, prompt_index=index))
        a_traces.append((a_nodes, a_internal))
    install_linear_profile(actual, BALANCED_ALGORITHM, q.shape[0], linear_profile=ALL_BOUNDARY_LINEAR_PROFILE)
    before_a, before_c = normal_snapshot(baseline), normal_snapshot(actual)
    b_model, d_model, reference_resources = build_references(baseline, actual, actual.model.norm._p, q)
    require(not hasattr(b_model, "_aloepri_linear_profile") and not hasattr(d_model, "_aloepri_linear_profile"), "audit reference acquired normal profile qualification")
    report = {"audit_completed": False, "diagnostic_only": True, "ac10_satisfied": False,
              "scope": "layer.0 only; no head/later layers/logits/cache/generation", "path_definitions": PATH_DEFINITIONS,
              "formal_G2": "not_run", "G3": "not_run", "G4": "not_run", "native_control_replay": native_rows,
              "C_installed_profile": linear_profile_summary(actual), "reference_resources": reference_resources, "prompts": []}
    for index, (prompt, (a_nodes, a_internal)) in enumerate(zip(fixture["prompts"], a_traces)):
        ids = prompt + fixture["continuation_ids"]
        c_nodes, c_internal = trace_first_block(actual, ids)
        compared = compare_nodes(a_nodes, c_nodes, q)
        assert_replay(compared, reference["profiles"][ALL_BOUNDARY_LINEAR_PROFILE]["prompts"][index])
        b_nodes, b_internal = trace_first_block(b_model, ids)
        d_nodes, d_internal = trace_first_block(d_model, ids)
        require(list(a_nodes) == list(b_nodes) == list(c_nodes) == list(d_nodes), "audit reference node order changed")
        require(torch.equal(a_nodes["embedding"], b_nodes["embedding"]) and torch.equal(c_nodes["embedding"], d_nodes["embedding"]), "audit reference embedding changed")
        row = {"prompt_index": index, "sequence_length": len(ids), "actual_C_replay": compared, "required_nodes": {}, "additional_projection_diagnostics": {}}
        for name, a in a_nodes.items():
            b, c, d = b_nodes[name], c_nodes[name], d_nodes[name]
            require(all(value.dtype == torch.float32 for value in [a, b, c, d]), "audit changed F32 node boundary")
            require(a.shape[-1] == b.shape[-1] == baseline.config.hidden_size and c.shape[-1] == d.shape[-1] == q.shape[0], "audit reference hidden space mismatch")
            c_hat, d_hat = c.double() @ q, d.double() @ q
            budget, vectors = audit_error_budget(a, b, c_hat, d_hat)
            budget["boundary_shapes"] = {"A": list(a.shape), "B": list(b.shape), "C": list(c.shape), "D": list(d.shape)}
            budget["boundary_dtype"] = "torch.float32; expanded recovery view torch.float64"
            row["required_nodes"][name] = budget
            private_vectors[f"prompt.{index}.required.{name}"] = vectors
        for name in ["gate", "up", "silu", "product", "down_input"]:
            values = [a_internal[name], b_internal[name], c_internal[name], d_internal[name]]
            require(all(value.dtype == torch.float32 for value in values), "audit changed projection F32 boundary")
            budget, vectors = audit_error_budget(*values[:2], values[2].double(), values[3].double())
            budget["comparison_space"] = "logical_projection_intermediate_direct_no_Q"
            row["additional_projection_diagnostics"][name] = budget
            private_vectors[f"prompt.{index}.projection.{name}"] = vectors
        report["prompts"].append(row)
    require(normal_snapshot(baseline) == before_a and normal_snapshot(actual) == before_c, "audit references polluted A/C modules/parameters/state/buffers/profile")
    report["normal_A_C_unchanged_after_references"] = True
    report["head_and_later_layers_not_executed"] = actual.lm_head.calls == 0 and all(getattr(getattr(layer, section), member).calls == 0 for layer in actual.model.layers[1:] for section, member in ROLE_MEMBERS)
    require(report["head_and_later_layers_not_executed"], "audit escaped first-block scope")
    report["reference_resources"]["B_max_linear_temporary_working_bytes"] = max(getattr(getattr(b_model.model.layers[0], section), member).peak_working_bytes for section, member in ROLE_MEMBERS)
    report["reference_resources"]["D_max_linear_temporary_working_bytes"] = max(getattr(getattr(d_model.model.layers[0], section), member).peak_working_bytes for section, member in ROLE_MEMBERS)
    report["audit_completed"] = True
    report["supported_boundary_categories_not_fixes"] = ["native_F32_baseline_arithmetic_fidelity", "expanded_coordinate_and_F32_boundary_propagation", "F32_transformed_weight_encoding"]
    report["conclusion_rule"] = "Interpret same-coordinate signed vectors; component maxima are not additive shares, lower bounds, or unique causes. B/D references never supply AC10 evidence."
    return report


def main():
    parser = argparse.ArgumentParser(description="Bounded floating semantics audit, not a repair profile or formal gate")
    for name in ["source", "artifact", "secret", "fixture", "reference-profile", "private-error-vectors", "report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require(not args.report.exists() and not args.private_error_vectors.exists(), "audit output exists")
    for model_root in [args.source.resolve(), args.artifact.resolve()]:
        require(not args.private_error_vectors.resolve().is_relative_to(model_root), "private audit vectors must be outside model artifacts")
    started = time.perf_counter()
    fixture, reference, identity = audit_identity(args.source, args.artifact, args.secret, args.fixture, args.reference_profile)
    baseline, _ = load_local_llama_model(args.source, LlamaForCausalLM, torch)
    actual, q, keys = load_runtime(args.artifact, args.secret)
    vectors = {}
    report = execute_audit(baseline, actual, q, fixture, reference, vectors)
    report.update(identity=identity, material_G1=keys, provenance=provenance(), audit_source_blake3=blake3.blake3(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter() - started, process_highwater_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  private_error_vectors_retained=True, private_vectors_not_uploaded=True)
    descriptor = os.open(args.private_error_vectors, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        torch.save(vectors, handle)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: report[name] for name in ["audit_completed", "diagnostic_only", "ac10_satisfied", "formal_G2", "G3", "G4"]}))


if __name__ == "__main__":
    main()
