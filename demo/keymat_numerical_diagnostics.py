import argparse
import json
import os
import platform
import resource
import struct
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import blake3
import torch
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM

from aloepri_runtime import ALGORITHM, BALANCED_ALGORITHM, F64AccumLinear, load_keymat, load_runtime, require
from hidden_expansion_demo import compare_nodes, metrics
from token_permutation_demo import artifact_fingerprint, load_json, load_local_llama_model, read_artifact

NATIVE_PROFILE = "f32-state-native-linear-v1"
CANDIDATE_PROFILE = "f32-state-f64-linear-acc-v1"


class FirstBlockCaptured(Exception):
    pass


def tensor_summary(value):
    finite = torch.isfinite(value).all().item()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": finite,
            "max_abs_value": value.double().abs().max().item() if finite else None,
            "l2_norm": value.double().norm().item() if finite else None}


def pair_summary(expected, actual):
    result = dict(metrics(expected, actual), expected=tensor_summary(expected), actual=tensor_summary(actual))
    if result["finite"]:
        delta = actual.double() - expected.double()
        flat = delta.abs().argmax().item()
        coordinate = []
        for size in reversed(delta.shape):
            coordinate.append(flat % size)
            flat //= size
        coordinate.reverse()
        location = tuple(coordinate)
        result.update(max_error_position=coordinate, expected_at_max=expected[location].item(),
                      actual_at_max=actual[location].item(), signed_mean=delta.mean().item(), delta_l2=delta.norm().item())
    return result


def difference_chain(stages):
    values = list(stages.items())
    origin = values[0][1].double()
    reconstructed = torch.zeros_like(origin)
    components, vectors = {}, {}
    for (before_name, before), (name, after) in zip(values, values[1:]):
        require(before.shape == after.shape, "difference chain shape mismatch")
        delta = after.double() - before.double()
        vectors[name] = delta
        components[name] = dict(pair_summary(before, after), from_stage=before_name)
        reconstructed += delta
    total = values[-1][1].double() - origin
    reconstruction_error = (reconstructed - total).abs().max().item()
    return {"diagnostic_only": True, "components": components,
            "total": pair_summary(values[0][1], values[-1][1]),
            "reconstruction_max_abs": reconstruction_error,
            "scalar_maxima_are_not_additive": True}, vectors


@torch.inference_mode()
def trace_first_block(model, ids):
    nodes, internal = OrderedDict(), {}
    handles = []
    block = model.model.layers[0]

    def post(module, name, destination=nodes, tuple_output=False):
        def hook(_module, _inputs, output):
            destination[name] = (output[0] if tuple_output else output).detach().clone()
        handles.append(module.register_forward_hook(hook))

    def pre(module, name, destination=nodes):
        def hook(_module, inputs):
            destination[name] = inputs[0].detach().clone()
        handles.append(module.register_forward_pre_hook(hook))

    post(model.model.embed_tokens, "embedding")
    pre(block, "layer.0.input")
    post(block.input_layernorm, "layer.0.input_norm")
    post(block.self_attn, "layer.0.attention_output", tuple_output=True)
    pre(block.post_attention_layernorm, "layer.0.post_attention_residual")
    post(block.post_attention_layernorm, "layer.0.post_attention_norm")
    post(block.mlp, "layer.0.ffn_output")
    post(block, "layer.0.block_output")
    post(block.mlp.gate_proj, "gate", internal)
    post(block.mlp.up_proj, "up", internal)
    pre(block.mlp.down_proj, "down_input", internal)

    def stop(_module, _inputs, _output):
        raise FirstBlockCaptured()
    handles.append(block.register_forward_hook(stop))
    try:
        model(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long), use_cache=False)
        raise ValueError("first block stop hook did not fire")
    except FirstBlockCaptured:
        internal["silu"] = block.mlp.act_fn(internal["gate"])
        internal["product"] = internal["silu"] * internal["up"]
        require(torch.equal(internal["product"], internal["down_input"]), "trace does not match pinned SwiGLU forward")
        require(all(value.dtype == torch.float32 for value in list(nodes.values()) + list(internal.values())), "non-F32 module boundary")
        return nodes, internal
    finally:
        for handle in handles:
            handle.remove()


def ulp_distance(expected, actual):
    require(expected.dtype == actual.dtype == torch.float32, "ULP comparison requires F32")
    require(expected.shape == actual.shape and torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item(), "invalid ULP inputs")
    def ordered_bits(value):
        bits = value.contiguous().view(torch.int32).to(torch.int64)
        return torch.where(bits < 0, -2147483648 - bits, bits)
    distances = (ordered_bits(actual) - ordered_bits(expected)).abs()
    return {"max_ulp": distances.max().item(), "different_elements": torch.count_nonzero(distances).item(),
            "total_elements": distances.numel(), "bitwise_equal": torch.equal(expected.view(torch.int32), actual.view(torch.int32))}


@torch.inference_mode()
def weight_oracle(baseline, runtime, p, q):
    results, theoretical = {}, {}
    for name in ["gate_proj", "up_proj", "down_proj"]:
        original = getattr(baseline.model.layers[0].mlp, name).weight
        stored = getattr(runtime.model.layers[0].mlp, name).weight
        require(original.dtype == stored.dtype == torch.float32, "weight oracle requires F32 storage")
        ideal = p.T @ original.double() if name == "down_proj" else original.double() @ q.T
        require(ideal.shape == stored.shape, "wrong transformed weight geometry")
        theoretical[name] = ideal
        results[name] = {"diagnostic_only": True, "storage_rounding": pair_summary(ideal, stored.double()),
                         "f32_encoding": dict(pair_summary(ideal.float(), stored), **ulp_distance(ideal.float(), stored)),
                         "stored_blake3": blake3.blake3(stored.detach().numpy().tobytes()).hexdigest()}
    return results, theoretical


@torch.inference_mode()
def input_linear_chain(x_base, z_run, original, stored, theoretical, p):
    z_ideal = x_base.double() @ p
    z_boundary = z_ideal.float()
    source_native = F.linear(x_base, original)
    source64 = F.linear(x_base.double(), original.double())
    theory64 = F.linear(z_ideal, theoretical)
    input_boundary64 = F.linear(z_boundary.double(), theoretical)
    storage64 = F.linear(z_boundary.double(), stored.double())
    output_boundary = storage64.float()
    native_same = F.linear(z_boundary, stored)
    native_actual = F.linear(z_run, stored)
    stages = OrderedDict(source_native=source_native, source_reference64=source64,
                         covariance64=theory64, expanded_input_boundary=input_boundary64,
                         storage_rounding=storage64, output_boundary=output_boundary,
                         native_arithmetic_same_input=native_same, upstream_actual_input=native_actual)
    report, vectors = difference_chain(stages)
    actual64 = F.linear(z_run.double(), stored.double())
    report["actual_input_f64_vs_native"] = pair_summary(actual64.float(), native_actual)
    return report, vectors


@torch.inference_mode()
def down_linear_chain(u_base, u_run, original, stored, theoretical, q):
    recover = lambda value: value.double() @ q
    source_native = F.linear(u_base, original)
    source64 = F.linear(u_base.double(), original.double())
    ideal64 = F.linear(u_base.double(), theoretical)
    stored64 = F.linear(u_base.double(), stored.double())
    stored_actual64 = F.linear(u_run.double(), stored.double())
    native_same = F.linear(u_base, stored)
    native_actual = F.linear(u_run, stored)
    stages = OrderedDict(source_native=source_native, source_reference64=source64,
                         covariance64=recover(ideal64), storage_rounding=recover(stored64),
                         output_boundary=recover(stored64.float()), native_arithmetic_same_input=recover(native_same),
                         upstream_actual_input=recover(native_actual))
    report, vectors = difference_chain(stages)
    report["actual_input_storage64_recovered"] = pair_summary(source_native, recover(stored_actual64))
    report["actual_input_output32_recovered"] = pair_summary(source_native, recover(stored_actual64.float()))
    report["actual_input_native_recovered"] = pair_summary(source_native, recover(native_actual))
    report["input_drift"] = pair_summary(u_base, u_run)
    return report, vectors


def state_summary(model, big_d):
    state = model.state_dict()
    parameters_f32 = all(value.dtype == torch.float32 for value in model.parameters())
    state_f32 = all(value.dtype == torch.float32 for value in state.values())
    head_independent = model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    require(parameters_f32 and state_f32 and head_independent, "invalid runtime F32 state or retied head")
    require(model.model.embed_tokens.embedding_dim == big_d, "invalid expanded embedding")
    return {"parameters_f32": parameters_f32, "state_dict_f32": state_f32, "physical_head_independent": head_independent}


def provenance():
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    hashes = {name: blake3.blake3((root / "demo" / name).read_bytes()).hexdigest()
              for name in ["aloepri_runtime.py", "hidden_expansion_demo.py", "keymat_numerical_diagnostics.py"]}
    return {"runtime_base_commit": head, "working_tree_dirty": bool(dirty), "executed_code_blake3": hashes,
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cpu": platform.machine(), "torch_threads": torch.get_num_threads(), "device": "cpu", "attention": "eager",
            "baseline_profile": NATIVE_PROFILE}


def check_replay_identity(source, artifact, secret_path, fixture_path, failure_report):
    manifest = load_json(artifact / "aloepri.json")
    fixture = load_json(fixture_path)
    failure = load_json(failure_report)
    require(failure["G1"]["status"] == "pass" and failure["G2"]["status"] == "fail", "not the original failing replay")
    require(fixture == failure["fixture"], "fixture differs from original failure")
    for field in ["source_fingerprint", "secret_id"]:
        require(manifest[field] == failure[field], "replay identity differs from original failure")
    config, tensors = read_artifact(source)
    require(all(tensor[0] == "F32" for tensor in tensors.values()), "SOURCE is not F32")
    require(artifact_fingerprint(source, config, tensors) == manifest["source_fingerprint"], "SOURCE fingerprint mismatch")
    binding = manifest["plan"]["keymat_binding"]
    require(fixture["h"] == binding["expansion_size"] and struct.unpack("<Q", struct.pack("<d", fixture["lambda"]))[0] == binding["lambda_bits"], "fixture parameters mismatch")
    require(load_json(secret_path)["master_seed"] == blake3.blake3(struct.pack("<Q", fixture["fixture_seed"])).hexdigest(), "fixture seed mismatch")
    snapshot = {"source_fingerprint": manifest["source_fingerprint"], "secret_id": manifest["secret_id"],
                "plan_hash": manifest["plan_hash"], "layout_hash": manifest["layout_hash"],
                "fixture_blake3": blake3.blake3(fixture_path.read_bytes()).hexdigest(),
                "original_failure_blake3": blake3.blake3(failure_report.read_bytes()).hexdigest(),
                "artifact_tensor_digests": {item["name"]: item["blake3"] for item in manifest["tensors"]}}
    return fixture, failure, snapshot


def check_construction_trial_identity(source, artifact, secret_path, fixture_path, failure_report, control_secret_path, control_diagnostic_path):
    manifest = load_json(artifact / "aloepri.json")
    fixture = load_json(fixture_path)
    failure = load_json(failure_report)
    control = load_json(control_diagnostic_path)
    require(failure["G1"]["status"] == "pass" and failure["G2"]["status"] == "fail", "missing historical failing control")
    require(fixture == failure["fixture"] and blake3.blake3(fixture_path.read_bytes()).hexdigest() == control["identity"]["fixture_blake3"], "construction trial changed fixture bytes")
    secret = load_json(secret_path)
    original = load_json(control_secret_path)
    require(secret["algorithm"] == BALANCED_ALGORITHM and original["algorithm"] == ALGORITHM, "construction trial requires exact v2 and v1 control algorithms")
    for field in ["source_fingerprint", "hidden_size", "expansion_size", "lambda_bits", "master_seed", "rng", "nullspace_cutoff"]:
        require(secret[field] == original[field], "construction trial changed fixed generation inputs")
    require(original["secret_id"] == failure["secret_id"] == control["identity"]["secret_id"] and manifest["source_fingerprint"] == failure["source_fingerprint"] == control["identity"]["source_fingerprint"], "historical control identity mismatch")
    require(manifest["secret_id"] == secret["secret_id"] and secret["secret_id"] != original["secret_id"], "construction trial must have a new Secret identity")
    require(secret["p_digest"] != original["p_digest"] and secret["q_digest"] != original["q_digest"], "real construction trial did not change P/Q")
    binding = manifest["plan"]["keymat_binding"]
    require(binding["algorithm"] == BALANCED_ALGORITHM and fixture["h"] == binding["expansion_size"] and struct.unpack("<Q", struct.pack("<d", fixture["lambda"]))[0] == binding["lambda_bits"], "construction trial binding/parameters mismatch")
    require(secret["master_seed"] == blake3.blake3(struct.pack("<Q", fixture["fixture_seed"])).hexdigest(), "construction trial seed mismatch")
    require(manifest["plan_hash"] != control["identity"]["plan_hash"], "construction trial reused old plan identity")
    config, tensors = read_artifact(source)
    require(all(tensor[0] == "F32" for tensor in tensors.values()), "construction SOURCE must be F32")
    require(artifact_fingerprint(source, config, tensors) == manifest["source_fingerprint"], "construction SOURCE fingerprint mismatch")
    snapshot = {"source_fingerprint": manifest["source_fingerprint"], "secret_id": manifest["secret_id"],
                "algorithm": secret["algorithm"], "plan_hash": manifest["plan_hash"], "layout_hash": manifest["layout_hash"],
                "fixture_blake3": blake3.blake3(fixture_path.read_bytes()).hexdigest(),
                "historical_secret_id": original["secret_id"], "historical_plan_hash": control["identity"]["plan_hash"],
                "historical_failure_blake3": blake3.blake3(failure_report.read_bytes()).hexdigest(),
                "artifact_tensor_digests": {item["name"]: item["blake3"] for item in manifest["tensors"]}}
    return fixture, failure, snapshot


@torch.inference_mode()
def diagnose(baseline, runtime, q, fixture, failure, private_vectors, construction_trial=False):
    p = runtime.model.norm._p
    big_d = q.shape[0]
    weights, theoretical = weight_oracle(baseline, runtime, p, q)
    report = {"diagnostic_only": True, "scope": "layer.0", "execution_profile": NATIVE_PROFILE,
              "G1": "pass", "formal_G2": "not_run", "G3": "not_run", "G4": "not_run",
              "state": state_summary(runtime, big_d), "complete_first_ffn_weight_oracle": weights, "prompts": []}
    for index, prompt in enumerate(fixture["prompts"]):
        ids = prompt + fixture["continuation_ids"]
        baseline_nodes, baseline_internal = trace_first_block(baseline, ids)
        runtime_nodes, runtime_internal = trace_first_block(runtime, ids)
        for value in runtime_nodes.values():
            require(value.shape[-1] == big_d and value.dtype == torch.float32, "non-expanded residual node")
        layerwise = compare_nodes(baseline_nodes, runtime_nodes, q)
        observed = next(item for item in layerwise["nodes"] if item["node"] == "layer.0.ffn_output")
        original = next(item for item in failure["G2"]["prompts"][index]["nodes"] if item["node"] == "layer.0.ffn_output")
        replay_delta = abs(observed["max_abs"] - original["max_abs"])
        if not construction_trial:
            require(replay_delta <= 1e-12, "native first-layer replay differs from failure snapshot")
        internal = {name: pair_summary(baseline_internal[name], runtime_internal[name]) for name in ["gate", "up", "silu", "product", "down_input"]}
        row = {"prompt_index": index, "sequence_length": len(ids), "native_first_block": layerwise,
               "native_failure_replay_delta": replay_delta, "internal": internal,
               "first_internal_over_tolerance": next((name for name in ["gate", "up", "silu", "product", "down_input"] if not internal[name]["pass"]), None),
               "input_linear_chains": {}}
        if construction_trial:
            del row["native_failure_replay_delta"]
            row["historical_control_ffn_max_abs"] = original["max_abs"]
            row["different_construction_identity"] = True
        for name in ["gate_proj", "up_proj"]:
            chain, vectors = input_linear_chain(baseline_nodes["layer.0.post_attention_norm"], runtime_nodes["layer.0.post_attention_norm"],
                                                getattr(baseline.model.layers[0].mlp, name).weight,
                                                getattr(runtime.model.layers[0].mlp, name).weight, theoretical[name], p)
            row["input_linear_chains"][name] = chain
            private_vectors[f"prompt.{index}.{name}"] = vectors
        row["down_chain"], vectors = down_linear_chain(baseline_internal["down_input"], runtime_internal["down_input"],
                                                      baseline.model.layers[0].mlp.down_proj.weight,
                                                      runtime.model.layers[0].mlp.down_proj.weight, theoretical["down_proj"], q)
        private_vectors[f"prompt.{index}.down_proj"] = vectors
        report["prompts"].append(row)
    if construction_trial:
        require(not any(isinstance(module, F64AccumLinear) for module in runtime.modules()), "construction trial must use native F32 Linear")
        report["construction_profile"] = BALANCED_ALGORITHM
        report["first_block_trial_pass"] = all(item["native_first_block"]["status"] == "pass" for item in report["prompts"])
        report["decision"] = "eligible_for_full_model_evaluation" if report["first_block_trial_pass"] else "stop_construction_local_gate_failed"
        report["historical_control_only_not_same_identity_replay"] = True
    return report


@torch.inference_mode()
def construction_norm_comparison(runtime, diagnostics, control_secret_path):
    binding, p1, q1, control = load_keymat(control_secret_path)
    require(binding["algorithm"] == ALGORITHM, "norm control must be v1")
    p2, q2 = runtime.model.norm._p, runtime.model.norm._q
    require(p1.shape == p2.shape and q1.shape == q2.shape, "norm comparison changed dimensions")
    p_gram = p1 @ p1.T - p2 @ p2.T
    q_gram = q1.T @ q1 - q2.T @ q2
    p_min = torch.linalg.eigvalsh((p_gram + p_gram.T) * 0.5).min().item()
    q_min = torch.linalg.eigvalsh((q_gram + q_gram.T) * 0.5).min().item()
    require(p_min >= -1e-9 and q_min >= -1e-9, "fixed construction violates Gram PSD relation")
    require(diagnostics["p_spectral_norm"] <= control["p_spectral_norm"] + 1e-9 and diagnostics["q_spectral_norm"] <= control["q_spectral_norm"] + 1e-9, "fixed construction increased spectral norms")
    return {"diagnostic_only": True, "control_v1": control, "trial_v2": diagnostics,
            "control_norm_product": control["p_spectral_norm"] * control["q_spectral_norm"],
            "trial_norm_product": diagnostics["p_spectral_norm"] * diagnostics["q_spectral_norm"],
            "p_gram_difference_min_eigenvalue": p_min, "q_gram_difference_min_eigenvalue": q_min,
            "coefficient_std_rule": "d^-1/2", "coefficient_std": 1.0 / p2.shape[0] ** 0.5,
            "not_a_g2_error_bound_or_condition_monotonicity_claim": True}


def local_candidate_passes(profile):
    return (profile["state"]["parameters_f32"] and profile["state"]["state_dict_f32"]
            and profile["state"]["physical_head_independent"] and profile["weight_bits_unchanged"]
            and bool(profile["prompts"]) and all(item["status"] == "pass" for item in profile["prompts"]))


@torch.inference_mode()
def evaluate_candidates(baseline, runtime, q, fixture, failure):
    from aloepri_runtime import F64AccumLinear, F64_LINEAR_ROW_CHUNK
    mlp = runtime.model.layers[0].mlp
    originals = {name: getattr(mlp, name) for name in ["gate_proj", "up_proj", "down_proj"]}
    original_digests = {name: blake3.blake3(module.weight.detach().numpy().tobytes()).hexdigest() for name, module in originals.items()}
    reference = [trace_first_block(baseline, prompt + fixture["continuation_ids"])[0] for prompt in fixture["prompts"]]
    profiles = {}
    try:
        for name, members in [("native_control", ()), ("down_only", ("down_proj",)), ("gate_up_down", tuple(originals))]:
            for member, module in originals.items():
                setattr(mlp, member, F64AccumLinear.from_linear(module) if member in members else module)
            started = time.perf_counter()
            profile = {"diagnostic_only": True, "execution_profile": CANDIDATE_PROFILE if members else NATIVE_PROFILE,
                       "scope": list(members), "state": state_summary(runtime, q.shape[0]), "prompts": []}
            for index, (prompt, baseline_nodes) in enumerate(zip(fixture["prompts"], reference)):
                nodes, internal = trace_first_block(runtime, prompt + fixture["continuation_ids"])
                require(all(value.shape[-1] == q.shape[0] and value.dtype == torch.float32 for value in nodes.values()), "candidate residual boundary changed")
                compared = compare_nodes(baseline_nodes, nodes, q)
                ffn_node = next(item for item in compared["nodes"] if item["node"] == "layer.0.ffn_output")
                if not members:
                    expected = next(item for item in failure["G2"]["prompts"][index]["nodes"] if item["node"] == "layer.0.ffn_output")
                    require(abs(ffn_node["max_abs"] - expected["max_abs"]) <= 1e-12, "candidate control does not replay original failure")
                compared["prompt_index"] = index
                compared["first_ffn"] = ffn_node
                profile["prompts"].append(compared)
            profile["weight_bits_unchanged"] = all(blake3.blake3(getattr(mlp, member).weight.detach().numpy().tobytes()).hexdigest() == digest for member, digest in original_digests.items())
            require(profile["weight_bits_unchanged"], "candidate mutated physical weights")
            profile["candidate_linear_costs"] = {member: {"output_row_chunk": F64_LINEAR_ROW_CHUNK,
                "calls": getattr(mlp, member).calls, "peak_f64_temporary_bytes": getattr(mlp, member).peak_f64_temporary_bytes,
                "peak_working_bytes": getattr(mlp, member).peak_working_bytes, "elapsed_seconds": getattr(mlp, member).elapsed_seconds}
                for member in members}
            profile["elapsed_seconds"] = time.perf_counter() - started
            profile["process_highwater_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            profile["local_gate_pass"] = local_candidate_passes(profile)
            profiles[name] = profile
    finally:
        for name, module in originals.items():
            setattr(mlp, name, module)
    passed = profiles["gate_up_down"]["local_gate_pass"]
    return {"diagnostic_only": True, "scope": "layer.0", "execution_profile": CANDIDATE_PROFILE,
            "fixed_output_row_chunk": F64_LINEAR_ROW_CHUNK, "profiles": profiles,
            "default_loader_unchanged": True, "candidate_local_gate_pass": passed,
            "decision": "eligible_for_full_model_evaluation" if passed else "stop_local_candidate_failed",
            "full_model_G2": "not_run", "G3": "not_run", "G4": "not_run",
            "rss_note": "process high-water includes models/oracles and is cumulative; tensor working bytes are counted separately"}


def main():
    parser = argparse.ArgumentParser(description="Bounded layer-0 diagnostic oracle; never a formal equivalence gate")
    for name in ["source", "artifact", "secret", "fixture", "failure-report", "report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--private-error-vectors", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate-candidate", action="store_true")
    mode.add_argument("--construction-trial", action="store_true")
    parser.add_argument("--diagnostic-input", type=Path)
    parser.add_argument("--control-secret", type=Path)
    parser.add_argument("--control-diagnostics", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    require(not args.report.exists(), "diagnostic report already exists")
    if args.evaluate_candidate:
        require(args.diagnostic_input is not None, "candidate evaluation requires prior diagnostics")
    else:
        require(args.private_error_vectors is not None and not args.private_error_vectors.exists(), "private error-vector output required and must not exist")
        for model_root in [args.source.resolve(), args.artifact.resolve()]:
            require(not args.private_error_vectors.resolve().is_relative_to(model_root), "private vectors must stay outside model artifacts")
    if args.construction_trial:
        require(args.control_secret is not None and args.control_diagnostics is not None, "construction trial requires original control inputs")
        fixture, failure, snapshot = check_construction_trial_identity(args.source, args.artifact, args.secret, args.fixture, args.failure_report, args.control_secret, args.control_diagnostics)
    else:
        fixture, failure, snapshot = check_replay_identity(args.source, args.artifact, args.secret, args.fixture, args.failure_report)
    if args.evaluate_candidate:
        prior = load_json(args.diagnostic_input)
        require(prior["identity"] == snapshot and prior["diagnostic_only"] and prior["candidate_evaluation"] == "not_run", "candidate input does not match native diagnostic snapshot")
        require(all(item["native_failure_replay_delta"] <= 1e-12 for item in prior["prompts"]), "native failure was not replayed")
    baseline, _ = load_local_llama_model(args.source, LlamaForCausalLM, torch)
    runtime, q, keymat = load_runtime(args.artifact, args.secret)
    if args.evaluate_candidate:
        report = evaluate_candidates(baseline, runtime, q, fixture, failure)
        report["native_diagnostic_blake3"] = blake3.blake3(args.diagnostic_input.read_bytes()).hexdigest()
    else:
        private_vectors = {}
        report = diagnose(baseline, runtime, q, fixture, failure, private_vectors, construction_trial=args.construction_trial)
        if args.construction_trial:
            report["key_norm_comparison"] = construction_norm_comparison(runtime, keymat, args.control_secret)
        descriptor = os.open(args.private_error_vectors, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(private_vectors, handle)
        report.update(private_error_vectors_retained=True, candidate_evaluation="not_run")
    report.update(identity=snapshot, provenance=provenance(), keymat=keymat,
                  elapsed_seconds=time.perf_counter() - started, process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"diagnostic_only": True, "scope": report["scope"], "decision": report.get("decision", "native_replay_matched")}))
    failed = (args.evaluate_candidate and not report["candidate_local_gate_pass"]) or (args.construction_trial and not report["first_block_trial_pass"])
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
