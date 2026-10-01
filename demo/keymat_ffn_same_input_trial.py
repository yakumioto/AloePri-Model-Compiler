import argparse
import copy
import json
import resource
import struct
import time
from pathlib import Path

import blake3
import torch
from transformers import LlamaForCausalLM

from aloepri_runtime import (BALANCED_ALGORITHM, F64AccumLinear, F64_LINEAR_ROW_CHUNK, NATIVE_LINEAR_PROFILE,
    SUPPORTED_LINEAR_PROFILES, linear_profile_summary, load_runtime, require)
from hidden_expansion_demo import compare_nodes
from keymat_float_semantics_audit import assert_replay, normal_snapshot
from keymat_numerical_diagnostics import pair_summary, provenance, trace_first_block
from token_permutation_demo import artifact_fingerprint, load_json, load_local_llama_model, read_artifact

PROPOSED_PROFILE = "balanced-v2-native-attn-head-f64-ffn-v1"
FFN_MEMBERS = ["gate_proj", "up_proj", "down_proj"]


def check_precheck_identity(source, artifact, secret_path, fixture_path, reference_audit):
    reference = load_json(reference_audit)
    require(reference.get("audit_completed") is True and reference.get("diagnostic_only") is True and reference.get("ac10_satisfied") is False, "precheck requires completed diagnostic-only audit")
    require(all(reference.get(field) == "not_run" for field in ["formal_G2", "G3", "G4"]), "precheck reference advanced formal gates")
    require(reference["path_definitions"]["A"]["kind"] == "native_source_target" and reference["normal_A_C_unchanged_after_references"], "precheck reference changed acceptance target or polluted models")
    manifest = load_json(artifact / "aloepri.json")
    secret = load_json(secret_path)
    fixture = load_json(fixture_path)
    identity = reference["identity"]
    require(secret["algorithm"] == BALANCED_ALGORITHM and manifest["plan"]["keymat_binding"]["algorithm"] == BALANCED_ALGORITHM, "precheck requires same balanced-v2 construction")
    for field in ["source_fingerprint", "secret_id", "plan_hash", "layout_hash"]:
        require(manifest[field] == identity[field], "precheck material identity changed")
    require(secret["secret_id"] == identity["secret_id"] and secret["source_fingerprint"] == identity["source_fingerprint"], "precheck Secret identity changed")
    require({row["name"]: row["blake3"] for row in manifest["tensors"]} == identity["artifact_tensor_digests"], "precheck artifact tensor digests changed")
    require(blake3.blake3(fixture_path.read_bytes()).hexdigest() == identity["fixture_blake3"], "precheck fixture bytes changed")
    require(len(fixture["prompts"]) == len(reference["native_control_replay"]) == len(reference["prompts"]) and fixture["continuation_ids"], "precheck sequence inventory changed")
    require([row["prompt_index"] for row in reference["native_control_replay"]] == list(range(len(fixture["prompts"]))), "precheck sequence order changed")
    require(secret["expansion_size"] == fixture["h"] and secret["lambda_bits"] == struct.unpack("<Q", struct.pack("<d", fixture["lambda"]))[0], "precheck h/lambda changed")
    require(secret["master_seed"] == blake3.blake3(struct.pack("<Q", fixture["fixture_seed"])).hexdigest(), "precheck generation seed changed")
    config, tensors = read_artifact(source)
    require(all(row[0] == "F32" for row in tensors.values()) and artifact_fingerprint(source, config, tensors) == identity["source_fingerprint"], "precheck SOURCE/dtype/fingerprint mismatch")
    return fixture, reference, dict(identity, reference_audit_blake3=blake3.blake3(reference_audit.read_bytes()).hexdigest())


def precheck_passes(rows):
    return bool(rows) and all(row["candidate_required_nodes"]["status"] == "pass" and row["native_mlp_bits_match_trace"] and row["native_block_bits_match_trace"] for row in rows)


@torch.inference_mode()
def same_input_precheck(baseline, actual, q, fixture, reference):
    require(linear_profile_summary(actual)["execution_profile"] == NATIVE_LINEAR_PROFILE, "precheck actual model is not native-v2")
    require(PROPOSED_PROFILE not in SUPPORTED_LINEAR_PROFILES, "precheck must precede production-profile support")
    before_a, before_model = normal_snapshot(baseline), normal_snapshot(actual)
    candidate = copy.deepcopy(actual.model.layers[0].mlp)
    original_bits = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    for member in FFN_MEMBERS:
        setattr(candidate, member, F64AccumLinear.from_linear(getattr(candidate, member)))
    candidate.eval()
    require(set(candidate.state_dict()) == set(original_bits) and all(value.dtype == torch.float32 and torch.equal(value, original_bits[name]) for name, value in candidate.state_dict().items()), "candidate clone changed actual artifact parameter bits/state")
    report = {"diagnostic_only": True, "scope": "layer.0 same-native-input FFN precheck", "construction_profile": BALANCED_ALGORITHM,
              "proposed_profile": PROPOSED_PROFILE, "production_profile_added": False, "default_loader_unchanged": True,
              "candidate_weight_source": "independent clone of actual artifact MLP; no SOURCE/theoretical weights or baseline hidden-state input",
              "attention_and_head_strategy": "native, unchanged; head/later layers not executed", "prompts": [],
              "formal_G2": "not_run", "G3": "not_run", "G4": "not_run", "ac10_satisfied": False}
    for index, prompt in enumerate(fixture["prompts"]):
        ids = prompt + fixture["continuation_ids"]
        source_nodes, _ = trace_first_block(baseline, ids)
        native_nodes, _ = trace_first_block(actual, ids)
        native_comparison = compare_nodes(source_nodes, native_nodes, q)
        assert_replay(native_comparison, reference["native_control_replay"][index])
        z1, z_norm = native_nodes["layer.0.post_attention_residual"], native_nodes["layer.0.post_attention_norm"]
        require(z1.dtype == z_norm.dtype == torch.float32 and z1.shape[-1] == z_norm.shape[-1] == q.shape[0], "same-input precheck changed F32/D inputs")
        native_ffn = actual.model.layers[0].mlp(z_norm)
        native_block = z1 + native_ffn
        native_ffn_matches = torch.equal(native_ffn, native_nodes["layer.0.ffn_output"])
        native_block_matches = torch.equal(native_block, native_nodes["layer.0.block_output"])
        require(native_ffn_matches and native_block_matches, "native same-input MLP/residual does not bitmatch trace")
        candidate_ffn = candidate(z_norm)
        candidate_block = z1 + candidate_ffn
        require(candidate_ffn.dtype == candidate_block.dtype == torch.float32 and candidate_ffn.shape == candidate_block.shape == native_ffn.shape and torch.isfinite(candidate_ffn).all().item() and torch.isfinite(candidate_block).all().item(), "candidate changed F32/D output or became non-finite")
        proposed_nodes = dict(native_nodes)
        proposed_nodes["layer.0.ffn_output"] = candidate_ffn
        proposed_nodes["layer.0.block_output"] = candidate_block
        compared = compare_nodes(source_nodes, proposed_nodes, q)
        report["prompts"].append({"prompt_index": index, "sequence_length": len(ids), "native_mlp_bits_match_trace": native_ffn_matches,
            "native_block_bits_match_trace": native_block_matches, "same_input_z_norm_blake3": blake3.blake3(z_norm.numpy().tobytes()).hexdigest(),
            "same_residual_z1_blake3": blake3.blake3(z1.numpy().tobytes()).hexdigest(), "native_required_nodes": native_comparison,
            "candidate_required_nodes": compared, "first_six_nodes_native_bits_unchanged": all(torch.equal(proposed_nodes[name], native_nodes[name]) for name in list(native_nodes)[:6]),
            "candidate_ffn": pair_summary(source_nodes["layer.0.ffn_output"], candidate_ffn.double() @ q),
            "candidate_block": pair_summary(source_nodes["layer.0.block_output"], candidate_block.double() @ q)})
    require(normal_snapshot(baseline) == before_a and normal_snapshot(actual) == before_model, "same-input candidate polluted normal models/parameters/state/buffers/metadata")
    require(all(value.dtype == torch.float32 and torch.equal(value, original_bits[name]) for name, value in candidate.state_dict().items()), "candidate changed cloned weight bits")
    passed = precheck_passes(report["prompts"])
    report["precheck_passed"] = passed
    report["decision"] = "eligible_for_conditional_per_role_profile_implementation" if passed else "stop_same_input_precheck_failed"
    report["normal_A_native_v2_unchanged"] = True
    report["candidate_clone_state_f32_bits_preserved"] = True
    report["first_failure"] = next(({"prompt_index": row["prompt_index"], "node": row["candidate_required_nodes"]["first_failing_node"]} for row in report["prompts"] if row["candidate_required_nodes"]["status"] != "pass"), None)
    costs = {member: {"calls": getattr(candidate, member).calls, "peak_f64_temporary_bytes": getattr(candidate, member).peak_f64_temporary_bytes,
                     "peak_working_bytes": getattr(candidate, member).peak_working_bytes, "elapsed_seconds": getattr(candidate, member).elapsed_seconds,
                     "fixed_output_row_chunk": F64_LINEAR_ROW_CHUNK} for member in FFN_MEMBERS}
    report["candidate_clone_costs"] = costs
    report["resource_note"] = "one actual-artifact MLP clone and temporary per-Linear F64, no full model double shadow; cumulative RSS is not a Compiler cap"
    return report


def main():
    parser = argparse.ArgumentParser(description="Fixed same-native-input artifact FFN precheck; failed gate prevents production profile")
    for name in ["source", "artifact", "secret", "fixture", "reference-audit", "report"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require(not args.report.exists(), "precheck report already exists")
    started = time.perf_counter()
    fixture, reference, identity = check_precheck_identity(args.source, args.artifact, args.secret, args.fixture, args.reference_audit)
    baseline, _ = load_local_llama_model(args.source, LlamaForCausalLM, torch)
    actual, q, keys = load_runtime(args.artifact, args.secret)
    report = same_input_precheck(baseline, actual, q, fixture, reference)
    report.update(identity=identity, material_G1=keys, provenance=provenance(), precheck_source_blake3=blake3.blake3(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter() - started, process_highwater_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: report[name] for name in ["diagnostic_only", "precheck_passed", "production_profile_added", "decision", "formal_G2", "G3", "G4"]}))
    raise SystemExit(0 if report["precheck_passed"] else 1)


if __name__ == "__main__":
    main()
