import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from transformers import LlamaForCausalLM
from aloepri_runtime import (TOLERANCE, NATIVE_LINEAR_PROFILE, SUPPORTED_LINEAR_PROFILES,
    load_runtime, greedy, require, linear_profile_summary, linear_profile_costs)
from token_permutation_demo import load_json, load_local_llama_model, read_artifact, artifact_fingerprint


def metrics(expected, actual):
    if expected.shape != actual.shape:
        return {"pass": False, "finite": False, "reason": "shape mismatch", "expected_shape": list(expected.shape), "actual_shape": list(actual.shape), "max_abs": None, "mean_abs": None}
    finite = torch.isfinite(expected).all().item() and torch.isfinite(actual).all().item()
    if not finite:
        return {"pass": False, "finite": False, "reason": "non-finite", "max_abs": None, "mean_abs": None}
    delta = (expected.double() - actual.double()).abs()
    max_abs = delta.max().item() if delta.numel() else 0.0
    return {"pass": max_abs <= TOLERANCE, "finite": True, "max_abs": max_abs, "mean_abs": delta.mean().item() if delta.numel() else 0.0}


@torch.inference_mode()
def trace(model, ids):
    nodes = OrderedDict()
    handles = []

    def post(module, name, tuple_output=False):
        def hook(_module, _inputs, output):
            value = output[0] if tuple_output else output
            nodes[name] = value.detach().clone()
        handles.append(module.register_forward_hook(hook))

    def pre(module, name):
        def hook(_module, inputs):
            nodes[name] = inputs[0].detach().clone()
        handles.append(module.register_forward_pre_hook(hook))

    post(model.model.embed_tokens, "embedding")
    for index, layer in enumerate(model.model.layers):
        prefix = f"layer.{index}."
        pre(layer, prefix + "input")
        post(layer.input_layernorm, prefix + "input_norm")
        post(layer.self_attn, prefix + "attention_output", True)
        pre(layer.post_attention_layernorm, prefix + "post_attention_residual")
        post(layer.post_attention_layernorm, prefix + "post_attention_norm")
        post(layer.mlp, prefix + "ffn_output")
        post(layer, prefix + "block_output")
    post(model.model.norm, "final_norm")
    try:
        output = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long), use_cache=False)
        return nodes, output.logits.detach()
    finally:
        for handle in handles:
            handle.remove()


def compare_nodes(baseline, expanded, q):
    require(list(baseline) == list(expanded), "trace nodes differ")
    results = []
    for name, x in baseline.items():
        z = expanded[name]
        if z.shape[-1] != q.shape[0]:
            result = {"pass": False, "finite": False, "reason": "residual is not D-dimensional", "max_abs": None, "mean_abs": None}
        else:
            result = metrics(x, z.double() @ q)
        results.append(dict(node=name, expanded_shape=list(z.shape), **result))
    first = next((item["node"] for item in results if not item["pass"]), None)
    return {"status": "pass" if first is None else "fail", "first_failing_node": first, "max_abs": max((item["max_abs"] for item in results if item["max_abs"] is not None), default=None), "nodes": results}


@torch.inference_mode()
def cache_logits(model, prompt, continuation):
    output = model(input_ids=torch.tensor([prompt]), attention_mask=torch.ones((1, len(prompt)), dtype=torch.long), use_cache=True)
    prefill = output.logits.detach()
    cache = output.past_key_values
    decoded = []
    for index, token in enumerate(continuation):
        length = len(prompt) + index + 1
        output = model(input_ids=torch.tensor([[token]]), attention_mask=torch.ones((1, length), dtype=torch.long), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        decoded.append(output.logits.detach())
        for layer in cache.layers:
            require(list(layer.keys.shape) == [1, model.config.num_key_value_heads, length, model.config.head_dim], "KV cache is not standard projection-space")
            require(layer.values.shape == layer.keys.shape, "KV cache shape mismatch")
    return prefill, decoded


def generation_gate(baseline, runtime, prompts, max_new_tokens, previous_pass):
    if not previous_pass:
        return {"status": "not_run"}
    comparisons = []
    for prompt in prompts:
        expected = greedy(baseline, prompt, max_new_tokens, baseline.config.eos_token_id)
        actual = greedy(runtime, prompt, max_new_tokens, runtime.config.eos_token_id)
        comparisons.append({"pass": expected == actual, "baseline_ids": expected, "runtime_ids": actual})
    return {"status": "pass" if all(item["pass"] for item in comparisons) else "fail", "sequences": comparisons}


def run_gates(baseline, runtime, q, diagnostics, fixture):
    prompts = fixture["prompts"]
    continuation = fixture["continuation_ids"]
    require(prompts and all(prompt and all(type(i) is int and 0 <= i < baseline.config.vocab_size for i in prompt) for prompt in prompts), "invalid/empty fixture prompts")
    require(continuation and all(type(i) is int and 0 <= i < baseline.config.vocab_size for i in continuation), "invalid continuation")
    require(type(fixture["max_new_tokens"]) is int and fixture["max_new_tokens"] >= 0, "invalid generation length")
    report = {"predicate": "finite && max_abs <= 1e-5; rtol=0", "G1": dict(status="pass", **diagnostics), "G2": {"status": "not_run"}, "G3": {"status": "not_run"}, "G4": {"status": "not_run"}}
    layerwise = []
    full_logits = []
    for index, prompt in enumerate(prompts):
        ids = prompt + continuation
        original, expected = trace(baseline, ids)
        expanded, actual = trace(runtime, ids)
        comparison = compare_nodes(original, expanded, q)
        layerwise.append(dict(prompt_index=index, **comparison))
        full_logits.append((expected, actual))
    first = next(({"prompt_index": item["prompt_index"], "node": item["first_failing_node"]} for item in layerwise if item["status"] != "pass"), None)
    report["G2"] = {"status": "pass" if first is None else "fail", "first_failure": first, "max_abs": max((item["max_abs"] for item in layerwise if item["max_abs"] is not None), default=None), "prompts": layerwise}
    report["diagnostic_full_logits"] = [dict(prompt_index=i, **metrics(expected, actual)) for i, (expected, actual) in enumerate(full_logits)]
    if first is not None:
        return report
    comparisons = []
    for index, (prompt, (full_expected, full_actual)) in enumerate(zip(prompts, full_logits)):
        expected_prefill, expected_decode = cache_logits(baseline, prompt, continuation)
        actual_prefill, actual_decode = cache_logits(runtime, prompt, continuation)
        checks = {"full": metrics(full_expected, full_actual), "prefill": metrics(expected_prefill, actual_prefill), "baseline_prefill_uncached": metrics(full_expected[:, :len(prompt)], expected_prefill), "runtime_prefill_uncached": metrics(full_actual[:, :len(prompt)], actual_prefill)}
        for step, (expected, actual) in enumerate(zip(expected_decode, actual_decode)):
            position = len(prompt) + step
            checks[f"decode.{step}"] = metrics(expected, actual)
            checks[f"baseline_decode_uncached.{step}"] = metrics(full_expected[:, position:position + 1], expected)
            checks[f"runtime_decode_uncached.{step}"] = metrics(full_actual[:, position:position + 1], actual)
        comparisons.append({"prompt_index": index, "checks": checks})
    passed = all(check["pass"] for item in comparisons for check in item["checks"].values())
    report["G3"] = {"status": "pass" if passed else "fail", "prompts": comparisons}
    report["G4"] = generation_gate(baseline, runtime, prompts, fixture["max_new_tokens"], passed)
    return report


def main():
    parser = argparse.ArgumentParser(description="Fail-closed KeyMat layerwise correctness oracle")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--linear-profile", choices=sorted(SUPPORTED_LINEAR_PROFILES), default=NATIVE_LINEAR_PROFILE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture = load_json(args.fixture)
    manifest = load_json(args.artifact / "aloepri.json")
    source_config, source_tensors = read_artifact(args.source)
    require(all(tensor[0] == "F32" for tensor in source_tensors.values()), "baseline fixture must be F32")
    require(artifact_fingerprint(args.source, source_config, source_tensors) == manifest["source_fingerprint"], "baseline source fingerprint mismatch")
    del source_tensors
    binding = manifest["plan"]["keymat_binding"]
    if "h" in fixture:
        import struct
        require(fixture["h"] == binding["expansion_size"] and struct.unpack("<Q", struct.pack("<d", fixture["lambda"]))[0] == binding["lambda_bits"], "fixture KeyMat parameters mismatch")
    if "fixture_seed" in fixture:
        import blake3
        import struct
        seed = blake3.blake3(struct.pack("<Q", fixture["fixture_seed"])).hexdigest()
        require(load_json(args.secret)["master_seed"] == seed, "fixture seed mismatch")
    baseline, _ = load_local_llama_model(args.source, LlamaForCausalLM, torch)
    runtime, q, diagnostics = load_runtime(args.artifact, args.secret, linear_profile=args.linear_profile)
    actual_profile = linear_profile_summary(runtime)
    require(actual_profile["execution_profile"] == args.linear_profile, "formal harness profile installation mismatch")
    report = run_gates(baseline, runtime, q, diagnostics, fixture)
    report["fixture"] = fixture
    report["source_fingerprint"] = manifest["source_fingerprint"]
    report["secret_id"] = manifest["secret_id"]
    report["physical_dimensions"] = manifest["runtime_contract"]["physical_dimensions"]
    report["construction_profile"] = binding["algorithm"]
    report["execution_profile"] = actual_profile["execution_profile"]
    report["installed_linear_profile"] = linear_profile_summary(runtime)
    report["linear_costs"] = linear_profile_costs(runtime)
    report["baseline_profile"] = NATIVE_LINEAR_PROFILE
    if report["G3"]["status"] == "fail":
        report["G3"]["failed_checks"] = [{"prompt_index": row["prompt_index"], "check": name,
            "path": "baseline" if name.startswith("baseline_") else ("runtime_self_consistency" if name.startswith("runtime_") else "cross_model"), **check}
            for row in report["G3"]["prompts"] for name, check in row["checks"].items() if not check["pass"]]
    report["plan_hash"] = manifest["plan_hash"]
    report["layout_hash"] = manifest["layout_hash"]
    from keymat_numerical_diagnostics import provenance
    report["provenance"] = provenance()
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({gate: report[gate]["status"] for gate in ["G1", "G2", "G3", "G4"]}))
    raise SystemExit(0 if all(report[gate]["status"] == "pass" for gate in ["G1", "G2", "G3", "G4"]) else 1)


if __name__ == "__main__":
    main()
