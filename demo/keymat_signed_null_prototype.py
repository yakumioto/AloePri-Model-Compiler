import argparse
import json
import resource
import time
from pathlib import Path

import blake3
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM

from aloepri_runtime import ExactCovariantNorm, require
from hidden_expansion_demo import compare_nodes
from keymat_float_semantics_audit import AuditFirstBlockReference, clone_first_block, normal_snapshot
from keymat_numerical_diagnostics import pair_summary, provenance, trace_first_block
from token_permutation_demo import artifact_fingerprint, load_json, load_local_llama_model, read_artifact

ALGORITHM = "algorithm1-signed-null-v3"


def validate_support(p, q):
    require(p.ndim == q.ndim == 2 and p.dtype == q.dtype == torch.float64 and p.shape[0] == q.shape[1] and p.shape[1] == q.shape[0], "prototype P/Q geometry/dtype mismatch")
    d, big_d = p.shape
    require(d > 0 and big_d > d and (big_d - d) % 4 == 0 and torch.isfinite(p).all().item() and torch.isfinite(q).all().item(), "prototype dimensions/finite invalid")
    diagonal = torch.diag(p[:, :d])
    require(torch.all((diagonal == 1) | (diagonal == -1)).item() and torch.any(diagonal == -1).item(), "prototype core is not nontrivial signed diagonal")
    require(torch.equal(p[:, :d], torch.diag(diagonal)) and torch.equal(q[:d], torch.diag(diagonal)), "prototype core has mixing or inconsistent signs")
    active_p = torch.any(p[:, d:] != 0, dim=0)
    active_q = torch.any(q[d:] != 0, dim=1)
    require(torch.all(active_p ^ active_q).item() and active_p.sum().item() == active_q.sum().item() == (big_d - d) // 2, "prototype auxiliary support is not complementary/nonzero")
    error = (p @ q - torch.eye(d, dtype=torch.float64)).abs()
    require(torch.isfinite(error).all().item() and error.max().item() <= 1e-5, "prototype G1 failed")
    ps, qs = torch.linalg.svdvals(p), torch.linalg.svdvals(q)
    report = {"d": d, "D": big_d, "h": (big_d-d)//2, "finite_p": True, "finite_q": True,
        "max_abs_pq_error": error.max().item(), "mean_abs_pq_error": error.mean().item(),
        "p_spectral_norm": ps.max().item(), "q_spectral_norm": qs.max().item(), "p_condition_estimate": (ps.max()/ps.min()).item(),
        "p_frobenius_norm": p.norm().item(), "q_frobenius_norm": q.norm().item(),
        "nonzero_P_auxiliary": True, "complementary_support": True, "core_signed_not_decoded": True,
        "negative_core_count": torch.count_nonzero(diagonal < 0).item(), "private_support_not_published": True}
    return active_p, active_q, report


class SignedColumnBlockedLinear(torch.nn.Linear):
    def __init__(self, weight, d, active_p):
        super().__init__(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=torch.float32)
        require(weight.dtype == torch.float32 and weight.device.type == "cpu" and weight.shape[1] > d, "prototype input weight contract invalid")
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.d = d
        self.register_buffer("_active_p", active_p, persistent=False)
        self.core_calls = self.aux_calls = 0
        self.peak_f32_temporary_weight_bytes = 0
        require(torch.count_nonzero(weight[:, d:][:, active_p]).item() == 0, "prototype input active-aux weight is nonzero")

    @torch.inference_mode()
    def forward(self, encoded):
        require(encoded.dtype == torch.float32 and encoded.shape[-1] == self.in_features and torch.isfinite(encoded).all().item(), "prototype input is not full-D F32/finite")
        core_weight = self.weight[:, :self.d].contiguous()
        core = F.linear(encoded[..., :self.d].contiguous(), core_weight, None)
        aux = F.linear(encoded[..., self.d:], self.weight[:, self.d:], None)
        self.core_calls += 1
        self.aux_calls += 1
        require(torch.count_nonzero(encoded[..., self.d:][..., ~self._active_p]).item() == 0, "prototype encoded inactive auxiliary state is nonzero")
        require(torch.isfinite(core).all().item() and torch.isfinite(aux).all().item() and torch.count_nonzero(aux).item() == 0, "prototype full auxiliary GEMM product is not zero")
        self.peak_f32_temporary_weight_bytes = max(self.peak_f32_temporary_weight_bytes, core_weight.numel()*4)
        result = core + aux
        require(result.dtype == torch.float32 and torch.isfinite(result).all().item(), "prototype input Linear output invalid")
        return result


class SignedRowBlockedLinear(torch.nn.Linear):
    def __init__(self, weight, d, active_p):
        super().__init__(weight.shape[1], weight.shape[0], bias=False, device="meta", dtype=torch.float32)
        require(weight.dtype == torch.float32 and weight.device.type == "cpu" and weight.shape[0] > d, "prototype output weight contract invalid")
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.d = d
        self.register_buffer("_active_p", active_p, persistent=False)
        self.core_calls = self.aux_calls = 0
        self.peak_f32_temporary_weight_bytes = 0
        require(torch.count_nonzero(weight[d:][~active_p]).item() == 0, "prototype inactive output auxiliary weight is nonzero")

    @torch.inference_mode()
    def forward(self, projected):
        require(projected.dtype == torch.float32 and projected.shape[-1] == self.in_features and torch.isfinite(projected).all().item(), "prototype output Linear input invalid")
        core = F.linear(projected, self.weight[:self.d], None)
        aux = F.linear(projected, self.weight[self.d:], None)
        self.core_calls += 1
        self.aux_calls += 1
        require(torch.isfinite(core).all().item() and torch.isfinite(aux).all().item() and torch.count_nonzero(aux[..., ~self._active_p]).item() == 0, "prototype output auxiliary support product invalid")
        result = torch.cat([core, aux], dim=-1)
        require(result.dtype == torch.float32 and result.shape[-1] == self.out_features, "prototype output is not full-D F32")
        return result


@torch.inference_mode()
def transform_weight(weight, p, q, role):
    require(weight.dtype == torch.float32, "prototype SOURCE weight is not F32")
    if role == "embedding": return (weight.double() @ p).float()
    if role == "output": return (p.T @ weight.double()).float()
    require(role == "input", "invalid prototype weight role")
    return (weight.double() @ q.T).float()


@torch.inference_mode()
def build_prototype(source, p, q):
    active_p, _, math = validate_support(p, q)
    d, big_d = p.shape
    require(source.config.hidden_size == d, "prototype source hidden dimension changed")
    block = clone_first_block(source)
    data_checks = []
    for section, members in [("self_attn", ["q_proj", "k_proj", "v_proj", "o_proj"]), ("mlp", ["gate_proj", "up_proj", "down_proj"])]:
        for member in members:
            original = getattr(getattr(source.model.layers[0], section), member)
            role = "output" if member in ["o_proj", "down_proj"] else "input"
            transformed = transform_weight(original.weight, p, q, role)
            data_checks.append({"name": f"model.layers.0.{section}.{member}", "role": role, "data_changed": not torch.equal(original.weight, transformed[:d] if role == "output" else transformed[:, :d]), "shape": list(transformed.shape)})
            replacement = SignedRowBlockedLinear(transformed, d, active_p) if role == "output" else SignedColumnBlockedLinear(transformed, d, active_p)
            setattr(getattr(block, section), member, replacement)
    for name in ["input_layernorm", "post_attention_layernorm"]:
        norm = ExactCovariantNorm(d, source.config.rms_norm_eps, p, q)
        norm.weight = torch.nn.Parameter(getattr(block, name).weight.detach().clone(), requires_grad=False)
        setattr(block, name, norm)
    model = AuditFirstBlockReference(source, block, "SIGNED_NULL_BLOCKED_PROTOTYPE_NOT_PRODUCTION")
    embedding_weight = transform_weight(source.model.embed_tokens.weight, p, q, "embedding")
    with torch.device("meta"):
        embedding = torch.nn.Embedding(source.config.vocab_size, big_d, source.config.pad_token_id)
    embedding.weight = torch.nn.Parameter(embedding_weight, requires_grad=False)
    model.model.embed_tokens = embedding
    model.eval()
    require(all(row["data_changed"] for row in data_checks) and torch.count_nonzero(embedding_weight[:, d:]).item() > 0 and not torch.equal(source.model.embed_tokens.weight, embedding_weight[:, :d]), "prototype became padding or an unchanged target")
    return model, math, data_checks


@torch.inference_mode()
def head_fidelity_check(source, p, q, active_p, ids):
    transformed = transform_weight(source.lm_head.weight, p, q, "input")
    module = SignedColumnBlockedLinear(transformed, p.shape[0], active_p)
    logical = source.model.embed_tokens(torch.tensor([ids]))
    encoded = (logical.double() @ p).float()
    expected = F.linear(logical, source.lm_head.weight, None)
    actual = module(encoded)
    report = pair_summary(expected, actual)
    report.update(core_and_auxiliary_both_computed=module.core_calls == module.aux_calls == 1,
                  signed_zero_bits_may_differ=True, weight_data_changed=not torch.equal(source.lm_head.weight, transformed[:, :p.shape[0]]),
                  peak_f32_core_weight_copy_bytes=module.peak_f32_temporary_weight_bytes)
    require(report["pass"] and report["weight_data_changed"], "prototype head arithmetic/nontriviality failed")
    return report


def main():
    parser = argparse.ArgumentParser(description="Signed-null mathematics/native-F32 prototype; not a production algorithm/artifact")
    for name in ["source", "control-secret", "material", "material-diagnostics", "fixture", "report"]:
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require(not args.report.exists(), "prototype report exists")
    started = time.perf_counter()
    metadata = load_json(args.material_diagnostics)
    control = load_json(args.control_secret)
    fixture = load_json(args.fixture)
    require(metadata["algorithm"] == ALGORITHM and metadata["diagnostic_only"] and metadata["production_algorithm_added"] is False, "prototype construction identity invalid")
    require(metadata["source_fingerprint"] == control["source_fingerprint"] and metadata["control_secret_id"] == control["secret_id"] and metadata["lambda_bits"] == control["lambda_bits"] and metadata["h"] == fixture["h"], "prototype changed fixed generation inputs")
    require(control["master_seed"] == blake3.blake3(int(fixture["fixture_seed"]).to_bytes(8,"little")).hexdigest(), "prototype changed fixed seed")
    config, tensors = read_artifact(args.source)
    require(all(value[0] == "F32" for value in tensors.values()) and artifact_fingerprint(args.source, config, tensors) == metadata["source_fingerprint"], "prototype source/dtype/fingerprint changed")
    del tensors
    data = args.material.read_bytes();d,big_d = metadata["d"],metadata["D"];split=8*d*big_d
    require(len(data)==2*split and blake3.blake3(data[:split]).hexdigest()==metadata["p_digest"] and blake3.blake3(data[split:]).hexdigest()==metadata["q_digest"], "prototype private P/Q bytes mismatch")
    values=torch.frombuffer(bytearray(data),dtype=torch.float64);p=values[:d*big_d].view(d,big_d);q=values[d*big_d:].view(big_d,d)
    source,_=load_local_llama_model(args.source,LlamaForCausalLM,torch)
    before=normal_snapshot(source)
    model,math,data_checks=build_prototype(source,p,q)
    active_p,_,_=validate_support(p,q)
    head=head_fidelity_check(source,p,q,active_p,fixture["prompts"][0]+fixture["continuation_ids"])
    report={"diagnostic_only":True,"algorithm":ALGORITHM,"production_algorithm_added":False,"production_profile_added":False,
            "B_perturbation_V_zero":True,"not_dense_gaussian_or_privacy_reproduction":True,"math":math,"first_layer_target_data_checks":data_checks,
            "head_linear_fidelity":head,"embedding_auxiliary_nonzero":True,"private_support_not_published":True,"prompts":[],"formal_G2":"not_run","G3":"not_run","G4":"not_run"}
    for index,prompt in enumerate(fixture["prompts"]):
        ids=prompt+fixture["continuation_ids"]
        baseline,_=trace_first_block(source,ids);actual,_=trace_first_block(model,ids)
        require(all(value.dtype==torch.float32 and value.shape[-1]==big_d for value in actual.values()),"prototype complete node boundary is not D/F32")
        compared=compare_nodes(baseline,actual,q)
        compared["prompt_index"]=index
        report["prompts"].append(compared)
    require(normal_snapshot(source)==before,"prototype polluted native SOURCE")
    report["core_and_auxiliary_gemm_calls"]={f"{section}.{member}":{"core":getattr(getattr(model.model.layers[0],section),member).core_calls,"aux":getattr(getattr(model.model.layers[0],section),member).aux_calls} for section,members in [("self_attn",["q_proj","k_proj","v_proj","o_proj"]),("mlp",["gate_proj","up_proj","down_proj"])] for member in members}
    require(all(row["core"]==row["aux"]==len(fixture["prompts"]) for row in report["core_and_auxiliary_gemm_calls"].values()),"prototype skipped an auxiliary block")
    passed=all(row["status"]=="pass" for row in report["prompts"])
    report.update(necessary_gate_passed=passed,decision="eligible_for_conditional_production_implementation" if passed else "stop_signed_null_prototype_gate_failed",
                  provenance=provenance(),prototype_source_blake3=blake3.blake3(Path(__file__).read_bytes()).hexdigest(),fixture_blake3=blake3.blake3(args.fixture.read_bytes()).hexdigest(),
                  prototype_material_identity=metadata,normal_source_unchanged=True,elapsed_seconds=time.perf_counter()-started,process_highwater_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({name:report[name] for name in ["diagnostic_only","necessary_gate_passed","production_algorithm_added","decision","formal_G2","G3","G4"]}))
    raise SystemExit(0 if passed else 1)


if __name__=="__main__":main()
