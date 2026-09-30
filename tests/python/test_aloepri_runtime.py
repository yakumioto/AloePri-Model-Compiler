import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
import aloepri_runtime as runtime
import hidden_expansion_demo as demo
import keymat_numerical_diagnostics as diagnostic


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source"
        cls.artifact = cls.root / "artifact"
        cls.secret = cls.root / "secret.json"
        torch.manual_seed(42)
        config = LlamaConfig(hidden_size=8, intermediate_size=13, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, vocab_size=19, tie_word_embeddings=True, eos_token_id=0)
        cls.baseline = LlamaForCausalLM(config).float().eval()
        with torch.no_grad():
            for parameter in cls.baseline.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(std=0.02)
        cls.baseline.save_pretrained(cls.source, safe_serialization=True)
        binary = ROOT / "target" / "debug" / "aloepri"
        result = subprocess.run([str(binary), "transform", str(cls.source), "--output", str(cls.artifact), "--method", "aloepri-keymat", "--expansion-size", "2", "--keymat-lambda", "0.3", "--keymat-fixture-seed", "42", "--secret-output", str(cls.secret), "--max-shard-size", "1KiB"], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.model, cls.q, cls.diagnostics = runtime.load_runtime(cls.artifact, cls.secret)
        cls.fixture = {"prompts": [[1, 4], [3, 7, 9]], "continuation_ids": [5, 8], "max_new_tokens": 3}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_tiny_gqa_layerwise_cache_logits_and_generation(self):
        report = demo.run_gates(self.baseline, self.model, self.q, self.diagnostics, self.fixture)
        self.assertEqual([report[g]["status"] for g in ["G1", "G2", "G3", "G4"]], ["pass"] * 4, report)
        self.assertEqual(len(report["G2"]["prompts"][0]["nodes"]), 16)
        self.assertTrue(all(item["expanded_shape"][-1] == 12 for item in report["G2"]["prompts"][0]["nodes"]))
        self.assertEqual(self.model.config.hidden_size, 8)
        self.assertEqual(self.model.model.layers[0].self_attn.head_dim, 4)
        self.assertEqual(self.model.model.layers[0].mlp.intermediate_size, 13)
        self.assertNotEqual(self.model.lm_head.weight.data_ptr(), self.model.model.embed_tokens.weight.data_ptr())
        self.assertFalse(any(key.endswith("._p") or key.endswith("._q") for key in self.model.state_dict()))

    def test_each_physical_weight_matches_independent_formula(self):
        p, q = self.model.model.norm._p, self.q
        original = self.baseline.state_dict()
        transformed = self.model.state_dict()
        for name, weight in transformed.items():
            source = original[name].double()
            if "norm" in name:
                expected = source
            elif name == "model.embed_tokens.weight":
                expected = source @ p
            elif name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                expected = p.T @ source
            else:
                expected = source @ q.T
            self.assertTrue(torch.allclose(weight.double(), expected, atol=1e-6, rtol=0), name)
        x = torch.randn(2, 8, dtype=torch.float64)
        for name in ["lm_head.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"]:
            expected = x @ original[name].double().T
            actual = x @ p @ transformed[name].double().T
            self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=0), name)
        for name in ["model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.down_proj.weight"]:
            u = torch.randn(2, original[name].shape[1], dtype=torch.float64)
            self.assertTrue(torch.allclose(u @ transformed[name].double().T, u @ original[name].double().T @ p, atol=1e-5, rtol=0), name)

    def test_exact_norm_matches_independent_formula(self):
        norm = self.model.model.norm
        z = torch.randn(2, 3, 12)
        x = (z.double() @ norm._q).float()
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight
        expected = (y.double() @ norm._p).float()
        self.assertTrue(torch.equal(norm(z), expected))

    def test_source_independent_loading_in_separate_process(self):
        hidden = self.root / "unavailable-source"
        self.source.rename(hidden)
        try:
            with patch.object(LlamaForCausalLM, "from_pretrained", side_effect=AssertionError("source loading forbidden")):
                loaded, _, _ = runtime.load_runtime(self.artifact, self.secret)
                self.assertEqual(tuple(loaded(input_ids=torch.tensor([[1, 4]]), use_cache=False).logits.shape), (1, 2, 19))
                for member in ["gate_proj", "up_proj", "down_proj"]:
                    setattr(loaded.model.layers[0].mlp, member, runtime.F64AccumLinear.from_linear(getattr(loaded.model.layers[0].mlp, member)))
                nodes, _ = diagnostic.trace_first_block(loaded, [1, 4])
                self.assertTrue(all(value.dtype == torch.float32 and value.shape[-1] == 12 for value in nodes.values()))
            result = subprocess.run([sys.executable, str(ROOT / "demo" / "aloepri_runtime.py"), "--artifact", str(self.artifact), "--secret", str(self.secret), "--token-ids", "1", "4"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["finite_logits"])
        finally:
            hidden.rename(self.source)

    def test_manifest_json_order_is_not_an_identity(self):
        path = self.artifact / "aloepri.json"
        original = path.read_bytes()
        try:
            path.write_text(json.dumps(json.loads(original), sort_keys=True))
            model, _, _ = runtime.load_runtime(self.artifact, self.secret)
            self.assertEqual(model.model.embed_tokens.weight.shape[1], 12)
        finally:
            path.write_bytes(original)

    def test_wrong_contracts_are_rejected(self):
        path = self.artifact / "aloepri.json"
        original = path.read_bytes()
        parsed = json.loads(original)
        try:
            for key, value in [("standard_hf_checkpoint", True), ("artifact_version", 3), ("secret_id", "00" * 32)]:
                changed = copy.deepcopy(parsed)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runtime.load_runtime(self.artifact, self.secret)
            for key, value in [("norm_mode", "paper"), ("version", "2"), ("kv_cache_format", "aloepri")]:
                changed = copy.deepcopy(parsed)
                changed["runtime_contract"][key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    runtime.load_runtime(self.artifact, self.secret)
        finally:
            path.write_bytes(original)

    def test_digest_and_duplicate_json_fail_closed(self):
        path = self.secret
        original = path.read_bytes()
        try:
            parsed = json.loads(original)
            parsed["p_digest"] = "00" * 32
            path.write_text(json.dumps(parsed))
            with self.assertRaises(ValueError):
                runtime.load_keymat(path)
            path.write_text('{"version":1,"version":1}')
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.load_keymat(path)
        finally:
            path.write_bytes(original)
        with self.assertRaises(ValueError):
            runtime.child_file(self.artifact, "../source/model.safetensors")

    def test_numerical_failure_is_fail_closed_and_reports_first_node(self):
        original = demo.trace
        def failing_trace(model, ids):
            nodes, logits = original(model, ids)
            if model is self.model:
                nodes["layer.0.input_norm"] = torch.full_like(nodes["layer.0.input_norm"], float("nan"))
            return nodes, logits
        with patch.object(demo, "trace", side_effect=failing_trace), patch.object(demo, "greedy", side_effect=AssertionError("generation must not run")):
            report = demo.run_gates(self.baseline, self.model, self.q, self.diagnostics, self.fixture)
        self.assertEqual(report["G2"]["first_failure"]["node"], "layer.0.input_norm")
        self.assertEqual(report["G3"]["status"], "not_run")
        self.assertEqual(report["G4"]["status"], "not_run")

    def test_eight_e_minus_five_logits_cannot_pass(self):
        original = demo.trace
        def failing_trace(model, ids):
            nodes, logits = original(model, ids)
            if model is self.model:
                logits = logits + 8e-5
            return nodes, logits
        with patch.object(demo, "trace", side_effect=failing_trace), patch.object(demo, "greedy", side_effect=AssertionError("generation must not run")):
            report = demo.run_gates(self.baseline, self.model, self.q, self.diagnostics, self.fixture)
        self.assertEqual(report["G2"]["status"], "pass")
        self.assertEqual(report["G3"]["status"], "fail")
        self.assertEqual(report["G4"]["status"], "not_run")

    def test_f64_candidate_preserves_f32_state_and_chunked_boundaries(self):
        original = torch.nn.Linear(5, runtime.F64_LINEAR_ROW_CHUNK + 7, bias=False).eval()
        before = original.weight.detach().clone()
        candidate = runtime.F64AccumLinear.from_linear(original)
        self.assertIs(candidate.weight, original.weight)
        candidate.load_state_dict(original.state_dict(), strict=True, assign=True)
        self.assertEqual(set(candidate.state_dict()), {"weight"})
        self.assertTrue(torch.equal(before, candidate.weight))
        for shape in [(5,), (2, 5), (2, 3, 5)]:
            inputs = torch.randn(shape)
            expected = torch.nn.functional.linear(inputs.double(), before.double()).float()
            actual = candidate(inputs)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.shape[-1], original.out_features)
        self.assertEqual(candidate.weight.dtype, torch.float32)
        self.assertFalse(any(value.dtype == torch.float64 for value in candidate.state_dict().values()))
        self.assertGreater(candidate.peak_f64_temporary_bytes, 0)
        self.assertFalse(list(candidate.buffers()))
        with self.assertRaises(ValueError):
            candidate(torch.randn(2, 5, dtype=torch.float64))
        with self.assertRaises(ValueError):
            candidate(torch.full((2, 5), float("nan")))
        with self.assertRaises(ValueError):
            runtime.F64AccumLinear(5, 7, bias=True)
        with self.assertRaises(ValueError):
            runtime.F64AccumLinear.from_linear(torch.nn.Linear(5, 7, dtype=torch.float64, bias=False))

    def test_candidate_requires_every_local_sequence_and_f32_constraint(self):
        profile = {"state": {"parameters_f32": True, "state_dict_f32": True, "physical_head_independent": True},
                   "weight_bits_unchanged": True, "prompts": [{"status": "pass"}] * 3}
        self.assertTrue(diagnostic.local_candidate_passes(profile))
        failed = copy.deepcopy(profile)
        failed["prompts"][2]["status"] = "fail"
        self.assertFalse(diagnostic.local_candidate_passes(failed))
        failed = copy.deepcopy(profile)
        failed["state"]["state_dict_f32"] = False
        self.assertFalse(diagnostic.local_candidate_passes(failed))
        original = demo.trace
        def failing_later_trace(model, ids):
            nodes, logits = original(model, ids)
            if model is self.model:
                nodes["layer.1.input_norm"] = torch.full_like(nodes["layer.1.input_norm"], float("nan"))
            return nodes, logits
        with patch.object(demo, "trace", side_effect=failing_later_trace), patch.object(demo, "cache_logits", side_effect=AssertionError("G3 must not run")), patch.object(demo, "greedy", side_effect=AssertionError("G4 must not run")):
            report = demo.run_gates(self.baseline, self.model, self.q, self.diagnostics, self.fixture)
        self.assertEqual(report["G2"]["first_failure"]["node"], "layer.1.input_norm")
        self.assertEqual(report["G3"]["status"], "not_run")
        self.assertEqual(report["G4"]["status"], "not_run")

    def test_error_vectors_reconstruct_without_adding_scalar_maxima(self):
        a = torch.tensor([1.0, -2.0], dtype=torch.float64)
        b = a + torch.tensor([3.0, -4.0], dtype=torch.float64)
        c = a + torch.tensor([0.01, -0.02], dtype=torch.float64)
        report, vectors = diagnostic.difference_chain({"origin": a, "first": b, "last": c})
        self.assertTrue(torch.allclose(sum(vectors.values()), c - a, atol=1e-15, rtol=0))
        self.assertLess(report["reconstruction_max_abs"], 1e-15)
        scalar_sum = sum(item["max_abs"] for item in report["components"].values())
        self.assertGreater(scalar_sum, report["total"]["max_abs"])
        zero = torch.tensor([0.0, -0.0])
        self.assertEqual(diagnostic.ulp_distance(zero, -zero)["max_ulp"], 0)
        value = torch.tensor([-1.0, 1.0])
        adjacent = torch.nextafter(value, torch.full_like(value, float("inf")))
        self.assertEqual(diagnostic.ulp_distance(value, adjacent)["max_ulp"], 1)

    def test_first_block_oracle_and_same_input_counterfactual_chains(self):
        ids = self.fixture["prompts"][0] + self.fixture["continuation_ids"]
        baseline, base_internal = diagnostic.trace_first_block(self.baseline, ids)
        expanded, internal = diagnostic.trace_first_block(self.model, ids)
        self.assertEqual(list(baseline)[-1], "layer.0.block_output")
        self.assertEqual(len(baseline), 8)
        self.assertTrue(torch.equal(internal["product"], internal["down_input"]))
        p = self.model.model.norm._p
        weights, theory = diagnostic.weight_oracle(self.baseline, self.model, p, self.q)
        self.assertTrue(all(item["f32_encoding"]["max_abs"] < 1e-6 for item in weights.values()))
        for name in ["gate_proj", "up_proj"]:
            chain, vectors = diagnostic.input_linear_chain(baseline["layer.0.post_attention_norm"], expanded["layer.0.post_attention_norm"],
                getattr(self.baseline.model.layers[0].mlp, name).weight, getattr(self.model.model.layers[0].mlp, name).weight, theory[name], p)
            self.assertLess(chain["reconstruction_max_abs"], 1e-12)
            self.assertLess(chain["components"]["covariance64"]["max_abs"], 1e-12)
        chain, vectors = diagnostic.down_linear_chain(base_internal["down_input"], internal["down_input"],
            self.baseline.model.layers[0].mlp.down_proj.weight, self.model.model.layers[0].mlp.down_proj.weight, theory["down_proj"], self.q)
        self.assertLess(chain["reconstruction_max_abs"], 1e-12)
        recovered_error = expanded["layer.0.ffn_output"].double() @ self.q - baseline["layer.0.ffn_output"].double()
        self.assertTrue(torch.allclose(sum(vectors.values()), recovered_error, atol=1e-12, rtol=0))
        self.assertTrue(diagnostic.state_summary(self.model, 12)["state_dict_f32"])

    def test_generation_edges_and_strict_metrics(self):
        class Fixed:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.tensor([[[1.0, 0.0]]]), past_key_values=object())
        self.assertEqual(runtime.greedy(Fixed(), [1], 0, 0), [1])
        self.assertEqual(runtime.greedy(Fixed(), [1], 4, [0]), [1, 0])
        self.assertEqual(runtime.greedy(Fixed(), [1], 4, None), [1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            runtime.greedy(Fixed(), [], 4, 0)
        self.assertFalse(demo.metrics(torch.tensor([1.0]), torch.tensor([1.0 + 8e-5]))["pass"])
        self.assertFalse(demo.metrics(torch.tensor([0.0]), torch.tensor([float("nan")]))["pass"])


if __name__ == "__main__":
    unittest.main()
