import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import blake3
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
import aloepri_runtime as runtime
import hidden_expansion_demo as gates
import keymat_float_semantics_audit as audit
from keymat_numerical_diagnostics import trace_first_block


class FloatSemanticsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.source = cls.root / "source"
        cls.artifact = cls.root / "artifact"
        private = cls.root / "private"
        private.mkdir()
        cls.secret = private / "secret.json"
        torch.manual_seed(317)
        config = LlamaConfig(hidden_size=8, intermediate_size=13, num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=1, vocab_size=19, tie_word_embeddings=True)
        cls.baseline = LlamaForCausalLM(config).float().eval()
        with torch.no_grad():
            for parameter in cls.baseline.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(std=1.0)
        cls.baseline.save_pretrained(cls.source, safe_serialization=True)
        result = subprocess.run([str(ROOT / "target/debug/aloepri"), "transform", str(cls.source), "--output", str(cls.artifact),
            "--method", "aloepri-keymat", "--keymat-algorithm", runtime.BALANCED_ALGORITHM, "--expansion-size", "2",
            "--keymat-lambda", "0.3", "--keymat-fixture-seed", "42", "--secret-output", str(cls.secret)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.fixture = {"prompts": [[1, 4], [3, 7, 9], [5, 2, 6, 4]], "continuation_ids": [5, 8], "h": 2, "lambda": 0.3, "fixture_seed": 42, "max_new_tokens": 0}
        cls.fixture_path = cls.root / "fixture.json"
        cls.fixture_path.write_text(json.dumps(cls.fixture))
        cls.reference_path = cls.root / "reference.json"
        model, q, _ = runtime.load_runtime(cls.artifact, cls.secret)
        base = [trace_first_block(cls.baseline, prompt + cls.fixture["continuation_ids"])[0] for prompt in cls.fixture["prompts"]]
        profiles = {}
        for profile in [runtime.NATIVE_LINEAR_PROFILE, runtime.ALL_BOUNDARY_LINEAR_PROFILE]:
            runtime.install_linear_profile(model, runtime.BALANCED_ALGORITHM, 12, linear_profile=profile)
            rows = []
            for index, (prompt, original) in enumerate(zip(cls.fixture["prompts"], base)):
                actual = trace_first_block(model, prompt + cls.fixture["continuation_ids"])[0]
                rows.append(dict(gates.compare_nodes(original, actual, q), prompt_index=index))
            profiles[profile] = {"execution_profile": profile, "prompts": rows,
                "installed_linear_profile": runtime.linear_profile_summary(model), "installed_scope_count": 15,
                "local_gate_pass": all(row["status"] == "pass" for row in rows)}
        if profiles[runtime.ALL_BOUNDARY_LINEAR_PROFILE]["local_gate_pass"]:
            raise AssertionError("bounded numerical test fixture must have an actual first-block failure")
        manifest = json.loads((cls.artifact / "aloepri.json").read_text())
        cls.reference = {"diagnostic_only": True, "construction_profile": runtime.BALANCED_ALGORITHM,
            "execution_profile": runtime.ALL_BOUNDARY_LINEAR_PROFILE, "candidate_local_gate_pass": False,
            "full_model_G2": "not_run", "G3": "not_run", "G4": "not_run", "profiles": profiles,
            "identity": {field: manifest[field] for field in ["source_fingerprint", "secret_id", "plan_hash", "layout_hash"]}}
        cls.reference["identity"].update(artifact_tensor_digests={item["name"]: item["blake3"] for item in manifest["tensors"]},
            fixture_blake3=blake3.blake3(cls.fixture_path.read_bytes()).hexdigest())
        cls.reference_path.write_text(json.dumps(cls.reference))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_independent_prefix_and_seven_reference_directions(self):
        actual, q, _ = runtime.load_runtime(self.artifact, self.secret, linear_profile=runtime.ALL_BOUNDARY_LINEAR_PROFILE)
        before_a = audit.normal_snapshot(self.baseline)
        before_c = audit.normal_snapshot(actual)
        p = actual.model.norm._p
        b, d, resources = audit.build_references(self.baseline, actual, p, q)
        self.assertEqual(len(b.model.layers), 1)
        self.assertEqual(len(d.model.layers), 1)
        self.assertEqual(len(self.baseline.model.layers), 2)
        self.assertEqual(len(actual.model.layers), 2)
        self.assertIs(b.model.embed_tokens, self.baseline.model.embed_tokens)
        self.assertIs(d.model.embed_tokens, actual.model.embed_tokens)
        self.assertIs(d.model.layers[0].input_layernorm._p, p)
        self.assertEqual(resources["ideal_coefficient_count"], 7)
        self.assertFalse(hasattr(b, "_aloepri_linear_profile"))
        self.assertFalse(hasattr(d, "_aloepri_linear_profile"))
        for section, member in audit.ROLE_MEMBERS:
            source = getattr(getattr(self.baseline.model.layers[0], section), member)
            b_module = getattr(getattr(b.model.layers[0], section), member)
            d_module = getattr(getattr(d.model.layers[0], section), member)
            self.assertIs(type(b_module), runtime.F64AccumLinear)
            self.assertIsNot(b_module.weight, source.weight)
            self.assertTrue(torch.equal(b_module.weight, source.weight))
            expected = p.T @ source.weight.double() if member in ["o_proj", "down_proj"] else source.weight.double() @ q.T
            self.assertTrue(torch.equal(d_module._audit_coefficient, expected))
            self.assertEqual(d_module._audit_coefficient.dtype, torch.float64)
            self.assertFalse(d_module.state_dict())
        ids = self.fixture["prompts"][0] + self.fixture["continuation_ids"]
        a_nodes, _ = trace_first_block(self.baseline, ids)
        b_nodes, _ = trace_first_block(b, ids)
        c_nodes, _ = trace_first_block(actual, ids)
        d_nodes, _ = trace_first_block(d, ids)
        self.assertEqual(list(a_nodes), list(b_nodes))
        self.assertEqual(list(c_nodes), list(d_nodes))
        self.assertTrue(torch.equal(a_nodes["embedding"], b_nodes["embedding"]))
        self.assertTrue(torch.equal(c_nodes["embedding"], d_nodes["embedding"]))
        self.assertTrue(all(value.dtype == torch.float32 and value.shape[-1] == 8 for value in b_nodes.values()))
        self.assertTrue(all(value.dtype == torch.float32 and value.shape[-1] == 12 for value in d_nodes.values()))
        self.assertEqual(before_a, audit.normal_snapshot(self.baseline))
        self.assertEqual(before_c, audit.normal_snapshot(actual))
        with self.assertRaises(ValueError):
            runtime.linear_profile_summary(b)
        with self.assertRaises(ValueError):
            runtime.linear_profile_summary(d)
        with self.assertRaises((ValueError, AttributeError)):
            gates.run_gates(b, actual, q, {}, self.fixture)
        self.assertEqual(before_a, audit.normal_snapshot(self.baseline))

    def test_error_chain_signed_cancellation_and_unrounded_recovery(self):
        a = torch.tensor([1.0, -2.0], dtype=torch.float32)
        b = a + torch.tensor([3.0, -4.0])
        d = a.double() + torch.tensor([0.01, -0.02], dtype=torch.float64)
        c = a.double() + torch.tensor([0.02, -0.03], dtype=torch.float64)
        report, vectors = audit.audit_error_budget(a, b, c, d)
        reconstructed = sum(vectors[name] for name in ["B_source_arithmetic", "D_expanded_representation", "C_actual_weight_encoding"])
        self.assertTrue(torch.allclose(reconstructed, c - a.double(), atol=1e-15, rtol=0))
        self.assertLess(report["reconstruction_max_abs"], 1e-15)
        self.assertGreater(sum(row["max_abs"] for row in report["components"].values()), report["comparisons"]["A_vs_C"]["max_abs"])
        self.assertLess(report["same_coordinate_reconstruction_error"], 1e-15)
        q = torch.tensor([[1.0], [0.25]], dtype=torch.float64)
        expanded_c = torch.tensor([[1.0, 1e-7]], dtype=torch.float32)
        expanded_d = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        recovered_c, recovered_d = expanded_c.double() @ q, expanded_d.double() @ q
        report, _ = audit.audit_error_budget(torch.ones(1, 1), torch.ones(1, 1), recovered_c, recovered_d)
        self.assertGreater(report["comparisons"]["A_vs_C"]["max_abs"], 0.0)
        self.assertTrue(torch.equal(recovered_c.float(), torch.ones(1, 1)))
        for invalid in [torch.tensor([float("nan"), 0.0]), torch.ones(3)]:
            with self.assertRaises(ValueError):
                audit.audit_error_budget(a, b, invalid, d)

    def test_identity_rejects_wrong_profiles_sequences_hashes_and_fixture_bytes(self):
        args = (self.source, self.artifact, self.secret, self.fixture_path, self.reference_path)
        fixture, reference, identity = audit.audit_identity(*args)
        self.assertEqual(fixture, self.fixture)
        self.assertEqual(identity["secret_id"], self.reference["identity"]["secret_id"])
        original = self.reference_path.read_bytes()
        try:
            for kind in ["profile", "identity", "digest", "sequence", "formal", "construction"]:
                changed = copy.deepcopy(self.reference)
                if kind == "profile": changed["execution_profile"] = runtime.NATIVE_LINEAR_PROFILE
                elif kind == "identity": changed["identity"]["secret_id"] = "00" * 32
                elif kind == "digest": changed["identity"]["artifact_tensor_digests"]["lm_head.weight"] = "00" * 32
                elif kind == "sequence": changed["profiles"][runtime.ALL_BOUNDARY_LINEAR_PROFILE]["prompts"].pop()
                elif kind == "formal": changed["G3"] = "pass"
                else: changed["construction_profile"] = runtime.ALGORITHM
                self.reference_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): audit.audit_identity(*args)
        finally:
            self.reference_path.write_bytes(original)
        original_fixture = self.fixture_path.read_bytes()
        try:
            self.fixture_path.write_bytes(original_fixture + b"\n")
            with self.assertRaises(ValueError): audit.audit_identity(*args)
        finally:
            self.fixture_path.write_bytes(original_fixture)

    def test_complete_tiny_audit_is_not_a_formal_gate_and_preserves_A_C(self):
        model, q, _ = runtime.load_runtime(self.artifact, self.secret)
        private = {}
        report = audit.execute_audit(self.baseline, model, q, self.fixture, self.reference, private)
        self.assertTrue(report["audit_completed"])
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["ac10_satisfied"])
        self.assertEqual([report[key] for key in ["formal_G2", "G3", "G4"]], ["not_run"] * 3)
        self.assertTrue(report["normal_A_C_unchanged_after_references"])
        self.assertTrue(report["head_and_later_layers_not_executed"])
        self.assertEqual(len(report["prompts"]), 3)
        self.assertEqual(len(report["prompts"][0]["required_nodes"]), 8)
        self.assertTrue(private)
        self.assertEqual(report["path_definitions"]["D"]["parameter_storage"], "private_F64_theory_coefficients_not_an_artifact")
        self.assertFalse(report["path_definitions"]["B"]["admissible_acceptance_target"])


if __name__ == "__main__":
    unittest.main()
