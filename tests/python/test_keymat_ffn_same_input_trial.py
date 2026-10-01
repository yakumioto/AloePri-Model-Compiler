import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "demo"))
import aloepri_runtime as runtime
import keymat_ffn_same_input_trial as precheck
import test_keymat_float_semantics_audit as audit_fixture
from keymat_float_semantics_audit import normal_snapshot


class SameInputPrecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = audit_fixture.FloatSemanticsAuditTests
        fixture.setUpClass()
        cls.fixture_owner = fixture
        cls.root, cls.source, cls.artifact, cls.secret = fixture.root, fixture.source, fixture.artifact, fixture.secret
        cls.baseline, cls.fixture, cls.fixture_path = fixture.baseline, fixture.fixture, fixture.fixture_path
        cls.reference = {"audit_completed": True, "diagnostic_only": True, "ac10_satisfied": False,
            "formal_G2": "not_run", "G3": "not_run", "G4": "not_run",
            "path_definitions": {"A": {"kind": "native_source_target"}}, "normal_A_C_unchanged_after_references": True,
            "identity": fixture.reference["identity"], "prompts": [{"prompt_index": index} for index in range(3)],
            "native_control_replay": fixture.reference["profiles"][runtime.NATIVE_LINEAR_PROFILE]["prompts"]}
        cls.reference_path = cls.root / "precheck-reference-audit.json"
        cls.reference_path.write_text(json.dumps(cls.reference))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_owner.tearDownClass()

    def test_same_input_native_bits_candidate_clone_and_no_profile_promotion(self):
        model, q, _ = runtime.load_runtime(self.artifact, self.secret)
        before_a, before = normal_snapshot(self.baseline), normal_snapshot(model)
        report = precheck.same_input_precheck(self.baseline, model, q, self.fixture, self.reference)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["production_profile_added"])
        self.assertTrue(report["normal_A_native_v2_unchanged"])
        self.assertTrue(report["candidate_clone_state_f32_bits_preserved"])
        self.assertEqual(len(report["prompts"]), 3)
        self.assertEqual(len(report["prompts"][0]["candidate_required_nodes"]["nodes"]), 8)
        self.assertTrue(all(row["native_mlp_bits_match_trace"] and row["native_block_bits_match_trace"] and row["first_six_nodes_native_bits_unchanged"] for row in report["prompts"]))
        self.assertEqual([report[field] for field in ["formal_G2", "G3", "G4"]], ["not_run"] * 3)
        self.assertEqual(before_a, normal_snapshot(self.baseline))
        self.assertEqual(before, normal_snapshot(model))
        self.assertNotIn(precheck.PROPOSED_PROFILE, runtime.SUPPORTED_LINEAR_PROFILES)
        self.assertTrue(all(cost["calls"] == 3 for cost in report["candidate_clone_costs"].values()))

    def test_gate_requires_all_sequences_and_exact_native_replay(self):
        rows = [{"candidate_required_nodes": {"status": "pass"}, "native_mlp_bits_match_trace": True, "native_block_bits_match_trace": True} for _ in range(3)]
        self.assertTrue(precheck.precheck_passes(rows))
        failed = copy.deepcopy(rows)
        failed[2]["candidate_required_nodes"]["status"] = "fail"
        self.assertFalse(precheck.precheck_passes(failed))
        failed = copy.deepcopy(rows)
        failed[0]["native_block_bits_match_trace"] = False
        self.assertFalse(precheck.precheck_passes(failed))
        model, q, _ = runtime.load_runtime(self.artifact, self.secret)
        original = model.model.layers[0].mlp.forward
        calls = 0
        def changed_replay(inputs):
            nonlocal calls
            calls += 1
            value = original(inputs)
            return value if calls == 1 else value + 8e-5
        with patch.object(model.model.layers[0].mlp, "forward", side_effect=changed_replay):
            with self.assertRaisesRegex(ValueError, "bitmatch"):
                precheck.same_input_precheck(self.baseline, model, q, self.fixture, self.reference)
        self.assertNotIn(precheck.PROPOSED_PROFILE, runtime.SUPPORTED_LINEAR_PROFILES)

    def test_strict_audit_identity_and_nonfinite_candidate_rejection(self):
        args = (self.source, self.artifact, self.secret, self.fixture_path, self.reference_path)
        fixture, reference, identity = precheck.check_precheck_identity(*args)
        self.assertEqual(identity["secret_id"], self.reference["identity"]["secret_id"])
        original_reference = self.reference_path.read_bytes()
        try:
            for field, value in [("audit_completed", False), ("ac10_satisfied", True), ("G3", "pass")]:
                changed = copy.deepcopy(self.reference)
                changed[field] = value
                self.reference_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): precheck.check_precheck_identity(*args)
        finally:
            self.reference_path.write_bytes(original_reference)
        original_fixture = self.fixture_path.read_bytes()
        try:
            self.fixture_path.write_bytes(original_fixture + b"\n")
            with self.assertRaises(ValueError): precheck.check_precheck_identity(*args)
        finally:
            self.fixture_path.write_bytes(original_fixture)
        model, q, _ = runtime.load_runtime(self.artifact, self.secret)
        with patch.object(runtime.F64AccumLinear, "forward", autospec=True, side_effect=lambda module, inputs: torch.full((*inputs.shape[:-1], module.out_features), float("nan"))):
            with self.assertRaises(ValueError):
                precheck.same_input_precheck(self.baseline, model, q, self.fixture, self.reference)
        self.assertIs(type(model.model.layers[0].mlp.down_proj), torch.nn.Linear)
        self.assertNotIn(precheck.PROPOSED_PROFILE, runtime.SUPPORTED_LINEAR_PROFILES)


if __name__ == "__main__":
    unittest.main()
