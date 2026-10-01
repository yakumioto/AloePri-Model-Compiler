import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaForCausalLM

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'demo'))
import aloepri_runtime as runtime
import hidden_expansion_demo as gates
import test_keymat_float_semantics_audit as fixture_module
from keymat_numerical_diagnostics import trace_first_block
from token_permutation_demo import load_local_llama_model


class SignedNullRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        owner=fixture_module.FloatSemanticsAuditTests;owner.setUpClass();cls.owner=owner
        cls.root,cls.source,cls.fixture=owner.root,owner.source,owner.fixture
        cls.owner_backend=owner.baseline.config._attn_implementation
        cls.baseline,_=load_local_llama_model(cls.source,LlamaForCausalLM,torch)
        private=cls.root/'signed-private';private.mkdir();cls.secret=private/'secret.json';cls.artifact=cls.root/'signed-artifact'
        result=subprocess.run([str(ROOT/'target/debug/aloepri'),'transform',str(cls.source),'--output',str(cls.artifact),'--method','aloepri-keymat','--keymat-algorithm',runtime.SIGNED_ALGORITHM,'--expansion-size','2','--keymat-lambda','0.3','--keymat-fixture-seed','42','--secret-output',str(cls.secret)],capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stderr)
    @classmethod
    def tearDownClass(cls):cls.owner.tearDownClass()

    def test_actual_signed_profile_preserves_parameters_full_D_and_all_gates(self):
        self.assertEqual(self.baseline.config._attn_implementation,'eager')
        self.assertEqual(self.owner.baseline.config._attn_implementation,self.owner_backend)
        original=self.owner.baseline.state_dict();loaded=self.baseline.state_dict()
        self.assertEqual(set(original),set(loaded))
        for name,value in loaded.items():
            self.assertEqual(value.dtype,torch.float32)
            self.assertEqual(value.shape,original[name].shape)
            self.assertTrue(torch.equal(value.view(torch.int32),original[name].view(torch.int32)),name)
        model,q,keys=runtime.load_runtime(self.artifact,self.secret)
        native=runtime.linear_profile_summary(model);self.assertEqual(native['execution_profile'],runtime.NATIVE_LINEAR_PROFILE)
        parameters=dict(model.named_parameters());bits={name:value.clone() for name,value in model.state_dict().items()}
        installed=runtime.install_linear_profile(model,runtime.SIGNED_ALGORITHM,12,linear_profile=runtime.SIGNED_LINEAR_PROFILE)
        self.assertEqual(installed['wrapped_role_count'],15)
        self.assertTrue(all(row['implementation'] in ['SignedColumnBlockedLinear','SignedRowBlockedLinear'] for row in installed['roles']))
        for name,value in model.named_parameters():self.assertIs(value,parameters[name]);self.assertTrue(torch.equal(value,bits[name]))
        self.assertEqual(set(model.state_dict()),set(bits));self.assertFalse(any(name.endswith('._active_p') for name in model.state_dict()))
        report=gates.run_gates(self.baseline,model,q,keys,self.fixture)
        self.assertEqual([report[name]['status'] for name in ['G1','G2','G3','G4']],['pass']*4,report)
        self.assertEqual(report['G2']['max_abs'],0.0)
        costs=runtime.linear_profile_costs(model)
        self.assertEqual(costs['executed_scope_count'],15)
        self.assertTrue(all(row['core_gemm_calls']==row['aux_gemm_calls']>0 for row in costs['roles']))
        self.assertEqual(runtime.install_linear_profile(model,runtime.SIGNED_ALGORITHM,12,linear_profile=runtime.SIGNED_LINEAR_PROFILE)['wrapped_role_count'],15)
        self.assertGreater(torch.count_nonzero(model.model.embed_tokens.weight[:,8:]).item(),0)

    def test_wrong_algorithm_support_state_and_cross_binding_rejected(self):
        with self.assertRaises(ValueError):runtime.load_runtime(self.owner.artifact,self.owner.secret,linear_profile=runtime.SIGNED_LINEAR_PROFILE)
        with self.assertRaises(ValueError):runtime.load_runtime(self.owner.artifact,self.secret)
        with self.assertRaises(ValueError):runtime.load_runtime(self.artifact,self.owner.secret)
        model,q,_=runtime.load_runtime(self.artifact,self.secret,linear_profile=runtime.SIGNED_LINEAR_PROFILE)
        ids=torch.tensor([[1,4]]);z=model.model.embed_tokens(ids);module=model.model.layers[0].self_attn.q_proj
        bad=z.clone();bad[...,8:][...,~module._active_p]=0.01
        with self.assertRaisesRegex(ValueError,'inactive'):module(bad)
        self.assertEqual(module.core_calls,module.aux_calls)
        broken=model.model.norm._p.clone();broken[0,0]=0.25
        with self.assertRaises(ValueError):runtime.signed_null_support(broken,q)
        model.model.layers[0].mlp.down_proj._active_p=torch.zeros_like(module._active_p)
        with self.assertRaises(ValueError):runtime.linear_profile_summary(model)
        original=self.secret.read_bytes()
        try:
            changed=json.loads(original);changed['algorithm']=runtime.ALGORITHM;self.secret.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):runtime.load_keymat(self.secret)
        finally:self.secret.write_bytes(original)

    def test_source_free_loading_and_cache_only_actual_artifact_parameters(self):
        absent=self.root/'signed-source-unavailable';self.source.rename(absent)
        try:
            with patch.object(LlamaForCausalLM,'from_pretrained',side_effect=AssertionError('SOURCE forbidden')):
                model,q,_=runtime.load_runtime(self.artifact,self.secret,linear_profile=runtime.SIGNED_LINEAR_PROFILE)
                nodes,_=trace_first_block(model,[1,4]);self.assertTrue(all(value.dtype==torch.float32 and value.shape[-1]==12 for value in nodes.values()))
                prefill,decode=gates.cache_logits(model,[1,4],[5,8]);self.assertEqual(prefill.shape[-1],19);self.assertEqual(len(decode),2)
            result=subprocess.run([sys.executable,str(ROOT/'demo/aloepri_runtime.py'),'--artifact',str(self.artifact),'--secret',str(self.secret),'--linear-profile',runtime.SIGNED_LINEAR_PROFILE,'--token-ids','1','4','--max-new-tokens','0'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads(result.stdout);self.assertTrue(output['finite_logits']);self.assertEqual(output['installed_linear_profile']['wrapped_role_count'],15)
        finally:absent.rename(self.source)


if __name__=='__main__':unittest.main()
