import sys
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'demo'))
import keymat_signed_null_prototype as prototype
from keymat_numerical_diagnostics import trace_first_block
from keymat_float_semantics_audit import normal_snapshot
from hidden_expansion_demo import compare_nodes
from aloepri_runtime import SUPPORTED_ALGORITHMS


def small_material(d,h):
    r=h//2;big_d=d+2*h
    signs=torch.ones(d,dtype=torch.float64);signs[::2]=-1
    c=torch.arange(1,d*r+1,dtype=torch.float64).view(d,r)/13
    e=torch.arange(2,d*r+2,dtype=torch.float64).view(d,r)/17
    f=torch.arange(3,r*d+3,dtype=torch.float64).view(r,d)/19
    n=torch.arange(4,r*d+4,dtype=torch.float64).view(r,d)/23
    permutation=torch.arange(2*h-1,-1,-1)
    t=torch.zeros(2*h,2*h,dtype=torch.float64)
    t[torch.arange(2*h),permutation]=torch.where(torch.arange(2*h)%2==0,1.,-1.).double()
    pre_p=torch.cat([torch.diag(signs),torch.cat([torch.zeros(d,r),c],dim=1),torch.cat([e,torch.zeros(d,r)],dim=1)],dim=1)
    pre_q=torch.cat([torch.diag(signs),torch.cat([f,torch.zeros(r,d)],dim=0),torch.cat([torch.zeros(r,d),n],dim=0)],dim=0)
    z=torch.block_diag(torch.eye(d,dtype=torch.float64),t)
    return pre_p@z,z.T@pre_q,(c,e,f,n,t,pre_p,pre_q,z)


class SignedNullPrototypeTests(unittest.TestCase):
    def test_block_structure_orthogonality_cross_terms_and_nontriviality(self):
        for d,h in [(1,2),(4,8),(8,2)]:
            p,q,(c,e,f,n,t,pre_p,pre_q,z)=small_material(d,h)
            active,inactive,report=prototype.validate_support(p,q)
            self.assertTrue(torch.equal(z@z.T,torch.eye(d+2*h,dtype=torch.float64)))
            self.assertTrue(torch.equal(t@t.T,torch.eye(2*h,dtype=torch.float64)))
            self.assertTrue(torch.equal(pre_p[:,d:d+h]@pre_q[d:d+h],torch.zeros(d,d,dtype=torch.float64)))
            self.assertTrue(torch.equal(pre_p[:,d+h:]@pre_q[d+h:],torch.zeros(d,d,dtype=torch.float64)))
            self.assertEqual(report['max_abs_pq_error'],0)
            self.assertTrue(torch.all(active^inactive).item())
            self.assertGreater(torch.count_nonzero(p[:,d:]).item(),0)
            corrupt=p.clone();corrupt[0,0]=0.5
            with self.assertRaises(ValueError):prototype.validate_support(corrupt,q)
            corrupt=q.clone();corrupt[-1,0]=float('nan')
            with self.assertRaises(ValueError):prototype.validate_support(p,corrupt)

    def test_fp32_physical_tiles_both_compute_and_reject_nonzero_inactive_state(self):
        p,q,_=small_material(4,2);active,_,_=prototype.validate_support(p,q)
        torch.manual_seed(91)
        x=torch.randn(2,3,4);w=torch.randn(5,4);u=torch.randn(2,3,3);out_weight=torch.randn(4,3)
        encoded=(x.double()@p).float()
        self.assertGreater(torch.count_nonzero(encoded[...,4:]).item(),0)
        input_weight=prototype.transform_weight(w,p,q,'input')
        module=prototype.SignedColumnBlockedLinear(input_weight,4,active)
        result=module(encoded)
        self.assertTrue(torch.equal(result,torch.nn.functional.linear(x,w)))
        self.assertEqual(module.core_calls,module.aux_calls)
        output_weight=prototype.transform_weight(out_weight,p,q,'output')
        output=prototype.SignedRowBlockedLinear(output_weight,4,active)(u)
        self.assertEqual(output.shape[-1],8)
        self.assertEqual(output.dtype,torch.float32)
        recovered=output.double()@q
        self.assertTrue(torch.equal(recovered.float(),torch.nn.functional.linear(u,out_weight)))
        corrupted=encoded.clone();corrupted[...,4:][...,~active]=0.25
        with self.assertRaisesRegex(ValueError,'inactive'):
            module(corrupted)
        self.assertEqual(module.core_calls,module.aux_calls)
        self.assertEqual(module.core_calls,2)
        bad_weight=input_weight.clone();bad_weight[:,4:][:,active]=1
        with self.assertRaises(ValueError):prototype.SignedColumnBlockedLinear(bad_weight,4,active)
        with self.assertRaises(ValueError):module(encoded.double())

    def test_full_D_first_block_and_source_model_not_replaced(self):
        torch.manual_seed(119)
        config=LlamaConfig(hidden_size=4,intermediate_size=7,num_hidden_layers=2,num_attention_heads=1,num_key_value_heads=1,vocab_size=13)
        source=LlamaForCausalLM(config).float().eval()
        p,q,_=small_material(4,2)
        before=normal_snapshot(source)
        model,math,changed=prototype.build_prototype(source,p,q)
        ids=[1,2,3,4]
        original,_=trace_first_block(source,ids)
        actual,_=trace_first_block(model,ids)
        compared=compare_nodes(original,actual,q)
        self.assertEqual(compared['status'],'pass',compared)
        self.assertTrue(all(value.dtype==torch.float32 and value.shape[-1]==8 for value in actual.values()))
        self.assertTrue(all(row['data_changed'] for row in changed))
        self.assertEqual(before,normal_snapshot(source))
        self.assertGreater(torch.count_nonzero(model.model.embed_tokens.weight[:,4:]).item(),0)
        head=prototype.head_fidelity_check(source,p,q,prototype.validate_support(p,q)[0],ids)
        self.assertTrue(head['pass'])
        self.assertTrue(head['core_and_auxiliary_both_computed'])


if __name__=='__main__':unittest.main()
