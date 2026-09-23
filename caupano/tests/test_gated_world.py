import unittest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from qwen_pano.caupano.gated_world import GatedAttention, GatedWorldCondition, type_log_prior
from qwen_pano.caupano.spatial_world import attention_bias
from qwen_pano.caupano.tools.parameter_report import training_parameter_report
import math


class ToyBlock(nn.Module):
    def forward(self, hidden, text):
        return text, hidden * 1.01


class GatedWorldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_frozen_base_passes_gradients_without_weight_updates(self):
        base=nn.Linear(4,4).requires_grad_(False)
        world=nn.Linear(4,4)
        optimizer=torch.optim.AdamW(world.parameters(),lr=.01)
        report=training_parameter_report(base,world,optimizer)
        self.assertEqual(report['base_trainable_parameters'],0)
        self.assertEqual(report['optimizer_parameters'],20)
        before={name:p.detach().clone() for name,p in base.named_parameters()}
        base(world(torch.randn(2,4))).square().mean().backward()
        self.assertGreater(float(world.weight.grad.abs().sum()),0.)
        optimizer.step()
        for name,p in base.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        wrong=torch.optim.AdamW([*world.parameters(),*base.parameters()])
        with self.assertRaises(RuntimeError):training_parameter_report(base,world,wrong)

    def test_type_balance_is_invariant_to_repeating_a_group(self):
        branch=GatedAttention(16,width=16,heads=4,residual_scale='hidden_rms',balance_types=True)
        hidden=torch.randn(1,8,16)
        features=torch.randn(1,7,16,requires_grad=True)
        tokens=torch.cat([features,type_log_prior((2,4,1),'cpu')],-1)
        self.assertTrue(torch.equal(branch(hidden,tokens),hidden))
        with torch.no_grad():branch.gate.fill_(.05)
        expected=branch(hidden,tokens)
        repeated=torch.cat([features[:,:6],features[:,6:].repeat(1,4,1)],1)
        repeated=torch.cat([repeated,type_log_prior((2,4,4),'cpu')],-1)
        actual=branch(hidden,repeated)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward()
        self.assertGreater(float(features.grad.abs().sum()),0.)

    def test_type_prior_handles_no_relations(self):
        prior=type_log_prior((2,4,0),'cpu')
        self.assertEqual(prior.shape,(1,6,1))
        probability=prior[0,:,0].softmax(0)
        torch.testing.assert_close(probability[:2].sum(),torch.tensor(.5))

    def test_spatial_prior_pixel_centres_and_padding(self):
        # H=2, W=4: upper-left ray has lon=-3pi/4, lat=pi/4.
        meta=torch.tensor([[[-.5,math.sqrt(.5),-.5,4.,0.]]])
        bias=attention_bias(meta,2,8)
        self.assertEqual(int(bias[0,:,0].argmax()),0)
        padded=attention_bias(meta,2,12).reshape(1,2,6,1)
        plain=bias.reshape(1,2,4,1)
        torch.testing.assert_close(padded[:,:,1:5],plain)
        torch.testing.assert_close(padded[:,:,0],plain[:,:,-1])
        torch.testing.assert_close(padded[:,:,-1],plain[:,:,0])
        # Object exactly at the back seam: both seam-adjacent columns agree.
        seam=attention_bias(torch.tensor([[[0.,0.,-1.,4.,0.]]]),2,8).reshape(2,4)
        torch.testing.assert_close(seam[:,0],seam[:,-1])

    def test_count_prior_removes_token_multiplicity_advantage(self):
        counts=[2,4,10]
        meta=torch.tensor([[[0.,0.,0.,0.,-math.log(n)] for n in counts for _ in range(n)]])
        prob=attention_bias(meta,2,8).softmax(-1)[0,0]
        offset=0
        for n in counts:
            torch.testing.assert_close(prob[offset:offset+n].sum(),torch.tensor(1/3))
            offset+=n

    def test_spatial_branch_uses_positions_and_backpropagates(self):
        branch=GatedAttention(16,width=16,heads=4,residual_scale='hidden_rms',spatial_height=2)
        hidden=torch.randn(1,8,16)
        features=torch.randn(1,2,16,requires_grad=True)
        metadata=torch.tensor([[[1.,0.,0.,4.,0.],[-1.,0.,0.,4.,0.]]])
        tokens=torch.cat([features,metadata],-1)
        self.assertTrue(torch.equal(branch(hidden,tokens),hidden))
        with torch.no_grad():branch.gate.fill_(.05)
        output=branch(hidden,tokens)
        changed=branch(hidden,torch.cat([features,metadata.flip(1)],-1))
        self.assertGreater(float((output-changed).abs().max()),1e-6)
        output.square().mean().backward()
        self.assertGreater(float(features.grad.abs().sum()),0.)

    def test_zero_gate_and_gradient_startup(self):
        branch=GatedAttention(16,width=16,heads=4,residual_scale='hidden_rms')
        hidden=torch.randn(1,5,16)*1e6
        tokens=torch.randn(1,3,16,requires_grad=True)
        output=branch(hidden,tokens)
        self.assertTrue(torch.equal(output,hidden))
        (output/1e6).square().mean().backward()
        self.assertGreater(float(branch.gate.grad.abs()),0.)
        self.assertEqual(float(tokens.grad.abs().sum()),0.)
        branch.zero_grad(set_to_none=True);tokens.grad=None
        with torch.no_grad():branch.gate.fill_(.01)
        (branch(hidden,tokens)/1e6).square().mean().backward()
        self.assertGreater(float(tokens.grad.abs().sum()),0.)

    def test_attention_audit_does_not_change_output(self):
        branch=GatedAttention(16,width=16,heads=4,residual_scale='hidden_rms')
        with torch.no_grad():branch.gate.fill_(.03)
        hidden=torch.randn(1,8,16);tokens=torch.randn(1,7,16)
        expected=branch(hidden,tokens)
        branch.audit_counts=dict(objects=2,layout=4,relations=1)
        actual=branch(hidden,tokens)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        total=sum(group['attention_mass'] for group in branch.attention_audit['groups'].values())
        self.assertAlmostEqual(total,1.,places=5)

    def test_residual_scale_and_bfloat16_effect(self):
        branch=GatedAttention(16,width=16,heads=4,residual_scale='hidden_rms')
        with torch.no_grad():branch.gate.fill_(.05)
        hidden=torch.randn(1,5,16)*100
        tokens=torch.randn(1,3,16)
        small=branch(hidden,tokens)
        large=branch(hidden*10000,tokens)/10000
        torch.testing.assert_close(small,large,rtol=2e-5,atol=2e-5)
        quantized=(hidden*10000).bfloat16()
        self.assertFalse(torch.equal(branch(quantized,tokens),quantized))

    def test_checkpoint_keeps_explicit_world_tokens(self):
        base=nn.Module()
        base.transformer_blocks=nn.ModuleList([ToyBlock()])
        base._gradient_checkpointing_func=lambda fn,*args: checkpoint(fn,*args,use_reentrant=False)
        world=GatedWorldCondition(16,block_indices=(0,),width=16,residual_scale='hidden_rms')
        world.attach(base)
        with torch.no_grad():world.branches['0'].gate.fill_(.03)
        hidden=torch.randn(1,5,16)
        tokens=torch.randn(1,3,16,requires_grad=True)
        world.current_tokens=tokens
        _,direct=base.transformer_blocks[0](hidden,hidden)
        expected=torch.autograd.grad(direct.square().mean(),tokens)[0]
        _,output=base._gradient_checkpointing_func(base.transformer_blocks[0],hidden,hidden)
        # Recompute must not silently read a changed runtime condition.
        world.current_tokens=torch.randn_like(tokens)
        actual=torch.autograd.grad(output.square().mean(),tokens)[0]
        torch.testing.assert_close(actual,expected)
        self.assertGreater(float(actual.abs().sum()),0.)


if __name__=='__main__':unittest.main()
