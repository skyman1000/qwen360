"""Zero-gated residual world attention; text sequence and base weights unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .world_condition import WorldCondition
from .spatial_world import token_geometry, attention_bias


def type_log_prior(counts, device):
    """Equal prior mass per nonempty type, before learned attention logits."""
    return torch.cat([torch.full((1,n,1),-math.log(n),device=device,dtype=torch.float32)
                      for n in counts if n>0],dim=1)


class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, width=256, heads=8, residual_scale='none', spatial_height=None, balance_types=False):
        super().__init__()
        self.heads=heads
        self.width=width
        self.spatial_height=spatial_height
        self.balance_types=balance_types
        self.residual_scale=residual_scale
        self.record_diagnostics=False
        self.diagnostics={}
        self.audit_counts=None
        self.attention_audit=None
        self.norm=nn.LayerNorm(hidden_dim)
        self.world_norm=nn.LayerNorm(width)
        self.q=nn.Linear(hidden_dim,width,bias=False)
        self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False)
        self.out=nn.Linear(width,hidden_dim,bias=False)
        self.gate=nn.Parameter(torch.zeros(()))

    def forward(self, hidden, tokens):
        # The revised branch uses the same attention precision in training and
        # sampling. Legacy checkpoints retain their previous behaviour.
        if self.residual_scale=='hidden_rms':
            with torch.autocast(hidden.device.type,dtype=torch.bfloat16,enabled=hidden.is_cuda):
                return self._forward(hidden,tokens)
        return self._forward(hidden,tokens)

    def _forward(self, hidden, tokens):
        # Trainable branch stays float32; attention uses autocast if enabled.
        b,n,_=hidden.shape; m=tokens.shape[1]
        bias=None
        if self.spatial_height is not None:
            # Metadata travels with tokens through checkpoint inputs, so a
            # shuffled world also uses its own positions on recomputation.
            with torch.autocast(hidden.device.type,enabled=False):
                bias=attention_bias(tokens[...,self.width:].float(),self.spatial_height,n)
            tokens=tokens[...,:self.width]
        elif self.balance_types:
            bias=tokens[...,self.width].float()[:,None,:].expand(b,n,m)
            tokens=tokens[...,:self.width]
        q=self.q(self.norm(hidden.float()))
        world=self.world_norm(tokens.float())
        k,v=self.k(world),self.v(world)
        def heads(t,length):
            return t.reshape(b,length,self.heads,-1).transpose(1,2)
        if self.audit_counts is not None and self.attention_audit is None:
            # Read-only first-call audit, using actual image queries. This does
            # not replace SDPA or modify the prediction. Limit query memory.
            with torch.no_grad(), torch.autocast(hidden.device.type,enabled=False):
                indices=torch.linspace(0,n-1,min(64,n),device=hidden.device).long()
                aq=heads(q,n)[:,:,indices].float();ak=heads(k,m).float()
                logits=(aq@ak.transpose(-1,-2))/(aq.shape[-1]**.5)
                if bias is not None:logits=logits+bias[:,None,indices].float()
                probs=logits.softmax(-1)
                offset=0; groups={}
                for name,count in self.audit_counts.items():
                    part=probs[...,offset:offset+count]
                    groups[name]=dict(tokens=count,uniform_mass=count/m,
                                      type_prior_mass=(1/sum(c>0 for c in self.audit_counts.values()) if count else 0.) if self.balance_types else count/m,
                                      attention_mass=float(part.sum(-1).mean()),
                                      per_head_attention_mass=part.sum(-1).mean((0,2)).cpu().tolist(),
                                      value_mean_vector=v[:,offset:offset+count].float().mean((0,1)).cpu().tolist() if count else None)
                    offset+=count
                attended=(probs@heads(v,m).float()).transpose(1,2).reshape(b,len(indices),-1)
                def summary(t):
                    return dict(mean_vector=t.float().mean((0,1)).cpu().tolist(),
                                within_sequence_std_rms=float(t.float().std(1,unbiased=False).square().mean().sqrt()))
                self.attention_audit=dict(query_count=len(indices),groups=groups,
                    entropy=float(-(probs*probs.clamp_min(1e-30).log()).sum(-1).mean()),
                    max_probability=float(probs.max(-1).values.mean()),
                    normalized_world=summary(world),values=summary(v),attended=summary(attended))
        attn=F.scaled_dot_product_attention(heads(q,n),heads(k,m),heads(v,m),
                                          attn_mask=bias[:,None].to(q.dtype) if bias is not None else None,dropout_p=0.)
        update=self.out(attn.transpose(1,2).reshape(b,n,-1))
        update=update.float()
        # A normalized query does not normalize the residual stream. Express
        # the new residual in that stream's units; detach the scale so the
        # branch does not learn through the magnitude of frozen activations.
        if self.residual_scale=='hidden_rms':
            scale=hidden.detach().float().square().mean(-1,keepdim=True).sqrt()
        else:
            scale=1.
        residual=torch.tanh(self.gate)*scale*update
        output=(hidden.float()+residual).to(hidden.dtype)
        if self.record_diagnostics:
            with torch.no_grad():
                def rms(t):return float(t.float().square().mean().sqrt())
                self.diagnostics=dict(hidden_rms=rms(hidden),raw_update_rms=rms(update),
                                      scaled_update_rms=rms(scale*update),residual_rms=rms(residual),
                                      effective_delta_rms=rms(output.float()-hidden.float()),
                                      changed_fraction=float((output!=hidden).float().mean()))
        return output


class GatedWorldCondition(WorldCondition):
    def __init__(self, hidden_dim, block_indices=(20,30,40), width=256, residual_scale='none', spatial_height=None, balance_types=False):
        super().__init__(width,width)
        if balance_types and spatial_height is not None:
            raise ValueError('Test type balancing independently from the spatial prior')
        self.spatial_height=spatial_height
        self.balance_types=balance_types
        self.branches=nn.ModuleDict({str(i):GatedAttention(hidden_dim,width,residual_scale=residual_scale,
                                                        spatial_height=spatial_height,balance_types=balance_types) for i in block_indices})
        self.current_tokens=None

    def forward(self,state):
        tokens=super().forward(state)
        if self.spatial_height is not None:
            tokens=torch.cat([tokens,token_geometry(state,tokens.device)],-1)
        elif self.balance_types:
            counts=(len(state['objects']),4,len(state['relations']))
            tokens=torch.cat([tokens,type_log_prior(counts,tokens.device)],-1)
        return tokens

    def attach(self, transformer):
        selected={}
        for key,branch in self.branches.items():
            block=transformer.transformer_blocks[int(key)]
            original=block.forward
            # Modules remain owned by this conditioner, not frozen transformer.
            def wrapped(*args,_original=original,_branch=branch,world_tokens=None,**kwargs):
                text,hidden=_original(*args,**kwargs)
                tokens=self.current_tokens if world_tokens is None else world_tokens
                if tokens is not None:
                    hidden=_branch(hidden,tokens)
                return text,hidden
            block.forward=wrapped
            selected[block]=True
        original_checkpoint=getattr(transformer,'_gradient_checkpointing_func',None)
        if original_checkpoint is None:
            return  # Inference-only pipeline, checkpointing was never enabled.
        def with_world(function,*inputs,**kwargs):
            if function not in selected or self.current_tokens is None:
                return original_checkpoint(function,*inputs,**kwargs)
            # World tokens are explicit checkpoint inputs. Backward recomputation
            # does not depend on a later update of the runtime condition.
            tokens=self.current_tokens
            def run(*packed):
                return function(*packed[:-1],world_tokens=packed[-1])
            return checkpoint(run,*inputs,tokens,use_reentrant=False)
        transformer._gradient_checkpointing_func=with_world

    def gate_values(self):
        return {k:float(torch.tanh(v.gate.detach())) for k,v in self.branches.items()}
