"""Matched confidence aggregation versus unrolled Huber patch aggregation."""
import torch
from torch import nn


def aggregate(votes, logits, target_std, steps=0):
    base=logits.softmax(1)
    estimate=(votes*base[...,None]).sum(1)
    weight=base
    for _ in range(steps):
        residual=torch.linalg.vector_norm((votes-estimate[:,None])*target_std,dim=-1)
        weight=base*(5./residual.clamp_min(1e-6)).clamp(max=1.)
        weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-12)
        estimate=(votes*weight[...,None]).sum(1)
    dispersion=((votes-estimate[:,None])*target_std).square().sum(-1)
    return estimate, (dispersion*weight).sum(1).clamp_min(0).sqrt()


class PatchVotes(nn.Module):
    def __init__(self, target_std, steps=0):
        super().__init__();self.steps=steps
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32))

    def forward(self, x, *, details=False):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],dim=1))
        vote=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],dim=-1))
        out,risk=aggregate(vote[...,:3],vote[...,3],self.target_std,self.steps)
        return (out,risk,context) if details else out
