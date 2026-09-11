"""Source ablations: prevent global-vote shortcut and activate robust weighting."""
import torch
from torch import nn
from skin_mskcc_vote import PatchVotes


class GlobalColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(36,512),nn.SiLU(),nn.Linear(512,768),nn.SiLU(),
                               nn.Linear(768,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU())
        self.head=nn.Linear(256,3)

    def forward(self,x,*,details=False):
        h=self.body(x);out=self.head(h)
        return (out,torch.zeros(len(x),device=x.device),h) if details else out


class VoteAblation(PatchVotes):
    def __init__(self,target_std,*,local_only,steps):
        super().__init__(target_std,steps);self.local_only=local_only

    def forward(self,x,*,details=False):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        broadcast=context[:,None].expand(-1,h.shape[1],-1)
        prediction=self.votes(torch.cat([h,broadcast],-1))
        local=(self.votes(torch.cat([h,torch.zeros_like(broadcast)],-1))[...,:3]
               if self.local_only else prediction[...,:3])
        base=prediction[...,3].softmax(1);weight=base
        out=(local*weight[...,None]).sum(1)
        for _ in range(self.steps):
            residual=torch.linalg.vector_norm((local-out[:,None])*self.target_std,dim=-1)
            weight=base*(.5/residual.clamp_min(1e-6)).clamp(max=1.)
            weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-12)
            out=(local*weight[...,None]).sum(1)
        risk=(((local-out[:,None])*self.target_std).square().sum(-1)*weight).sum(1).clamp_min(0).sqrt()
        return (out,risk,context) if details else out
