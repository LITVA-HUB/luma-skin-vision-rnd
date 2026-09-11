"""Matched conventional spatial processing and anchored latent graph diffusion."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['plain','conv1','conv3','graph1','graph3','graph3_scrambled']


def grid_adjacency():
    ids=torch.arange(64);row=ids//8;col=ids%8
    return ((row[:,None]-row[None]).abs()+(col[:,None]-col[None]).abs()==1).float()


def screened_diffusion(h,anchor,weights,steps):
    """Jacobi iteration for (diag(anchor)+L_weights) z=anchor*h.

    Positive anchors and nonnegative weights give a convex-combination update.
    Finite steps approximate this quadratic latent problem, not skin physics.
    """
    z=h;denominator=anchor+weights.sum(-1,keepdim=True)
    for _ in range(steps):z=(anchor*h+weights@z)/denominator
    return z


class SpatialColor(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.steps=0 if arm=='plain' else (1 if arm.endswith('1') else 3)
        # Core initialized first and identically to the historical plain PatchVotes.
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        # All arms store the same modules, with active counts reported separately.
        self.relation=nn.Linear(256,256,bias=False)
        self.anchor=nn.Linear(256,1)
        self.temperature=nn.Parameter(torch.tensor(0.))
        self.coupling=nn.Parameter(torch.tensor(0.))
        self.spatial=nn.Conv2d(256,256,3,padding=1,groups=256)
        self.register_buffer('adjacency',grid_adjacency())
        permutation=torch.randperm(64,generator=torch.Generator().manual_seed(20260911))
        self.register_buffer('permutation',permutation)
        self.register_buffer('inverse_permutation',torch.argsort(permutation))

    def active_parameters(self):
        groups=[self.local,self.context,self.votes]
        extra=0
        if self.arm!='plain':groups.append(self.relation)
        if self.arm.startswith('graph'):groups.append(self.anchor);extra=2
        if self.arm.startswith('conv'):groups.append(self.spatial)
        return sum(p.numel() for g in groups for p in g.parameters())+extra

    def forward(self,x):
        h=self.local(x)
        if self.steps:
            if self.arm.startswith('graph'):
                base=h[:,self.permutation] if self.arm.endswith('scrambled') else h
                square=base.square().mean(-1)
                distances=(square[:,:,None]+square[:,None,:]-2*(base@base.transpose(1,2))/256).clamp_min(0)
                weights=self.adjacency*torch.exp(-distances/(F.softplus(self.temperature)+.001))*F.softplus(self.coupling)
                anchor=F.softplus(self.anchor(base))+.05
                residual=screened_diffusion(base,anchor,weights,self.steps)-base
                if self.arm.endswith('scrambled'):residual=residual[:,self.inverse_permutation]
            else:
                z=h.transpose(1,2).reshape(-1,256,8,8)
                for _ in range(self.steps):z=z+.25*F.silu(self.spatial(z))
                residual=z.flatten(2).transpose(1,2)-h
            h=h+.5*torch.tanh(self.relation(residual))
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weight=v[...,3].softmax(1);color=(v[...,:3]*weight[...,None]).sum(1)
        return color,v[...,:3],weight
