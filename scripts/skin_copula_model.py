"""Absolute-color patch core plus matched marginal/dependence histogram context."""
import torch
from torch import nn

ARMS=['none','rgb_hist','copula','rank_only']


class DistributionColor(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(896,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        self.distribution=nn.Sequential(nn.Linear(512,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())

    def forward(self,x):
        h=self.local(x[:,:1152].reshape(-1,64,18))
        distribution=self.distribution(x[:,1152:].clamp_min(0).sqrt())
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0),distribution],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weights=v[...,3].softmax(1);color=(v[...,:3]*weights[...,None]).sum(1)
        return color,v[...,:3],weights

    def active_parameters(self):return sum(p.numel() for p in self.parameters())
