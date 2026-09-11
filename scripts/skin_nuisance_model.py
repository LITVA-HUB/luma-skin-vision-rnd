"""Causal controls for the training-only latent branch; no new inference block."""
import torch
from skin_spatial_model import SpatialColor,screened_diffusion

ARMS=['learned','fixed_grid','global','bias']


def control_residual(h,kind,adjacency):
    if kind=='bias':return torch.full_like(h,.05)
    anchor=torch.full_like(h[...,:1],.05+.6931471805599453)
    if kind=='fixed_grid':weights=adjacency*.6931471805599453
    elif kind=='global':
        weights=(torch.ones_like(adjacency)-torch.eye(64,device=h.device,dtype=h.dtype))*(3.5*.6931471805599453/63)
    else:raise ValueError(kind)
    return screened_diffusion(h,anchor,weights,3)-h


class NuisanceColor(SpatialColor):
    def __init__(self,control):
        if control not in ARMS:raise ValueError(control)
        super().__init__('graph3');self.control=control

    def forward(self,x):
        if not self.training:
            previous=self.steps;self.steps=0
            try:return super().forward(x)
            finally:self.steps=previous
        if self.control=='learned':return super().forward(x)
        h=self.local(x);h=h+.5*torch.tanh(self.relation(control_residual(h,self.control,self.adjacency)))
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weight=v[...,3].softmax(1);color=(v[...,:3]*weight[...,None]).sum(1)
        return color,v[...,:3],weight

    def active_parameters(self):
        return sum(p.numel() for g in [self.local,self.context,self.votes] for p in g.parameters())

    def training_active_parameters(self):
        return self.active_parameters()+self.relation.weight.numel()+(259 if self.control=='learned' else 0)
