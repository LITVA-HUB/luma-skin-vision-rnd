"""Matched image color heads with a fixed measured-material decoder."""
import torch
from torch import nn
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.skin_capture_model import CaptureColor

ARMS=['direct','tangent','material','tangent_residual','material_residual']


class MaterialImage(CaptureColor):
    def __init__(self,arm,prior,target_mean,target_std):
        super().__init__('mixture')
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        old=self.votes[-1];self.votes[-1]=nn.Linear(256,45)
        with torch.no_grad():
            self.votes[-1].weight[32:].copy_(old.weight)
            self.votes[-1].bias[32:].copy_(old.bias)
        for name in ('mu','basis','matrix','white','base','jacobian'):
            self.register_buffer(name,torch.as_tensor(prior[name],dtype=torch.float32).clone())
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32).clone())
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32).clone())

    def material_decode(self,z):
        r=torch.sigmoid(z@self.basis.T+self.mu)
        q=(r@self.matrix)/self.white;d=6/29
        f=torch.where(q>d**3,q.clamp_min(1e-12).pow(1/3),q/(3*d*d)+4/29)
        return torch.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],-1)

    def active_parameters(self):
        n=sum(p.numel() for p in self.parameters())
        if self.arm=='direct':return n-32*257
        if not self.arm.endswith('_residual'):return n-12*257
        return n

    def forward(self,x):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        w=v[...,44].softmax(1)
        z=(v[...,:32].reshape(len(x),h.shape[1],4,8)*w[:,:,None,None]).sum(1)
        free=(v[...,32:44].reshape(len(x),h.shape[1],4,3)*w[:,:,None,None]).sum(1)
        if self.arm=='direct':hypotheses=free
        else:
            native=self.base+z@self.jacobian if self.arm.startswith('tangent') else self.material_decode(z)
            hypotheses=(native-self.target_mean)/self.target_std
            if self.arm.endswith('_residual'):hypotheses=hypotheses+free
        logits=self.gate(context);gate=logits.softmax(1)
        prediction=(hypotheses*gate[...,None]).sum(1)
        return prediction,logits,hypotheses,free
