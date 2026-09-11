"""Capacity-matched residual and differentiable local-reference color adapters."""
import torch
from torch import nn
from skin_capture_model import CaptureColor

ARMS=('residual','mean','affine')


def core_features(core,tokens):
    h=core.local(tokens)
    context=core.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
    v=core.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
    weights=v[...,12].softmax(1)
    hypotheses=(v[...,:12].reshape(len(tokens),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
    return (hypotheses*core.gate(context).softmax(1)[...,None]).sum(1),context


def reference_correction(query,bank,residual,allowed,kind):
    if kind not in ('mean','affine'):raise ValueError('Unknown reference solver')
    if allowed.shape!=(len(query),len(bank)) or not bool(allowed.any(1).all()):
        raise ValueError('Every query needs nonempty allowed support')
    if len(bank)!=len(residual):raise ValueError('Bank mismatch')
    q,b,y=query.double(),bank.double(),residual.double()
    logs=-.5*(q[:,None]-b[None]).square().mean(-1)
    weights=logs.masked_fill(~allowed,float('-inf')).softmax(1)
    ym=weights@y
    if kind=='mean':return ym.to(query.dtype)
    xm=weights@b;xc=b[None]-xm[:,None];yc=y[None]-ym[:,None]
    gram=xc.transpose(1,2)@(weights[:,:,None]*xc)
    rhs=xc.transpose(1,2)@(weights[:,:,None]*yc)
    identity=torch.eye(b.shape[1],dtype=b.dtype,device=b.device)
    coefficient=torch.linalg.solve(gram+.01*identity,rhs)
    result=ym+((q-xm)[:,None]@coefficient).squeeze(1)
    return result.to(query.dtype)


class ColorAdapter(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown color adapter')
        self.arm=arm
        self.features=nn.Sequential(nn.Linear(551,256),nn.SiLU(),nn.Linear(256,192),nn.SiLU(),nn.Linear(192,16),nn.Tanh())
        self.output=nn.Linear(16,3)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def from_embedding(self,z,bank_z,bank_residual,allowed):
        gate=self.output(z).tanh()
        if self.arm=='residual':return gate
        return gate*reference_correction(z,bank_z,bank_residual,allowed,self.arm)

    def forward(self,x,bank,bank_residual,allowed):
        z=self.features(x)
        bank_z=None if self.arm=='residual' else self.features(bank)
        return self.from_embedding(z,bank_z,bank_residual,allowed)


class DeployedColor(nn.Module):
    """Prepared features in; native Lab out. Only fixed training memory is stored."""
    def __init__(self,arm,bank_size):
        super().__init__();self.core=CaptureColor('mixture');self.adapter=ColorAdapter(arm)
        for name,size in [('feature_mean',548),('feature_std',548),('target_mean',3),('target_std',3)]:
            self.register_buffer(name,torch.zeros(size))
        self.register_buffer('bank_z',torch.zeros(bank_size,16))
        self.register_buffer('bank_residual',torch.zeros(bank_size,3))

    def forward(self,tokens,color):
        p,c=core_features(self.core,tokens)
        x=torch.cat([(torch.cat([c,color],1)-self.feature_mean)/self.feature_std,p],1)
        z=self.adapter.features(x)
        allowed=torch.ones(len(x),len(self.bank_z),dtype=torch.bool,device=x.device)
        correction=self.adapter.from_embedding(z,self.bank_z,self.bank_residual,allowed)
        return (p+correction)*self.target_std+self.target_mean
