"""Remove mixture decomposition or explicitly anchor real native-Lab experts."""
import torch
from torch import nn
from skin_capture_model import CaptureColor

MECHANISMS=('baseline','plain','uniform_anchor','conditional')
AUGMENTATIONS=('raw','paired')
ARMS=tuple(m+'_'+a for m in MECHANISMS for a in AUGMENTATIONS)


class PlainColor(nn.Module):
    """Actual single color head; initial function equals the uniform old model."""
    def __init__(self,base):
        super().__init__()
        self.local=base.local;self.context=base.context
        last=nn.Linear(256,4)
        with torch.no_grad():
            last.weight[:3].copy_(base.votes[-1].weight[:12].reshape(4,3,256).mean(0))
            last.bias[:3].copy_(base.votes[-1].bias[:12].reshape(4,3).mean(0))
            last.weight[3].copy_(base.votes[-1].weight[12]);last.bias[3].copy_(base.votes[-1].bias[12])
        self.votes=nn.Sequential(base.votes[0],base.votes[1],last)

    def forward(self,x):
        h=self.local(x)
        c=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,c[:,None].expand(-1,h.shape[1],-1)],-1))
        p=(v[...,:3]*v[...,3].softmax(1)[...,None]).sum(1)
        return p,p.new_zeros((len(x),4)),p[:,None]


def make_model(mechanism):
    if mechanism not in MECHANISMS:raise ValueError('Unknown mechanism')
    base=CaptureColor('mixture')
    return PlainColor(base) if mechanism=='plain' else base


def objective(mechanism,p,g,h,y,q):
    color=(p-y).square().mean()
    if mechanism=='plain':return color
    mode=-(q*g.log_softmax(1)).sum(1).mean()
    if mechanism=='baseline':return color+.1*mode
    weights=q if mechanism=='conditional' else torch.full_like(q,.25)
    if mechanism not in ('conditional','uniform_anchor'):raise ValueError('Unknown mechanism')
    anchor=((h-y[:,None]).square().mean(2)*weights).sum(1).mean()
    return .5*color+.5*anchor+.1*mode
