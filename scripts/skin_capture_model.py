"""Matched latent-capture color hypotheses and instrument perceptual loss."""
import torch
from torch import nn

MODES=['NP-C','NP-NC','P-C','P-NC']
ARCHES=['plain','uniform','mixture']


def _root(x):
    # Finite zero subgradient convention; nonzero branch retains the formula.
    return torch.where(x>0,torch.sqrt(x.clamp_min(1e-24)),torch.zeros_like(x))


def _hue(a,b):
    neutral=(a==0)&(b==0)
    return torch.rad2deg(torch.atan2(torch.where(neutral,torch.zeros_like(b),b),
                                   torch.where(neutral,torch.ones_like(a),a))).remainder(360)


def delta_e00_squared(lab1,lab2):
    """CIEDE2000 squared, kL=kC=kH=1, computed in double for hue boundaries.

    Autograd is piecewise valid; this does not make the original formula globally
    smooth. At exact zero chroma/root arguments use a finite zero convention.
    """
    x,y=torch.broadcast_tensors(lab1.double(),lab2.double())
    l1,a1,b1=x.unbind(-1);l2,a2,b2=y.unbind(-1)
    c1=_root(a1*a1+b1*b1);c2=_root(a2*a2+b2*b2);cm=(c1+c2)/2
    g=.5*(1-_root(cm**7/(cm**7+25**7)))
    ap1=(1+g)*a1;ap2=(1+g)*a2
    cp1=_root(ap1*ap1+b1*b1);cp2=_root(ap2*ap2+b2*b2)
    hp1=_hue(ap1,b1);hp2=_hue(ap2,b2);zero=cp1*cp2==0
    dh=hp2-hp1;dh=torch.where(zero,torch.zeros_like(dh),torch.where(dh>180,dh-360,torch.where(dh< -180,dh+360,dh)))
    dl=l2-l1;dc=cp2-cp1;dhp=2*_root(cp1*cp2)*torch.sin(torch.deg2rad(dh/2))
    lm=(l1+l2)/2;cpm=(cp1+cp2)/2;hs=hp1+hp2
    hm=torch.where(zero,hs,torch.where(abs(hp1-hp2)<=180,hs/2,torch.where(hs<360,(hs+360)/2,(hs-360)/2)))
    t=1-.17*torch.cos(torch.deg2rad(hm-30))+.24*torch.cos(torch.deg2rad(2*hm))+.32*torch.cos(torch.deg2rad(3*hm+6))-.20*torch.cos(torch.deg2rad(4*hm-63))
    sl=1+.015*(lm-50)**2/torch.sqrt(20+(lm-50)**2);sc=1+.045*cpm;sh=1+.015*cpm*t
    rt=-2*_root(cpm**7/(cpm**7+25**7))*torch.sin(torch.deg2rad(60*torch.exp(-((hm-275)/25)**2)))
    return ((dl/sl)**2+(dc/sc)**2+(dhp/sh)**2+rt*(dc/sc)*(dhp/sh)).clamp_min(0)


class CaptureColor(nn.Module):
    """All arms share exact parameter shapes; camera/mode is never an input."""
    def __init__(self,arch):
        super().__init__()
        if arch not in ARCHES:raise ValueError('Unknown architecture')
        self.arch=arch
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,13))
        self.gate=nn.Linear(512,4)

    def forward(self,x):
        h=self.local(x);context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        weights=v[...,12].softmax(1)
        hypotheses=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        logits=self.gate(context);gate=logits.softmax(1)
        coefficient=gate if self.arch=='mixture' else torch.full_like(gate,.25)
        prediction=(hypotheses*coefficient[...,None]).sum(1)
        return prediction,logits,hypotheses
