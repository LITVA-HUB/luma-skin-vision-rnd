"""ChromaSeed-Seg1: spatial facial-skin masks, separate from instrument color regression."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),
                         nn.GroupNorm(math.gcd(8,cout),cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),
                         nn.GroupNorm(math.gcd(8,cout),cout),nn.SiLU())


class SkinUNet(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.width = width
        widths = [width*2**i for i in range(5)]
        self.encoder = nn.ModuleList([ConvBlock(3,widths[0])] +
                                     [ConvBlock(a,b) for a,b in zip(widths[:-1],widths[1:],strict=True)])
        self.decoder = nn.ModuleList([ConvBlock(widths[i]+widths[i-1],widths[i-1]) for i in range(4,0,-1)])
        self.output = nn.Conv2d(width,1,1)

    def forward(self,x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] % 16 or x.shape[3] % 16:
            raise ValueError('RGB spatial dimensions must be a multiple of 16')
        skips = []
        for i,block in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x,2)
            x = block(x)
            skips.append(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1]),strict=True):
            x = F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x = block(torch.cat((x,skip),dim=1))
        return self.output(x)


def binary_mask(labels):
    labels = np.asarray(labels)
    if not np.isin(labels,np.arange(11)).all():
        raise ValueError('invalid LaPa labels')
    return ((labels == 1) | (labels == 6)).astype(np.uint8)


def skin_loss(logits,target):
    logits = logits.float()
    probability = torch.sigmoid(logits)
    axes = (1,2,3)
    dice = (2*(probability*target).sum(axes)+1)/(probability.sum(axes)+target.sum(axes)+1)
    return F.binary_cross_entropy_with_logits(logits,target)+(1-dice).mean()


def scores(tp,fp,fn,tn):
    tp,fp,fn,tn = map(int,(tp,fp,fn,tn))
    if min(tp,fp,fn,tn) < 0:
        raise ValueError('negative confusion count')
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else 1.,
                recall=tp/(tp+fn) if tp+fn else 1.)


def augment(x,y):
    flip = torch.rand((len(x),1,1,1),device=x.device) < .5
    x,y = torch.where(flip,x.flip(-1),x),torch.where(flip,y.flip(-1),y)
    gain = .9+.2*torch.rand((len(x),3,1,1),device=x.device)
    exposure = .85+.3*torch.rand((len(x),1,1,1),device=x.device)
    gamma = .85+.3*torch.rand((len(x),1,1,1),device=x.device)
    return ((x*gain*exposure).clamp(0,1)**gamma).clamp(0,1),y
