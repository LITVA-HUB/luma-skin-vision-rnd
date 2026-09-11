"""Actual observed patch-bag mixing within a fixed instrument-reference site."""
import numpy as np
import torch

ARMS=['baseline','self_bootstrap','soft_mode_control','paired_union','paired_stratified']


def make_plan(batch,patches,rng):
    return {'augment':rng.random(batch)<.5,'fraction':rng.random(batch),
        'first':rng.integers(0,patches,size=(batch,patches)),
        'second':rng.integers(0,patches,size=(batch,patches)),
        'coin':rng.random((batch,patches))}


def apply_plan(x,mode,partner,plan,arm):
    if arm not in ARMS:raise ValueError('Unknown observed-patch augmentation')
    b,n,c=x.shape;device=x.device
    natural=torch.arange(n,device=device).expand(b,-1)
    augment=torch.as_tensor(plan['augment'],device=device)
    if arm=='baseline':augment=torch.zeros_like(augment)
    origin=torch.as_tensor(plan['coin']<plan['fraction'][:,None],device=device)
    origin=origin & augment[:,None]
    if arm in ('baseline','self_bootstrap'):origin=torch.zeros_like(origin)
    fraction=origin.float().mean(1,keepdim=True)
    if arm=='soft_mode_control':origin=torch.zeros_like(origin)
    if arm in ('paired_stratified','soft_mode_control'):index=natural
    else:
        first=torch.as_tensor(plan['first'],device=device);second=torch.as_tensor(plan['second'],device=device)
        index=torch.where(origin,second,first)
        index=torch.where(augment[:,None],index,natural)
    source=torch.where(origin,partner[:,None],torch.arange(b,device=device)[:,None])
    out=x[source,index]
    target=mode*(1-fraction)+mode[partner]*fraction
    return out,target,origin,index
