"""Compact native-Lab density and deterministic downstream color decisions.

Original local implementation of established mixture-density/Bayes principles.
Quadrature and finite candidate decisions are approximations, not error bounds.
"""
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skin_capture_model import CaptureColor
from luma_skin_vision.color import delta_e00

ARMS=['mse_mode','mse','gaussian','mdn4']


class ColorDistribution(CaptureColor):
    def __init__(self):
        super().__init__('mixture')
        self.scale_head=nn.Linear(512,12)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias,math.log(math.expm1(.95)))

    def forward(self,x):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        weights=v[...,12].softmax(1)
        means=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        logits=self.gate(context)
        point=(means*logits.softmax(1)[...,None]).sum(1)
        scales=.05+F.softplus(self.scale_head(context).reshape(len(x),4,3))
        return point,logits,means,scales


def density_nll(y,means,scales,logits):
    log_density=-.5*(((y[:,None]-means)/scales)**2+2*scales.log()+math.log(2*math.pi)).sum(-1)
    return -torch.logsumexp(logits.log_softmax(1)+log_density,1)


def quadrature(means,scales,probabilities,order=3):
    """Tensor Gauss-Hermite rule for normal components; deterministic FP64."""
    means=np.asarray(means,dtype=np.float64);scales=np.asarray(scales,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64)
    if order not in (2,3):raise ValueError('Frozen quadrature orders are 2 or 3')
    if means.shape!=scales.shape or means.shape[:2]!=p.shape or means.shape[-1]!=3:raise ValueError('Invalid density shapes')
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales<0):raise ValueError('Invalid color density')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(1)<=0):raise ValueError('Invalid probabilities')
    p=p/p.sum(1,keepdims=True)
    node,weight=np.polynomial.hermite.hermgauss(order)
    triples=np.array(list(itertools.product(range(order),repeat=3)))
    offsets=np.sqrt(2)*node[triples]
    weights=np.prod(weight[triples]/np.sqrt(np.pi),axis=1)
    points=means[:,:,None,:]+scales[:,:,None,:]*offsets[None,None]
    return points.reshape(len(means),-1,3),(p[:,:,None]*weights[None,None]).reshape(len(means),-1)


def color_decision(means,scales,probabilities,order=3):
    """Choose among mean, component centers and +/-0.25,0.5 marginal SD offsets.

    No reference target is accepted. Expected errors integrate the predicted
    distribution only, and require independent calibration before deployment.
    """
    means=np.asarray(means,dtype=np.float64);scales=np.asarray(scales,dtype=np.float64)
    p=np.asarray(probabilities,dtype=np.float64);p=p/p.sum(1,keepdims=True)
    center=(means*p[...,None]).sum(1)
    marginal=np.sqrt(((scales**2+(means-center[:,None])**2)*p[...,None]).sum(1))
    offsets=np.concatenate([np.eye(3)*a for a in (.25,-.25,.5,-.5)])
    candidates=np.concatenate([center[:,None],means,center[:,None]+marginal[:,None]*offsets[None]],1)
    nodes,w=quadrature(means,scales,p,order)
    losses=delta_e00(candidates[:,:,None,:],nodes[:,None,:,:])
    risks=(losses*w[:,None]).sum(-1)
    index=risks.argmin(1)
    return {'prediction':candidates[np.arange(len(center)),index],
        'mean':center,'expected_error':risks[np.arange(len(center)),index],
        'mean_expected_error':risks[:,0],'decision_index':index}
