"""Candidate skin-color loss fields using real training-reference dictionaries."""
import numpy as np
import torch
from torch import nn
from skin_capture_model import CaptureColor

ARMS=['direct','soft_ce','risk_simplex','risk_affine']


def risk_embedding(cost):
    """Full Gram factor; preserves uniform candidate-risk MSE, not a PCA truncation."""
    cost=np.asarray(cost,dtype=np.float64)
    center=cost.mean(0);c=cost-center
    scale=float(np.sqrt(np.mean(c*c)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Degenerate risk dictionary')
    gram=c@c.T/c.shape[1]/scale**2
    values,vectors=np.linalg.eigh(gram)
    if values.min() < -1e-10:raise ValueError('Invalid Gram matrix')
    phi=vectors*np.sqrt(values.clip(0))[None]
    return phi,center,scale


def affine_weights(logits):
    return (1+logits-logits.mean(1,keepdim=True))/logits.shape[1]


class LossFieldImage(CaptureColor):
    def __init__(self,atoms):
        super().__init__('mixture')
        self.atom_head=nn.Linear(256,atoms)

    def forward(self,x):
        h=self.local(x);context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        hidden=self.votes[1](self.votes[0](torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1)))
        v=self.votes[2](hidden);weights=v[...,12].softmax(1)
        hypotheses=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        gate=self.gate(context);point=(hypotheses*gate.softmax(1)[...,None]).sum(1)
        atoms=self.atom_head((hidden*weights[...,None]).sum(1))
        return point,gate,hypotheses,atoms
