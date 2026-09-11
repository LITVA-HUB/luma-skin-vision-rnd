"""Single-image skin models with TRAIN-only paired-capture regularization."""
import numpy as np
import torch
from torch import nn
from skin_mskcc_vote import PatchVotes

ARMS=['raw','standardized','quotient3','output','vicreg']


def pair_indices(sites,count,rng):
    groups=[np.flatnonzero(sites==s) for s in np.unique(sites)]
    a=[];b=[]
    for j in rng.integers(0,len(groups),size=count):
        group=groups[j];chosen=rng.choice(group,size=2,replace=len(group)==1)
        a.append(chosen[0]);b.append(chosen[1])
    return np.array(a),np.array(b)


def nuisance_transform(tokens,sites,remove):
    """Within-site covariance on mean token channels; never uses camera labels."""
    x=np.asarray(tokens,dtype=np.float64);center=x.mean((0,1));scale=x.std((0,1))
    scale=np.where(scale<1e-8,1.,scale);summary=((x-center)/scale).mean(1);blocks=[]
    for site in np.unique(sites):
        z=summary[sites==site]
        if len(z)>1:
            residual=z-z.mean(0);blocks.append(residual.T@residual/(len(z)-1))
    if not blocks:raise ValueError('No actual paired capture sites')
    covariance=np.mean(blocks,axis=0);values,vectors=np.linalg.eigh(covariance)
    q=np.eye(x.shape[-1])
    if remove:
        directions=vectors[:,-remove:];q-=directions@directions.T
    return center,scale,q,values


def vicreg_loss(a,b):
    similarity=(a-b).square().mean()
    variance=(torch.relu(1-torch.sqrt(a.var(0)+1e-4)).mean()+torch.relu(1-torch.sqrt(b.var(0)+1e-4)).mean())/2
    covariance=0.
    for z in [a,b]:
        z=z-z.mean(0);c=z.T@z/(len(z)-1)
        covariance+=(c.square().sum()-c.diagonal().square().sum())/z.shape[1]
    return 25*similarity+25*variance+covariance


class PairedColor(nn.Module):
    def __init__(self,target_std,arm):
        super().__init__();self.backbone=PatchVotes(target_std,0)
        self.projection=nn.Linear(512,64) if arm=='vicreg' else None

    def forward(self,x,features=False):
        y,_,context=self.backbone(x,details=True)
        return (y,context) if features else y
