"""Stable-color stacked correction with subject-excluded encoder controls."""
import numpy as np
import torch
from torch import nn
from skin_support_curve import SkinRepresentation

ARMS=('in_full','in_matched','out_person')


def inner_folds(person,camera):
    if len(person)!=len(camera):raise ValueError('Role length mismatch')
    result=np.full(len(person),-1,np.int64);rng=np.random.default_rng(917031)
    for device,n in (('SLR',6),('ipod',12)):
        people=np.unique(person[camera==device])
        if len(people)!=n:raise ValueError('Expected18 support people in6/12 camera groups')
        if any(len(np.unique(camera[person==p]))!=1 for p in people):raise ValueError('Ambiguous person camera')
        for fold,ids in enumerate(np.split(rng.permutation(people),3)):result[np.isin(person,ids)]=fold
    if np.any(result<0):raise ValueError('Unsupported camera')
    return result


def choose_predictions(predictions,fold):
    if predictions.shape!=(3,len(fold),3) or not np.isin(fold,[0,1,2]).all():raise ValueError('Invalid fold prediction table')
    rows=np.arange(len(fold))
    return predictions[fold,rows].copy(),predictions[(fold+1)%3,rows].copy()


def features(color,native_prediction,color_mean,color_std,target_mean,target_std):
    return np.column_stack([(color-color_mean)/color_std,(native_prediction-target_mean)/target_std]).astype(np.float32)


class StableHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(39,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),nn.Linear(384,64),nn.SiLU(),nn.Linear(64,3),nn.Tanh())
        nn.init.zeros_(self.net[-2].weight);nn.init.zeros_(self.net[-2].bias)

    def forward(self,x):return self.net(x)


class StableColorModel(nn.Module):
    """One prepared image, one core and one correction head; no reference bank."""
    def __init__(self):
        super().__init__();self.base=SkinRepresentation('baseline');self.head=StableHead()
        for name,size in [('color_mean',36),('color_std',36),('target_mean',3),('target_std',3)]:
            self.register_buffer(name,torch.zeros(size))

    def forward(self,tokens,color):
        p=self.base(tokens,None)[0]
        native=p*self.target_std+self.target_mean
        x=torch.cat([(color-self.color_mean)/self.color_std,(native-self.target_mean)/self.target_std],1)
        return native+self.head(x)*self.target_std
