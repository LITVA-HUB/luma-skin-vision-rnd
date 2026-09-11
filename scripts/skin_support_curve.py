"""TRAIN-person support and learned patch-information falsifier."""
import numpy as np
import torch
from torch import nn
from skin_capture_model import CaptureColor

ARMS=('baseline','statistics','pixels')


def patient_roles(patient,device,count,seed):
    if count not in (6,12,18):raise ValueError('Frozen person counts only')
    fixed=np.random.default_rng(20260911);draw=np.random.default_rng(seed)
    selected=[];held=[]
    for camera,hold_n in [('SLR',2),('ipod',4)]:
        ids=np.unique(patient[device==camera]);expected=8 if camera=='SLR' else 16
        if len(ids)!=expected:raise ValueError('Expected original TRAIN population')
        if any(len(np.unique(device[patient==p]))!=1 for p in ids):raise ValueError('Ambiguous person camera')
        ids=fixed.permutation(ids);held.extend(ids[:hold_n])
        available=draw.permutation(ids[hold_n:]);selected.extend(available[:count*hold_n//6])
    a=np.isin(patient,selected);b=np.isin(patient,held)
    if np.any(a&b):raise ValueError('Person leakage')
    return a,b


def pixel_patches(rgb):
    if rgb.ndim!=4 or rgb.shape[1:]!=(128,128,3):raise ValueError('Expected NHWC128RGB')
    return rgb.reshape(-1,8,16,8,16,3).permute(0,1,3,5,2,4).reshape(-1,3,16,16)


class SkinRepresentation(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown representation')
        self.arm=arm;self.core=CaptureColor('mixture')
        if arm=='statistics':
            self.adapter=nn.Sequential(nn.Linear(18,64),nn.SiLU(),nn.Linear(64,120),nn.SiLU(),nn.Linear(120,18))
        elif arm=='pixels':
            self.adapter=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.SiLU(),nn.Conv2d(16,24,3,2,1),nn.SiLU(),
                nn.Conv2d(24,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,18))
        if arm!='baseline':nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)

    def forward(self,tokens,rgb):
        if self.arm=='statistics':tokens=tokens+self.adapter(tokens)
        elif self.arm=='pixels':tokens=tokens+self.adapter(pixel_patches(rgb)).reshape(len(tokens),64,18)
        return self.core(tokens)
