"""Compact student and physically limited sensor-response augmentation."""
import torch
from torch import nn
from torch.nn import functional as F

from luma_skin_vision.cc.v2 import EPS, CompactResidualCC, input_validity, validate_image


def sensor_matrix(n,*,generator,device="cpu",dtype=torch.float32,identity_fraction=.25,mixing=.35):
    if n<1 or not 0<=identity_fraction<=1 or not 0<=mixing<.5:
        raise ValueError("Invalid virtual sensor configuration")
    kwargs={"generator":generator,"device":device,"dtype":dtype}
    a=torch.rand(n,3,3,**kwargs).clamp_min(torch.finfo(dtype).tiny)
    a=a/a.sum(-1,keepdim=True)
    epsilon=torch.rand(n,1,1,**kwargs)*mixing
    identity=torch.eye(3,device=device,dtype=dtype)[None].expand(n,-1,-1)
    gain=((torch.rand(n,3,**kwargs)*2-1)*.7).exp()
    matrix=gain[:,:,None]*((1-epsilon)*identity+epsilon*a)
    keep=torch.rand(n,**kwargs)<identity_fraction
    return torch.where(keep[:,None,None],identity,matrix)


def sensor_transform(image,gt,matrix):
    if image.ndim!=4 or image.shape[1]!=3 or gt.shape!=(len(image),3) or matrix.shape!=(len(image),3,3):
        raise ValueError("Unaligned image/GT/sensor arrays")
    transformed=torch.einsum("nij,njhw->nihw",matrix,image)
    label=F.normalize(torch.einsum("nij,nj->ni",matrix,gt),dim=-1)
    return transformed,label


def teacher_render(image,gt=None):
    validate_image(image)
    rgb=image.float()
    if gt is not None:
        if gt.shape!=(len(image),3) or not (torch.isfinite(gt)&(gt>0)).all():
            raise ValueError("Positive aligned training GT required")
        rgb=rgb/gt.float()[:,:,None,None]
    scale=torch.quantile(rgb.flatten(1),.95,dim=1).clamp_min(1e-8)
    rgb=(rgb/scale[:,None,None,None]).clamp(0,1).pow(1/2.2)
    rgb=F.interpolate(rgb,size=(224,224),mode="bilinear",align_corners=False)
    mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return (rgb-mean)/std


class SemanticColorNet(CompactResidualCC):
    def __init__(self):
        super().__init__("direct","large")
        self.teacher_projection=nn.Conv2d(960,384,1)

    def forward(self,image):
        validate_image(image)
        rms=image.square().mean((1,2,3),keepdim=True).clamp_min(EPS**2).sqrt()
        features=self.backbone(image/rms)
        context=self.context(features.mean((-2,-1)))
        pred=F.normalize(self.illuminant(context).clamp(-2,2).exp(),dim=-1)
        pooled=features if features.shape[-2:]==(4,4) else F.adaptive_avg_pool2d(features,(4,4))
        projected=self.teacher_projection(pooled).flatten(2).transpose(1,2)
        return {"pred":pred,"context":context,"teacher_features":F.normalize(projected,dim=-1),"valid":input_validity(image)}
