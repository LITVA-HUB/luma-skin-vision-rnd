"""Standard MobileNetV3-small regressor and matched hypothesis-mixture variant."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


class CompactCC(nn.Module):
    def __init__(self, mixture=False):
        super().__init__()
        self.mixture = mixture
        self.backbone = mobilenet_v3_small(weights=None).features
        self.context = nn.Sequential(nn.Linear(576, 64), nn.SiLU())
        self.illuminant = nn.Linear(64, 3)
        # Same parameter budget in baseline; unused mixture branch excluded in forward.
        self.weights = nn.Sequential(nn.Linear(64 + 12, 32), nn.SiLU(), nn.Linear(32, 5))

    def forward(self, image, experts):
        pooled = self.backbone(image).mean(dim=(-1, -2))
        context = self.context(pooled)
        direct = F.normalize(F.softplus(self.illuminant(context)) + 1e-5, dim=-1)
        logits = self.weights(torch.cat([context, experts.flatten(1)], dim=1))
        if self.mixture:
            candidates = torch.cat([experts, direct[:, None]], dim=1)
            pred = F.normalize((logits.softmax(-1)[..., None] * candidates).sum(1), dim=-1)
        else:
            pred = direct
        return pred, context


def reproduction_loss(pred, gt):
    ratio = gt / pred.clamp_min(1e-6)
    cosine = F.normalize(ratio, dim=-1).sum(-1) / (3**0.5)
    # Stable angle training clamp is explicit; evaluation uses unclipped valid cosine.
    return torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7)).mean() * (180 / torch.pi)
