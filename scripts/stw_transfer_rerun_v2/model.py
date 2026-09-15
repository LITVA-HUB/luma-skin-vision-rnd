"""Fixed single architecture for STW_TRANSFER_RERUN_V2. No network calls."""
import math
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


class Encoder(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        if pretrained:
            backbone.load_state_dict(torch.load(pretrained, map_location='cpu', weights_only=True), strict=True)
        self.stem = backbone.features[:9]
        self.tail = backbone.features[9:]
        self.projection = nn.Sequential(nn.Linear(1152, 128), nn.ReLU(), nn.LayerNorm(128))
        self.stem.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.stem.eval()
        for module in self.tail.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, early, mask):
        feature = self.tail(early)
        mask = nn.functional.adaptive_avg_pool2d(mask, feature.shape[-2:])
        mean = feature.mean((-2, -1))
        den = mask.sum((-2, -1))
        skin = (feature * mask).sum((-2, -1)) / den.clamp_min(1e-6)
        skin = torch.where(den > 0, skin, mean)
        return self.projection(torch.cat([mean, skin], 1))


class Model(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.score = nn.Linear(128, 1)
        self.origin = nn.Parameter(torch.tensor(-2.))
        self.gaps = nn.Parameter(torch.full((8,), math.log(math.expm1(.5))))

    def forward(self, early, mask):
        embedding = self.encoder(early, mask)
        thresholds = torch.cat([self.origin.view(1), self.origin + nn.functional.softplus(self.gaps).cumsum(0)])
        return self.score(embedding) - thresholds[None], embedding


def lab_head():
    return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
