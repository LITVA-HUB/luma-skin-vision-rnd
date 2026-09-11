"""Spatial branch used only during fitting; deployment always uses the plain core."""
import torch
from skin_spatial_model import SpatialColor

ARMS=['graph_always','conv_always','graph_drop','conv_drop']


class TrainingBranchColor(SpatialColor):
    def __init__(self,strategy):
        if strategy not in ARMS:raise ValueError(strategy)
        super().__init__(strategy.split('_')[0]+'3')
        self.strategy=strategy

    def forward(self,x):
        previous=self.steps
        use=self.training
        if use and self.strategy.endswith('_drop'):
            use=bool(torch.rand(())>=.5)
        self.steps=3 if use else 0
        try:return super().forward(x)
        finally:self.steps=previous

    def active_parameters(self):
        return sum(p.numel() for g in [self.local,self.context,self.votes] for p in g.parameters())

    def training_active_parameters(self):return super().active_parameters()
