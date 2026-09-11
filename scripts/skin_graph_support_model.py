"""Graph receives original images only; shared core also learns observed bags."""
from skin_spatial_model import SpatialColor

ARMS=('plain_raw','plain_paired','graph_raw','graph_paired')


class GraphSupportColor(SpatialColor):
    def __init__(self):
        super().__init__('graph3')

    def forward(self,x,branch=False):
        previous=self.steps
        self.steps=3 if self.training and branch else 0
        try:return super().forward(x)
        finally:self.steps=previous

    def inference_parameters(self):
        return sum(p.numel() for group in (self.local,self.context,self.votes) for p in group.parameters())
