"""Identical RGB mixture heads with aligned/global/shuffled frozen descriptors."""
import torch
from torch import nn
from skin_capture_model import CaptureColor

ARMS = ['plain', 'aligned', 'global', 'shuffled']


class LocalTeacherColor(CaptureColor):
    def __init__(self, arm):
        if arm not in ARMS: raise ValueError('Unknown local teacher arm')
        super().__init__('mixture'); self.arm = arm
        self.adapter = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 256))
        self.register_buffer('teacher_mean', torch.zeros(384))
        self.register_buffer('teacher_std', torch.ones(384))

    def teacher_tokens(self, x):
        values = x[..., 18:]
        if self.arm == 'global': values = values.mean(1, keepdim=True).expand(-1, 64, -1)
        return (values-self.teacher_mean)/self.teacher_std

    def active_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total-sum(p.numel() for p in self.adapter.parameters()) if self.arm == 'plain' else total

    def forward(self, x):
        h = self.local(x[..., :18].contiguous())
        if self.arm != 'plain': h = h+.5*torch.tanh(self.adapter(self.teacher_tokens(x)))
        context = self.context(torch.cat([h.mean(1), h.amax(1), h.std(1, correction=0)], 1))
        v = self.votes(torch.cat([h, context[:, None].expand(-1, h.shape[1], -1)], -1))
        weights = v[..., 12].softmax(1)
        hypotheses = (v[..., :12].reshape(len(x), h.shape[1], 4, 3)*weights[:, :, None, None]).sum(1)
        logits = self.gate(context); gate = logits.softmax(1)
        return (hypotheses*gate[..., None]).sum(1), logits, hypotheses
