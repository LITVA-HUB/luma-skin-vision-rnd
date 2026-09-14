"""Unintegrated fused-AdamW prototype; CPU feasibility is not GPU acceptance."""

from __future__ import annotations

import torch
from torch.optim.adamw import adamw


class FusedBankAdamW:
    """Preserve one flat parameter bank while updating independent row views.

    No model, sampler or backward computation changes. Floating-point update
    ordering differs from BankAdamW, so exact old-trajectory identity is unproven.
    """

    def __init__(self, theta, lrs, weight_decay=0.01, max_norm=5.0):
        if theta.ndim != 2 or not theta.is_contiguous() or theta.dtype != torch.float32:
            raise ValueError("contiguous FP32 slots-by-parameters bank required")
        rates = torch.as_tensor(lrs, dtype=theta.dtype, device=theta.device).clone()
        if (
            rates.shape != (theta.shape[0],)
            or not torch.isfinite(rates).all()
            or torch.any(rates <= 0)
        ):
            raise ValueError("one positive finite rate per slot required")
        if weight_decay < 0 or max_norm <= 0:
            raise ValueError("invalid decay or clipping norm")
        self.theta, self.lrs = theta, rates
        self.initial_lrs = rates.clone()
        self.m, self.v = torch.zeros_like(theta), torch.zeros_like(theta)
        self.steps = torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)
        self.weight_decay, self.max_norm = weight_decay, max_norm

    @torch.no_grad()
    def reset(self, initial):
        if (
            initial.shape != self.theta.shape
            or initial.dtype != self.theta.dtype
            or initial.device != self.theta.device
        ):
            raise ValueError("matching initial parameter snapshot required")
        self.theta.copy_(initial)
        self.m.zero_()
        self.v.zero_()
        self.steps.zero_()
        self.lrs.copy_(self.initial_lrs)

    @torch.no_grad()
    def step(self):
        gradient = self.theta.grad
        if gradient is None or gradient.shape != self.theta.shape:
            raise ValueError("complete flat gradient bank required")
        norm = gradient.square().sum(1).sqrt()
        coefficient = (self.max_norm / (norm + 1e-6)).clamp(max=1.0)
        clipped = gradient * coefficient[:, None]
        for slot in range(self.theta.shape[0]):
            rate = self.lrs[slot] if self.theta.is_cuda else float(self.lrs[slot])
            adamw(
                [self.theta[slot]],
                [clipped[slot]],
                [self.m[slot]],
                [self.v[slot]],
                [],
                [self.steps[slot]],
                foreach=False,
                fused=True,
                capturable=self.theta.is_cuda,
                differentiable=False,
                amsgrad=False,
                beta1=0.9,
                beta2=0.999,
                lr=rate,
                weight_decay=self.weight_decay,
                eps=1e-8,
                maximize=False,
            )
