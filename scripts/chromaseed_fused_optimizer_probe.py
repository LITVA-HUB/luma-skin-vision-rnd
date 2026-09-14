"""Small CPU synthetic feasibility probes; no CUDA context and no production data."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from chromaseed_architecture_scale import Bank
from chromaseed_fused_optimizer import FusedBankAdamW
from chromaseed_refine import BankAdamW
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/benchmarks/chromaseed_fused_optimizer_cpu_probe"


def stream_probe(steps=512):
    rng = np.random.default_rng(91823)
    initial = rng.normal(size=(6, 101)).astype(np.float32)
    old = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    new = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    rates = torch.tensor([1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.1])
    reference = BankAdamW([old], rates.numpy().copy())
    candidate = FusedBankAdamW(new, rates.numpy().copy())
    snapshots = []
    max_weights = max_m = max_v = 0.0
    for step in range(1, steps + 1):
        gradient = rng.normal(size=initial.shape).astype(np.float32)
        gradient[0] = 0
        gradient[1] *= 100
        gradient[2] *= 1e-6
        g = torch.from_numpy(gradient)
        factor = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (step - 1) / steps))
        actual_rates = rates * np.float32(factor)
        reference.lrs.copy_(actual_rates)
        candidate.lrs.copy_(actual_rates)
        old.grad, new.grad = g.clone(), g.clone()
        reference.step()
        candidate.step()
        max_weights = max(max_weights, float((old - new).abs().max().detach()))
        max_m = max(max_m, float((reference.m[0] - candidate.m).abs().max()))
        max_v = max(max_v, float((reference.v[0] - candidate.v).abs().max()))
        torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-6)
        if step in (1, 8, 64, steps):
            snapshots.append(
                dict(
                    step=step,
                    parameters_bitwise=torch.equal(old, new),
                    max_parameter_abs=float((old - new).abs().max().detach()),
                )
            )
    return dict(
        steps=steps,
        slots=6,
        parameters_per_slot=101,
        max_parameter_abs=max_weights,
        max_m_abs=max_m,
        max_v_abs=max_v,
        snapshots=snapshots,
        tolerance="parameter atol=rtol=2e-6; not a bitwise identity requirement",
    )


def coupled_probe(steps=64):
    rng = np.random.default_rng(32901)
    x = torch.from_numpy(rng.normal(size=(6, 3, 36)).astype(np.float32))
    tokens = torch.from_numpy(rng.normal(size=(6, 3, 64, 18)).astype(np.float32))
    base = torch.from_numpy(rng.normal(0, 0.2, (6, 3, 3)).astype(np.float32))
    target = torch.from_numpy(rng.normal(0, 0.3, (6, 3, 3)).astype(np.float32))
    old, new = Bank("soft_small"), Bank("soft_small")
    assert torch.equal(old.theta, new.theta)
    rates = np.array([1e-5, 1e-4] * 3, np.float32)
    reference = BankAdamW(old.parameters(), rates.copy())
    candidate = FusedBankAdamW(new.theta, rates.copy())
    max_weights = max_output = 0.0
    for _ in range(steps):
        for model, optimizer in ((old, reference), (new, candidate)):
            model.zero_grad(set_to_none=True)
            out, _ = model(x, tokens, base)
            losses = (out - target.unsqueeze(-2)).square().mean((1, 3))
            (0.6 * losses.mean(-1) + 0.4 * losses[:, -1]).sum().backward()
            optimizer.step()
        with torch.no_grad():
            a, _ = old(x, tokens, base)
            b, _ = new(x, tokens, base)
            max_weights = max(max_weights, float((old.theta - new.theta).abs().max()))
            max_output = max(max_output, float((a - b).abs().max()))
            torch.testing.assert_close(old.theta, new.theta, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    return dict(
        steps=steps,
        slots=6,
        architecture="soft_small four-pass residual branch",
        trainable_parameters_per_slot=old.theta.shape[1],
        max_parameter_abs=max_weights,
        max_normalized_output_abs=max_output,
        parameters_bitwise=torch.equal(old.theta, new.theta),
        tolerance="weights atol=rtol=2e-6; synthetic normalized outputs atol=rtol=1e-5",
    )


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    value = dict(
        classification="CPU synthetic numerical feasibility only; no GPU speed/capture or production-quality claim",
        stream=stream_probe(),
        coupled=coupled_probe(),
        passed=True,
        torch=torch.__version__,
        numpy=np.__version__,
        python=sys.version,
        cuda_context_initialized=torch.cuda.is_initialized(),
    )
    assert not value["cuda_context_initialized"]
    source = {}
    for p in (
        "scripts/chromaseed_fused_optimizer.py",
        "scripts/chromaseed_fused_optimizer_probe.py",
        "tests/test_chromaseed_fused_optimizer.py",
        "scripts/chromaseed_refine.py",
        "scripts/chromaseed_architecture_scale.py",
    ):
        source[p] = sha(ROOT / p)
    libraries = {}
    for name in ("torch.optim.adamw", "torch.optim.adam", "torch.utils._foreach_utils"):
        path = Path(importlib.import_module(name).__file__)
        libraries[str(path)] = sha(path)
    value.update(source_sha256=source, local_library_sha256=libraries)
    path = OUT / "probe.json"
    assert not path.exists(), "preserve this feasibility receipt; use a distinct probe for changes"
    write_json(path, value)
    print(
        "CPU FUSED PROBE",
        value["stream"],
        value["coupled"],
        "CUDA initialized",
        value["cuda_context_initialized"],
    )


if __name__ == "__main__":
    main()
