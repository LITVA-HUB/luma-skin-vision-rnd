"""Fixed18-slot long continuation of exact NP heads with bounded variation banks."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from chromaseed_gate_stability import affine_features, basic_legal
from chromaseed_local_denoise import Bank
from chromaseed_neural_prefix_numpy import Predictor
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import synchronize

SEEDS = (17, 29, 43)
MODES = (0, 16, 256)
RATES = (0.0001, 0.0003)
HORIZON = 131072
SLOTS = tuple(
    dict(seed=seed, variants=count, lr=lr) for seed in SEEDS for count in MODES for lr in RATES
)
_STREAM = None


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_pool(x, rows):
    x, rows = np.asarray(x, np.float32), np.asarray(rows)
    if (
        x.ndim != 2
        or x.shape[1] != 36
        or not len(x)
        or rows.shape != (len(x),)
        or rows.dtype.kind not in "iu"
    ):
        raise ValueError("color36 and integer source row identities required")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or not basic_legal(x).all():
        raise ValueError("legal unique source rows required")
    pool = np.empty((len(x), 257, 36), np.float32)
    pool[:, 0] = x
    for i, row in enumerate(rows):
        draws = np.random.default_rng(880003 + 1009 * int(row)).random((256, 4))
        pool[i, 1:] = affine_features(
            np.repeat(x[i : i + 1], 256, axis=0), draws[:, 3] * 4 / 255, draws[:, :3]
        )
    return pool


def uniforms(seeds, steps):
    return np.stack(
        [
            np.random.default_rng(int(seed) + 910003).random((steps, 64), dtype=np.float32)
            for seed in seeds
        ]
    )


def variant_indices(uniform, counts):
    return torch.where(
        (uniform < 0.5) | (counts == 0),
        torch.zeros_like(uniform),
        torch.floor((2 * uniform - 1) * counts) + 1,
    ).long()


def lr_factors(steps):
    if not 0 < steps <= HORIZON:
        raise ValueError("steps must fit original131072 horizon")
    return (0.1 + 0.45 * (1 + np.cos(np.pi * np.arange(steps) / (HORIZON - 1)))).astype(np.float32)


class HeadBank(torch.nn.Module):
    layer = Bank.layer

    def __init__(self, warm):
        super().__init__()
        if len(warm) != 3:
            raise ValueError("three original seed warm starts required")
        for model in warm:
            Predictor(model)
            if str(model["family"]) != "blind4" or model["w0"].shape != (36, 16):
                raise ValueError("NP643-parameter blind head required")
            for key in ("x_mean", "x_std", "y_mean", "y_std", "family", "original_k", "prefix"):
                np.testing.assert_array_equal(model[key], warm[0][key])
        self.d, self.h, self.m = 36, 16, 18
        initial = []
        for slot in SLOTS:
            model = warm[SEEDS.index(slot["seed"])]
            initial.append(np.concatenate([model[k].reshape(-1) for k in ("w0", "b0", "v0", "c0")]))
        self.theta = torch.nn.Parameter(torch.from_numpy(np.stack(initial)))

    def forward(self, x):
        return self.layer(x, self.theta)

    def export(self, slot, warm):
        row = self.theta.detach()[slot].cpu().numpy()
        result = {k: v.copy() for k, v in warm[SEEDS.index(SLOTS[slot]["seed"])].items()}
        result.update(
            w0=row[:576].reshape(36, 16).copy(),
            b0=row[576:592].copy(),
            v0=row[592:640].reshape(16, 3).copy(),
            c0=row[640:].copy(),
        )
        return result


def sample_inventory(indices, uniform, n):
    records = []
    for si, seed in enumerate(SEEDS):
        for count in MODES:
            v = (
                np.zeros_like(indices[si])
                if count == 0
                else np.where(
                    uniform[si] < 0.5, 0, np.floor((2 * uniform[si] - 1) * count) + 1
                ).astype(np.int64)
            )
            occupied = np.bincount((indices[si] * 257 + v).ravel(), minlength=n * 257).reshape(
                n, 257
            )
            records.append(
                dict(
                    seed=seed,
                    variants=count,
                    presentations=int(v.size),
                    augmented_presentations=int(np.count_nonzero(v)),
                    distinct_observed_variants=int(np.count_nonzero(occupied)),
                    distinct_source_rows=int(np.count_nonzero(occupied.sum(1))),
                )
            )
    return records


def fit(x, y, weights, rows, warm, steps, checkpoints, device="cpu", engine="eager", progress=None):
    global _STREAM
    if (
        steps <= 0
        or steps > HORIZON
        or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
        or not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != steps
    ):
        raise ValueError("checkpoints must start0 and end at positive steps within fixed horizon")
    if engine not in ("eager", "cuda_graph") or (
        engine == "cuda_graph" and not str(device).startswith("cuda")
    ):
        raise ValueError("CUDA graph requires CUDA")
    setup(device)
    synchronize(device)
    started = time.perf_counter()
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float64)
    if y.shape != (len(x), 3) or not np.isfinite(y).all():
        raise ValueError("finite matching Lab required")
    net = HeadBank(warm).to(device)
    prep = warm[0]
    for prefix, value in (("x", x.astype(float)), ("y", y)):
        np.testing.assert_array_equal(prep[prefix + "_mean"], value.mean(0).astype(np.float32))
        np.testing.assert_array_equal(
            prep[prefix + "_std"], np.maximum(value.std(0), 1e-6).astype(np.float32)
        )
    pool = make_pool(x, rows)
    pool_hash = array_hash(pool)
    xt = torch.as_tensor(((pool - prep["x_mean"]) / prep["x_std"]).reshape(-1, 36), device=device)
    yt = torch.as_tensor(((y - prep["y_mean"]) / prep["y_std"]).astype(np.float32), device=device)
    index_cpu = sampling_indices(weights, SEEDS, steps)
    uniform_cpu = uniforms(SEEDS, steps)
    index_hash, uniform_hash = array_hash(index_cpu), array_hash(uniform_cpu)
    inventory = sample_inventory(index_cpu, uniform_cpu, len(x))
    indices, uniform = [torch.as_tensor(v, device=device) for v in (index_cpu, uniform_cpu)]
    slot_seed = torch.as_tensor([SEEDS.index(s["seed"]) for s in SLOTS], device=device)
    counts = torch.as_tensor([s["variants"] for s in SLOTS], device=device)[:, None]
    rates = np.array([s["lr"] for s in SLOTS], np.float32)
    optimizer = BankAdamW(net.parameters(), rates, weight_decay=0.01, max_norm=5.0)
    # CPU as_tensor can share rates with optimizer.lrs; the immutable base needs its own storage.
    base_rates = torch.as_tensor(rates, device=device).clone()
    factors = torch.as_tensor(lr_factors(steps), device=device)
    corrections = torch.as_tensor(
        np.array([[1 - 0.9**t, np.sqrt(1 - 0.999**t)] for t in range(1, steps + 1)], np.float32),
        device=device,
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    models = {0: [net.export(i, warm) for i in range(18)]}

    def iteration(index, u, factor, correction):
        net.zero_grad(set_to_none=True)
        source = index.index_select(0, slot_seed)
        random = u.index_select(0, slot_seed)
        variants = variant_indices(random, counts)
        xb, yb = xt[source * 257 + variants], yt[source]
        output = net(xb)
        per_model = (output - yb).square().mean((1, 2))
        per_model.sum().backward()
        optimizer.lrs.copy_(base_rates * factor)
        optimizer.step(correction)
        return per_model

    graph = None
    if engine == "cuda_graph":
        counter = torch.zeros(1, dtype=torch.int64, device=device)

        def capture_step():
            index = indices.index_select(1, counter).squeeze(1)
            u = uniform.index_select(1, counter).squeeze(1)
            factor = factors.index_select(0, counter)[0]
            correction = corrections.index_select(0, counter)[0]
            loss = iteration(index, u, factor, correction)
            counter.add_(1)
            return loss

        initial = net.theta.detach().clone()

        def reset():
            with torch.no_grad():
                net.theta.copy_(initial)
                for v in optimizer.m + optimizer.v:
                    v.zero_()
                optimizer.lrs.copy_(base_rates)
                optimizer.t = 0
                counter.zero_()

        if _STREAM is None:
            _STREAM = torch.cuda.Stream()
        _STREAM.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_STREAM):
            for _ in range(min(3, steps)):
                capture_step()
        torch.cuda.current_stream().wait_stream(_STREAM)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = capture_step()
        reset()
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    trace = []
    for step in range(1, steps + 1):
        if graph is None:
            loss = iteration(
                indices[:, step - 1], uniform[:, step - 1], factors[step - 1], corrections[step - 1]
            )
        else:
            graph.replay()
        if step in checkpoints or step % 8192 == 0:
            synchronize(device)
            losses = loss.detach().cpu().numpy()
            if not np.isfinite(losses).all() or not torch.isfinite(net.theta).all():
                raise ValueError("nonfinite LT trajectory")
            entry = dict(
                step=step, seconds=time.perf_counter() - started, model_loss=losses.tolist()
            )
            trace.append(entry)
            if progress is not None:
                progress(entry)
        if step in checkpoints:
            models[step] = [net.export(i, warm) for i in range(18)]
    synchronize(device)
    return models, dict(
        steps=steps,
        checkpoints=list(checkpoints),
        schedule_horizon=HORIZON,
        trajectory_count=18,
        original_rows=len(x),
        pool_rows=len(x) * 257,
        pool_sha256=pool_hash,
        sampling_indices_sha256=index_hash,
        uniforms_sha256=uniform_hash,
        sample_inventory=inventory,
        setup_seconds=setup_seconds,
        full_bank_seconds=time.perf_counter() - started,
        trace=trace,
        engine=engine,
        device=device,
        slots=list(SLOTS),
        numeric_bytes=2886,
        final_learning_rates=optimizer.lrs.detach().cpu().numpy().tolist(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    )
