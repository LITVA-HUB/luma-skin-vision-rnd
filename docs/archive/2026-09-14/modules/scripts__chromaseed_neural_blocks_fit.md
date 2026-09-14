# `scripts/chromaseed_neural_blocks_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Complete ND-equivalent construction of independently retained local blocks.

SHA-256 исходника: `abe549765f230b0a7cfab51fddcf2e1e56b540bddd0477cf4ad0a773c57efe95`. Строк: **255**.

## Зависимости

```python
from __future__ import annotations
import math
import time
import numpy as np
import torch
from chromaseed_local_denoise import SPECS, Bank, preprocessor, schedule
from chromaseed_local_denoise_fit import noise_sequences
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_neural_blocks_fit.py#L17)

```python
_STREAM = None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `RetainedBank` | torch.nn.Module | [L39](../../../../scripts/chromaseed_neural_blocks_fit.py#L39) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `blocks_for` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_neural_blocks_fit.py#L20) |
| `retained_noise` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_neural_blocks_fit.py#L33) |
| `RetainedBank` | ClassDef | См. реализацию | [L39](../../../../scripts/chromaseed_neural_blocks_fit.py#L39) |
| `to_prefix` | FunctionDef | См. реализацию | [L80](../../../../scripts/chromaseed_neural_blocks_fit.py#L80) |
| `fit` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_neural_blocks_fit.py#L103) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L43–65</summary>

```python
def __init__(self, family, seeds, prefix):
        super().__init__()
        if not seeds:
            raise ValueError("nonempty model slots required")
        self.block_indices = blocks_for(family, prefix)
        self.family, self.prefix, self.m = family, int(prefix), len(seeds)
        self.original_k, self.d, self.h = SPECS[family]
        self.k = len(self.block_indices)
        self.p = (self.d + 1) * self.h + (self.h + 1) * 3
        initial = np.zeros((self.m, self.k, self.p), np.float32)
        for i, seed in enumerate(seeds):
            for slot, original in enumerate(self.block_indices):
                rng = np.random.default_rng(int(seed) + 310003 + 1009 * int(original))
                initial[i, slot, : self.d * self.h] = rng.uniform(
                    -1 / math.sqrt(self.d), 1 / math.sqrt(self.d), self.d * self.h
                )
                start = (self.d + 1) * self.h
                initial[i, slot, start : start + self.h * 3] = rng.uniform(
                    -1 / math.sqrt(self.h), 1 / math.sqrt(self.h), self.h * 3
                )
        self.theta = torch.nn.Parameter(torch.from_numpy(initial.reshape(self.m * self.k, self.p)))
        a, s = schedule(self.original_k)
        self.a, self.s = a[self.block_indices], s[self.block_indices]
```

</details>

<details><summary>export · L67–77</summary>

```python
def export(self, slot, prep):
        if not 0 <= slot < self.m:
            raise ValueError("invalid model slot")
        return {
            **{k: v.copy() for k, v in prep.items()},
            "family": np.array(self.family),
            "original_k": np.array(self.original_k, np.uint8),
            "prefix": np.array(self.prefix, np.uint8),
            "block_indices": self.block_indices.copy(),
            "theta": self.theta.detach().reshape(self.m, self.k, self.p)[slot].cpu().numpy().copy(),
        }
```

</details>

<details><summary>fit · L103–255</summary>

```python
def fit(
    x,
    y,
    weights,
    family,
    slots,
    prefix,
    steps,
    checkpoints,
    device="cpu",
    engine="eager",
    progress=None,
):
    global _STREAM
    if (
        steps <= 0
        or not slots
        or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
        or not checkpoints
        or checkpoints[-1] != steps
        or checkpoints[0] <= 0
    ):
        raise ValueError("positive ordered checkpoints ending at steps required")
    if engine not in ("eager", "cuda_graph") or (
        engine == "cuda_graph" and not str(device).startswith("cuda")
    ):
        raise ValueError("CUDA graph requires CUDA")
    setup(device)
    synchronize(device)
    started = time.perf_counter()
    prep = preprocessor(x, y)
    xn = (np.asarray(x, np.float32) - prep["x_mean"]) / prep["x_std"]
    yn = ((np.asarray(y, np.float64) - prep["y_mean"]) / prep["y_std"]).astype(np.float32)
    seeds, lrs = zip(*slots, strict=True)
    if not all(np.isfinite(lr) and lr > 0 for lr in lrs):
        raise ValueError("positive finite rates required")
    net = RetainedBank(family, seeds, prefix).to(device)
    optimizer = BankAdamW(net.parameters(), np.repeat(lrs, net.k), weight_decay=0.01, max_norm=5.0)
    xt, yt = [torch.as_tensor(v, device=device) for v in (xn, yn)]
    indices = torch.as_tensor(sampling_indices(weights, seeds, steps), device=device)
    noisy = family in ("local2", "local4")
    noise = (
        torch.as_tensor(
            retained_noise(seeds, steps, net.original_k, net.block_indices), device=device
        )
        if noisy
        else None
    )
    a = torch.as_tensor(net.a, dtype=torch.float32, device=device)[None, :, None, None]
    s = torch.as_tensor(net.s, dtype=torch.float32, device=device)[None, :, None, None]
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def iteration(index, epsilon=None, corrections=None):
        net.zero_grad(set_to_none=True)
        xb, yb = xt[index], yt[index]
        state = (
            a * yb[:, None] + s * epsilon
            if noisy
            else torch.zeros((len(slots), net.k, 64, 3), device=device)
        )
        output = net.local(xb, state)
        per_block = (output - yb[:, None]).square().mean((2, 3))
        per_block.sum().backward()
        optimizer.step(corrections)
        return per_block

    graph = None
    if engine == "cuda_graph":
        corrections = torch.as_tensor(
            np.array(
                [[1 - 0.9**t, np.sqrt(1 - 0.999**t)] for t in range(1, steps + 1)], np.float32
            ),
            device=device,
        )
        counter = torch.zeros(1, dtype=torch.int64, device=device)

        def capture_step():
            index = indices.index_select(1, counter).squeeze(1)
            eps = noise.index_select(1, counter).squeeze(1) if noisy else None
            correction = corrections.index_select(0, counter)[0]
            loss = iteration(index, eps, correction)
            counter.add_(1)
            return loss

        initial = net.theta.detach().clone()

        def reset():
            with torch.no_grad():
                net.theta.copy_(initial)
                for value in optimizer.m + optimizer.v:
                    value.zero_()
                counter.zero_()
                optimizer.t = 0

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
    models, trace = {}, []
    for step in range(1, steps + 1):
        if graph is None:
            loss = iteration(indices[:, step - 1], None if noise is None else noise[:, step - 1])
        else:
            graph.replay()
        if step in checkpoints or step % 256 == 0:
            synchronize(device)
            losses = loss.detach().cpu().numpy()
            if not np.isfinite(losses).all() or not torch.isfinite(net.theta).all():
                raise ValueError("nonfinite retained trajectory")
            entry = dict(
                step=step, seconds=time.perf_counter() - started, block_loss=losses.tolist()
            )
            trace.append(entry)
            if progress is not None:
                progress(entry)
        if step in checkpoints:
            models[step] = [net.export(i, prep) for i in range(len(slots))]
    synchronize(device)
    return models, dict(
        family=family,
        slots=[dict(seed=int(seed), lr=float(lr)) for seed, lr in slots],
        prefix=prefix,
        block_indices=net.block_indices.tolist(),
        original_k=net.original_k,
        steps=steps,
        checkpoints=list(checkpoints),
        batch_size=64,
        n_fit=len(x),
        engine=engine,
        device=device,
        retained_training_parameters=net.k * net.p,
        setup_seconds=setup_seconds,
        full_bank_seconds=time.perf_counter() - started,
        trace=trace,
        generated_noise_bytes=0
        if not noisy
        else len(set(seeds)) * steps * net.original_k * 64 * 3 * 4,
        gpu_noise_bytes=0 if noise is None else noise.nelement() * noise.element_size(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    )
```

</details>
