# `scripts/chromaseed_local_denoise_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Complete independent ND model construction, with optional CUDA Graph batching.

SHA-256 исходника: `5e85850114316bab4ca6af34f03d14c0898c088e18edd21cd7235b50ac9b0ee9`. Строк: **162**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
import torch
from chromaseed_local_denoise import Bank, preprocessor
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_local_denoise_fit.py#L14)

```python
_STREAM = None
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `noise_sequences` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_local_denoise_fit.py#L17) |
| `fit` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_local_denoise_fit.py#L27) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L27–162</summary>

```python
def fit(
    x, y, weights, family, slots, steps, checkpoints, device="cpu", engine="eager", progress=None
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
        raise ValueError("positive finite learning rates required")
    net = Bank(family, seeds).to(device)
    optimizer = BankAdamW(net.parameters(), np.repeat(lrs, net.k), weight_decay=0.01, max_norm=5.0)
    xt, yt = [torch.as_tensor(v, device=device) for v in (xn, yn)]
    indices = torch.as_tensor(sampling_indices(weights, seeds, steps), device=device)
    noisy = family in ("local2", "local4")
    noise = torch.as_tensor(noise_sequences(seeds, steps, net.k), device=device) if noisy else None
    a = torch.as_tensor(net.a[:-1], dtype=torch.float32, device=device)[None, :, None, None]
    s = torch.as_tensor(net.s[:-1], dtype=torch.float32, device=device)[None, :, None, None]
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def iteration(index, epsilon=None, corrections=None):
        net.zero_grad(set_to_none=True)
        xb, yb = xt[index], yt[index]
        if family == "e2e4":
            output = net.rollout(xb)
        else:
            state = (
                a * yb[:, None] + s * epsilon
                if noisy
                else torch.zeros((len(slots), net.k, 64, 3), device=device)
            )
            output = net.local(xb, state)
        per_block = (output - yb[:, None]).square().mean((2, 3))
        loss = per_block.sum()
        loss.backward()
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
                raise ValueError("nonfinite ND trajectory")
            entry = dict(
                step=step, seconds=time.perf_counter() - started, block_loss=losses.tolist()
            )
            trace.append(entry)
            if progress is not None:
                progress(entry)
        if step in checkpoints:
            models[step] = [net.export(i, prep) for i in range(len(slots))]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return models, dict(
        family=family,
        slots=[dict(seed=int(seed), lr=float(lr)) for seed, lr in slots],
        steps=steps,
        checkpoints=list(checkpoints),
        batch_size=64,
        n_fit=len(x),
        engine=engine,
        device=device,
        parameters=net.k * net.p,
        numeric_bytes=sum(v.nbytes for v in models[steps][0].values() if v.dtype.kind != "U"),
        setup_seconds=setup_seconds,
        full_bank_seconds=elapsed,
        trace=trace,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    )
```

</details>
