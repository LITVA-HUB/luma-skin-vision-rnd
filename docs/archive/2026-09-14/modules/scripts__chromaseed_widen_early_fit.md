# `scripts/chromaseed_widen_early_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_early_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Rate-pair adapter preserving the original WIDE bank and8192-step schedule.

SHA-256 исходника: `139bb79ac4226be8e067187123a81a77495329aa9b901a6af9b4066cbc441f07`. Строк: **185**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
import torch
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from chromaseed_widen import HORIZON, SEEDS, Bank, array_hash, rate_factors, token_normalizers
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_widen_early_fit.py#L14)

```python
FIRST_RATES = (0.00001, 0.00003, 0.0001)
```

[Строка 15](../../../../scripts/chromaseed_widen_early_fit.py#L15)

```python
_STREAM = None
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_widen_early_fit.py#L18) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L18–185</summary>

```python
def fit(
    x,
    tokens,
    y,
    weights,
    warm,
    variant,
    first_rate,
    steps,
    checkpoints,
    device="cpu",
    engine="eager",
    progress=None,
):
    global _STREAM
    if first_rate not in FIRST_RATES:
        raise ValueError("registered first rate required")
    slots = tuple(dict(seed=s, lr=r) for s in SEEDS for r in (first_rate, 0.001))
    if (
        not 0 < steps <= HORIZON
        or tuple(sorted(set(checkpoints))) != tuple(checkpoints)
        or not checkpoints
        or checkpoints[0] != 0
        or checkpoints[-1] != steps
    ):
        raise ValueError("ordered checkpoints start0 and end at steps within fixed horizon")
    if engine not in ("eager", "cuda_graph") or (
        engine == "cuda_graph" and not str(device).startswith("cuda")
    ):
        raise ValueError("CUDA graph requires CUDA")
    setup(device)
    synchronize(device)
    start = time.perf_counter()
    x, tokens, y = (
        np.asarray(x, np.float32),
        np.asarray(tokens, np.float32),
        np.asarray(y, np.float64),
    )
    if (
        x.shape != (len(y), 36)
        or tokens.shape != (len(y), 64, 18)
        or y.shape != (len(x), 3)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("matching finite original FIT arrays required")
    prep = token_normalizers(tokens)
    for name, data in (("x", x.astype(np.float64)), ("y", y)):
        np.testing.assert_array_equal(warm[0][name + "_mean"], data.mean(0).astype(np.float32))
        np.testing.assert_array_equal(
            warm[0][name + "_std"], np.maximum(data.std(0), 1e-6).astype(np.float32)
        )
    net = Bank(warm, variant).to(device)
    normal = warm[0]
    xt, tt, yt = [
        torch.as_tensor(v, device=device)
        for v in (
            (x - normal["x_mean"]) / normal["x_std"],
            (tokens - prep["t_mean"]) / prep["t_std"],
            ((y - normal["y_mean"]) / normal["y_std"]).astype(np.float32),
        )
    ]
    cpu_indices = sampling_indices(weights, SEEDS, steps)
    indices = torch.as_tensor(cpu_indices, device=device)
    slot_seed = torch.as_tensor([i for i in range(3) for _ in range(2)], device=device)
    rates = np.array([s["lr"] for s in slots], np.float32)
    optimizer = BankAdamW(net.parameters(), rates, weight_decay=0.01, max_norm=5.0)
    base_rates = torch.as_tensor(rates, device=device).clone()
    factors = torch.as_tensor(rate_factors(steps), device=device)
    corrections = torch.as_tensor(
        np.array([[1 - 0.9**t, np.sqrt(1 - 0.999**t)] for t in range(1, steps + 1)], np.float32),
        device=device,
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    models = {0: [net.export(i, warm, prep) for i in range(6)]}

    def iteration(index, factor, correction):
        net.zero_grad(set_to_none=True)
        index = index.index_select(0, slot_seed)
        loss = (net(xt[index], tt[index]) - yt[index]).square().mean((1, 2))
        loss.sum().backward()
        optimizer.lrs.copy_(base_rates * factor)
        optimizer.step(correction)
        return loss

    graph = None
    if engine == "cuda_graph":
        counter = torch.zeros(1, dtype=torch.int64, device=device)

        def capture():
            loss = iteration(
                indices.index_select(1, counter).squeeze(1),
                factors.index_select(0, counter)[0],
                corrections.index_select(0, counter)[0],
            )
            counter.add_(1)
            return loss

        initial = net.theta.detach().clone()

        def reset():
            with torch.no_grad():
                net.theta.copy_(initial)
                for v in optimizer.m + optimizer.v:
                    v.zero_()
                optimizer.t = 0
                optimizer.lrs.copy_(base_rates)
                counter.zero_()

        if _STREAM is None:
            _STREAM = torch.cuda.Stream()
        _STREAM.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_STREAM):
            for _ in range(min(3, steps)):
                capture()
        torch.cuda.current_stream().wait_stream(_STREAM)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = capture()
        reset()
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    trace = []
    for step in range(1, steps + 1):
        if graph is None:
            loss = iteration(indices[:, step - 1], factors[step - 1], corrections[step - 1])
        else:
            graph.replay()
        if step in checkpoints or step % 512 == 0:
            synchronize(device)
            value = loss.detach().cpu().numpy()
            if not np.isfinite(value).all() or not torch.isfinite(net.theta).all():
                raise ValueError("nonfinite WE trajectory")
            item = dict(
                step=step, seconds=time.perf_counter() - start, minibatch_loss=value.tolist()
            )
            trace.append(item)
            if progress is not None:
                progress(item)
        if step in checkpoints:
            models[step] = [net.export(i, warm, prep) for i in range(6)]
    synchronize(device)
    info = dict(
        steps=steps,
        checkpoints=list(checkpoints),
        schedule_horizon=HORIZON,
        trajectory_count=6,
        variant=variant,
        first_rate=first_rate,
        slots=list(slots),
        original_rows=len(x),
        sampling_sha256=array_hash(cpu_indices),
        fit_tokens_sha256=array_hash(tokens),
        token_mean=prep["t_mean"].tolist(),
        token_std=prep["t_std"].tolist(),
        setup_seconds=setup_seconds,
        full_bank_seconds=time.perf_counter() - start,
        trace=trace,
        engine=engine,
        device=device,
        final_learning_rates=optimizer.lrs.detach().cpu().numpy().tolist(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
    )
    return models, info
```

</details>
