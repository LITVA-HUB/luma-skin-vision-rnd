# `scripts/chromaseed_patch8_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_patch8_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed18-slot continuation of NP with a small optional local branch.

SHA-256 исходника: `8a8a8516e0131260ba566e8c991500c3f71b383e67ca2b90c9eeee5c06f2fe50`. Строк: **263**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
import torch
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_patch8_numpy import Predictor
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_patch8_fit.py#L16)

```python
SEEDS = (17, 29, 43)
```

[Строка 17](../../../../scripts/chromaseed_patch8_fit.py#L17)

```python
ARMS = ("stats", "patch8")
```

[Строка 18](../../../../scripts/chromaseed_patch8_fit.py#L18)

```python
RATES = (0.0001, 0.0003, 0.001)
```

[Строка 19](../../../../scripts/chromaseed_patch8_fit.py#L19)

```python
HORIZON = 32768
```

[Строка 20](../../../../scripts/chromaseed_patch8_fit.py#L20)

```python
SLOTS = tuple(dict(seed=s, arm=a, lr=r) for s in SEEDS for a in ARMS for r in RATES)
```

[Строка 21](../../../../scripts/chromaseed_patch8_fit.py#L21)

```python
_STREAM = None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Bank` | torch.nn.Module | [L44](../../../../scripts/chromaseed_patch8_fit.py#L44) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `array_hash` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_patch8_fit.py#L24) |
| `token_normalizers` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_patch8_fit.py#L28) |
| `rate_factors` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_patch8_fit.py#L38) |
| `Bank` | ClassDef | См. реализацию | [L44](../../../../scripts/chromaseed_patch8_fit.py#L44) |
| `fit` | FunctionDef | См. реализацию | [L112](../../../../scripts/chromaseed_patch8_fit.py#L112) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L45–70</summary>

```python
def __init__(self, warm):
        super().__init__()
        if len(warm) != 3:
            raise ValueError("three NP warm starts required")
        for m in warm:
            BasePredictor(m)
            if str(m["family"]) != "blind4" or m["w0"].shape != (36, 16):
                raise ValueError("NP643 warm head required")
            for k in ("family", "original_k", "prefix", "x_mean", "x_std", "y_mean", "y_std"):
                np.testing.assert_array_equal(m[k], warm[0][k])
        initial = np.zeros((18, 1179), np.float32)
        local = {
            s: np.random.default_rng(s + 770003)
            .uniform(-1 / np.sqrt(18), 1 / np.sqrt(18), (18, 8))
            .astype(np.float32)
            for s in SEEDS
        }
        for i, slot in enumerate(SLOTS):
            m = warm[SEEDS.index(slot["seed"])]
            initial[i, :643] = np.concatenate([m[k].ravel() for k in ("w0", "b0", "v0", "c0")])
            initial[i, 643:787] = local[slot["seed"]].ravel()
        self.theta = torch.nn.Parameter(torch.from_numpy(initial))
        self.register_buffer(
            "active",
            torch.tensor([s["arm"] == "patch8" for s in SLOTS], dtype=torch.float32)[:, None, None],
        )
```

</details>

<details><summary>forward · L72–88</summary>

```python
def forward(self, x, tokens):
        p = self.theta
        w, b, v, c = (
            p[:, :576].reshape(18, 36, 16),
            p[:, 576:592],
            p[:, 592:640].reshape(18, 16, 3),
            p[:, 640:643],
        )
        u, e, g = p[:, 643:787].reshape(18, 18, 8), p[:, 787:795], p[:, 795:].reshape(18, 24, 16)
        local = torch.relu(torch.bmm(tokens.reshape(18, -1, 18), u) + e[:, None, :]).reshape(
            18, x.shape[1], 64, 8
        )
        mean = local.mean(2)
        spread = ((local - mean[:, :, None, :]).square().mean(2) + 1e-6).sqrt()
        pooled = torch.cat((mean, spread, local.amax(2)), dim=-1) * self.active
        hidden = torch.relu((torch.bmm(x, w) + b[:, None, :]) + torch.bmm(pooled, g))
        return torch.bmm(hidden, v) + c[:, None, :]
```

</details>

<details><summary>export · L90–109</summary>

```python
def export(self, slot, warm, prep):
        row = self.theta.detach()[slot].cpu().numpy()
        out = {k: v.copy() for k, v in warm[slot // 6].items()}
        out.update(
            w0=row[:576].reshape(36, 16).copy(),
            b0=row[576:592].copy(),
            v0=row[592:640].reshape(16, 3).copy(),
            c0=row[640:643].copy(),
        )
        if SLOTS[slot]["arm"] == "patch8":
            out.update(
                u=row[643:787].reshape(18, 8).copy(),
                e=row[787:795].copy(),
                g=row[795:].reshape(24, 16).copy(),
                **{k: v.copy() for k, v in prep.items()},
            )
        else:
            assert np.count_nonzero(row[795:]) == 0
        Predictor(out)
        return out
```

</details>

<details><summary>fit · L112–263</summary>

```python
def fit(
    x, tokens, y, weights, warm, steps, checkpoints, device="cpu", engine="eager", progress=None
):
    global _STREAM
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
    net = Bank(warm).to(device)
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
    slot_seed = torch.as_tensor([i for i in range(3) for _ in range(6)], device=device)
    rates = np.array([s["lr"] for s in SLOTS], np.float32)
    optimizer = BankAdamW(net.parameters(), rates, weight_decay=0.01, max_norm=5.0)
    base_rates = torch.as_tensor(rates, device=device).clone()
    factors = torch.as_tensor(rate_factors(steps), device=device)
    corrections = torch.as_tensor(
        np.array([[1 - 0.9**t, np.sqrt(1 - 0.999**t)] for t in range(1, steps + 1)], np.float32),
        device=device,
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    models = {0: [net.export(i, warm, prep) for i in range(18)]}

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
                raise ValueError("nonfinite P8 trajectory")
            item = dict(
                step=step, seconds=time.perf_counter() - start, minibatch_loss=value.tolist()
            )
            trace.append(item)
            if progress is not None:
                progress(item)
        if step in checkpoints:
            models[step] = [net.export(i, warm, prep) for i in range(18)]
    synchronize(device)
    info = dict(
        steps=steps,
        checkpoints=list(checkpoints),
        schedule_horizon=HORIZON,
        trajectory_count=18,
        slots=list(SLOTS),
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
