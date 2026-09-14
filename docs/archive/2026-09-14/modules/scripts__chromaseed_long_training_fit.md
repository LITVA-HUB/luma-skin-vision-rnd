# `scripts/chromaseed_long_training_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_long_training_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed18-slot long continuation of exact NP heads with bounded variation banks.

SHA-256 исходника: `e04b95f792b995acc38c673999a81a57e6c39abc4c4d6ea62d5913bb50f2bc32`. Строк: **285**.

## Зависимости

```python
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
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_long_training_fit.py#L17)

```python
SEEDS = (17, 29, 43)
```

[Строка 18](../../../../scripts/chromaseed_long_training_fit.py#L18)

```python
MODES = (0, 16, 256)
```

[Строка 19](../../../../scripts/chromaseed_long_training_fit.py#L19)

```python
RATES = (0.0001, 0.0003)
```

[Строка 20](../../../../scripts/chromaseed_long_training_fit.py#L20)

```python
HORIZON = 131072
```

[Строка 21](../../../../scripts/chromaseed_long_training_fit.py#L21)

```python
SLOTS = tuple(
    dict(seed=seed, variants=count, lr=lr) for seed in SEEDS for count in MODES for lr in RATES
)
```

[Строка 24](../../../../scripts/chromaseed_long_training_fit.py#L24)

```python
_STREAM = None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `HeadBank` | torch.nn.Module | [L76](../../../../scripts/chromaseed_long_training_fit.py#L76) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `array_hash` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_long_training_fit.py#L27) |
| `make_pool` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_long_training_fit.py#L31) |
| `uniforms` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_long_training_fit.py#L53) |
| `variant_indices` | FunctionDef | См. реализацию | [L62](../../../../scripts/chromaseed_long_training_fit.py#L62) |
| `lr_factors` | FunctionDef | См. реализацию | [L70](../../../../scripts/chromaseed_long_training_fit.py#L70) |
| `HeadBank` | ClassDef | См. реализацию | [L76](../../../../scripts/chromaseed_long_training_fit.py#L76) |
| `sample_inventory` | FunctionDef | См. реализацию | [L111](../../../../scripts/chromaseed_long_training_fit.py#L111) |
| `fit` | FunctionDef | См. реализацию | [L138](../../../../scripts/chromaseed_long_training_fit.py#L138) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L79–94</summary>

```python
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
```

</details>

<details><summary>forward · L96–97</summary>

```python
def forward(self, x):
        return self.layer(x, self.theta)
```

</details>

<details><summary>export · L99–108</summary>

```python
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
```

</details>

<details><summary>fit · L138–285</summary>

```python
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
```

</details>
