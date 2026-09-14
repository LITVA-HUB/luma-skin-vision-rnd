# `scripts/chromaseed_widen.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Quality-first patch/color models: common warm function, independent capacity banks.

SHA-256 исходника: `be47c8e0aa03e7641ce0fb9bec75a5947527b50a9c186f90ec9d16713d37cdeb`. Строк: **372**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
import torch
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_patch8_fit import array_hash, token_normalizers
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_widen.py#L15)

```python
SPECS = {
    "tiny": (8, 16),
    "m31": (64, 128),
    "m61": (64, 256),
    "m111": (128, 256),
    "m832": (256, 1024),
}
```

[Строка 22](../../../../scripts/chromaseed_widen.py#L22)

```python
SEEDS = (17, 29, 43)
```

[Строка 23](../../../../scripts/chromaseed_widen.py#L23)

```python
RATES = (0.0001, 0.001)
```

[Строка 24](../../../../scripts/chromaseed_widen.py#L24)

```python
SLOTS = tuple(dict(seed=s, lr=r) for s in SEEDS for r in RATES)
```

[Строка 25](../../../../scripts/chromaseed_widen.py#L25)

```python
HORIZON = 8192
```

[Строка 26](../../../../scripts/chromaseed_widen.py#L26)

```python
_STREAM = None
```

[Строка 27](../../../../scripts/chromaseed_widen.py#L27)

```python
NORMAL = {"x_mean": 36, "x_std": 36, "y_mean": 3, "y_std": 3, "t_mean": 18, "t_std": 18}
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L48](../../../../scripts/chromaseed_widen.py#L48) |
| `Bank` | torch.nn.Module | [L128](../../../../scripts/chromaseed_widen.py#L128) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `shapes` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_widen.py#L30) |
| `rate_factors` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_widen.py#L35) |
| `capacity` | FunctionDef | См. реализацию | [L41](../../../../scripts/chromaseed_widen.py#L41) |
| `Predictor` | ClassDef | One color36 +64x18 tokens, including patch preparation in every response. | [L48](../../../../scripts/chromaseed_widen.py#L48) |
| `predict` | FunctionDef | См. реализацию | [L111](../../../../scripts/chromaseed_widen.py#L111) |
| `Bank` | ClassDef | См. реализацию | [L128](../../../../scripts/chromaseed_widen.py#L128) |
| `fit` | FunctionDef | См. реализацию | [L210](../../../../scripts/chromaseed_widen.py#L210) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>capacity · L41–45</summary>

```python
def capacity(model):
    return dict(
        parameters=sum(model[k].size for k in shapes(str(model["variant"]))),
        numeric_bytes=sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc"),
    )
```

</details>

<details><summary>__init__ · L51–83</summary>

```python
def __init__(self, model):
        variant = str(model["variant"])
        if variant not in SPECS or np.asarray(model["variant"]).shape != ():
            raise ValueError("unknown architecture")
        expected = {"variant", "warm_original_k", "warm_prefix", *NORMAL, *shapes(variant)}
        if set(model) != expected:
            raise ValueError("unexpected or missing model arrays")
        self.prep = {}
        self.weights = {}
        for k, n in NORMAL.items():
            v = np.asarray(model[k])
            if (
                v.shape != (n,)
                or v.dtype != np.float32
                or not np.isfinite(v).all()
                or (k.endswith("std") and np.any(v <= 0))
            ):
                raise ValueError("invalid fit normalizer")
            self.prep[k] = v.copy()
        for k, shape in shapes(variant).items():
            v = np.asarray(model[k])
            if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
                raise ValueError("invalid learned weights")
            self.weights[k] = v.astype(np.float64)
        for k in ("warm_original_k", "warm_prefix"):
            v = np.asarray(model[k])
            if v.shape != () or v.dtype != np.uint8:
                raise ValueError("invalid warm lineage scalar")
        if int(model["warm_original_k"]) != 4 or not 1 <= int(model["warm_prefix"]) <= 4:
            raise ValueError("NP blind4 warm lineage required")
        self.cached_array_bytes = sum(v.nbytes for v in self.prep.values()) + sum(
            v.nbytes for v in self.weights.values()
        )
```

</details>

<details><summary>__call__ · L99–108</summary>

```python
def __call__(self, x, tokens):
        x, tokens = np.asarray(x, np.float32), np.asarray(tokens, np.float32)
        if (
            x.shape != (36,)
            or tokens.shape != (64, 18)
            or not np.isfinite(x).all()
            or not np.isfinite(tokens).all()
        ):
            raise ValueError("one finite color36 and tokens64x18 required")
        return self._run(x, tokens)
```

</details>

<details><summary>predict · L111–125</summary>

```python
def predict(model, x, tokens):
    x, tokens = np.asarray(x, np.float32), np.asarray(tokens, np.float32)
    if (
        x.ndim != 2
        or x.shape[1] != 36
        or not len(x)
        or tokens.shape != (len(x), 64, 18)
        or not np.isfinite(x).all()
        or not np.isfinite(tokens).all()
    ):
        raise ValueError("finite matching colorNx36 and tokensNx64x18 required")
    consumer = Predictor(model)
    return np.concatenate(
        [consumer._run(x[i : i + 32], tokens[i : i + 32]) for i in range(0, len(x), 32)]
    )
```

</details>

<details><summary>__init__ · L129–166</summary>

```python
def __init__(self, warm, variant):
        super().__init__()
        if variant not in SPECS or len(warm) != 3:
            raise ValueError("known capacity and three NP warm models required")
        self.variant = variant
        self.slices = {}
        start = 0
        for k, shape in shapes(variant).items():
            stop = start + int(np.prod(shape))
            self.slices[k] = (slice(start, stop), shape)
            start = stop
        initial = np.zeros((6, start), np.float32)
        e, h = SPECS[variant]
        for si, seed in enumerate(SEEDS):
            old = warm[si]
            BasePredictor(old)
            if str(old["family"]) != "blind4" or old["w0"].shape != (36, 16):
                raise ValueError("NP643 warm head required")
            for key in ("family", "original_k", "prefix", "x_mean", "x_std", "y_mean", "y_std"):
                np.testing.assert_array_equal(old[key], warm[0][key])
            values = {k: np.zeros(shape, np.float32) for k, shape in shapes(variant).items()}
            values["w0"][:, :16] = old["w0"]
            values["w0"][:, 16:] = (
                np.random.default_rng(seed + 880003)
                .uniform(-1 / 6, 1 / 6, (36, h - 16))
                .astype(np.float32)
            )
            values["b0"][:16] = old["b0"]
            values["v0"][:16] = old["v0"]
            values["c0"][:] = old["c0"]
            values["u"][:] = (
                np.random.default_rng(seed + 770003)
                .uniform(-1 / np.sqrt(18), 1 / np.sqrt(18), (18, e))
                .astype(np.float32)
            )
            for k, (sl, _) in self.slices.items():
                initial[2 * si : 2 * si + 2, sl] = values[k].ravel()
        self.theta = torch.nn.Parameter(torch.from_numpy(initial))
```

</details>

<details><summary>forward · L171–181</summary>

```python
def forward(self, x, tokens):
        p = self.parts()
        e, _ = SPECS[self.variant]
        local = torch.relu(
            torch.bmm(tokens.reshape(6, -1, 18), p["u"]) + p["e"][:, None, :]
        ).reshape(6, x.shape[1], 64, e)
        mean = local.mean(2)
        spread = ((local - mean[:, :, None, :]).square().mean(2) + 1e-6).sqrt()
        pool = torch.cat((mean, spread, local.amax(2)), dim=-1)
        hidden = torch.relu((torch.bmm(x, p["w0"]) + p["b0"][:, None, :]) + torch.bmm(pool, p["g"]))
        return torch.bmm(hidden, p["v0"]) + p["c0"][:, None, :]
```

</details>

<details><summary>export · L183–194</summary>

```python
def export(self, slot, warm, prep):
        old = warm[slot // 2]
        out = {k: old[k].copy() for k in NORMAL if k not in ("t_mean", "t_std")}
        out.update({k: v.copy() for k, v in prep.items()})
        out.update(
            variant=np.array(self.variant),
            warm_original_k=old["original_k"].copy(),
            warm_prefix=old["prefix"].copy(),
        )
        out.update({k: p.detach()[slot].cpu().numpy().copy() for k, p in self.parts().items()})
        Predictor(out)
        return out
```

</details>

<details><summary>fit · L210–372</summary>

```python
def fit(
    x,
    tokens,
    y,
    weights,
    warm,
    variant,
    steps,
    checkpoints,
    device="cpu",
    engine="eager",
    progress=None,
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
                raise ValueError("nonfinite WIDE trajectory")
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
