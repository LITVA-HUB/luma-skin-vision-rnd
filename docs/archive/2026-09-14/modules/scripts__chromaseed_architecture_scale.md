# `scripts/chromaseed_architecture_scale.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched R-style small/large residual architectures; immutable NP warm anchor.

SHA-256 исходника: `ad76f5b04f65536df1768a7c60c1051020547a77cc6776b244417e53569a9047`. Строк: **428**.

## Зависимости

```python
from __future__ import annotations
import math
import time
import zlib
import numpy as np
import torch
from chromaseed_patch8_fit import array_hash, token_normalizers
from chromaseed_refine import BankAdamW, _Layer
from chromaseed_refine_train import setup
from scipy.special import expit
from skin_local_search_train import synchronize
from torch import nn
from torch.nn import functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_architecture_scale.py#L19)

```python
VARIANTS = (
    "patch_small",
    "patch5m",
    "soft_small",
    "soft5m",
    "dynamic_small",
    "dynamic5m",
    "pool5m",
)
```

[Строка 28](../../../../scripts/chromaseed_architecture_scale.py#L28)

```python
SEEDS = (17, 29, 43)
```

[Строка 29](../../../../scripts/chromaseed_architecture_scale.py#L29)

```python
RATES = (0.00001, 0.0001)
```

[Строка 30](../../../../scripts/chromaseed_architecture_scale.py#L30)

```python
SLOTS = tuple((seed, lr) for seed in SEEDS for lr in RATES)
```

[Строка 31](../../../../scripts/chromaseed_architecture_scale.py#L31)

```python
HORIZON = 8192
```

[Строка 32](../../../../scripts/chromaseed_architecture_scale.py#L32)

```python
BATCH = 64
```

[Строка 33](../../../../scripts/chromaseed_architecture_scale.py#L33)

```python
_STREAM = None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Bank` | nn.Module | [L80](../../../../scripts/chromaseed_architecture_scale.py#L80) |
| `Predictor` | — | [L152](../../../../scripts/chromaseed_architecture_scale.py#L152) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `specs` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_architecture_scale.py#L36) |
| `capacity` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_architecture_scale.py#L57) |
| `gate` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_architecture_scale.py#L61) |
| `Bank` | ClassDef | См. реализацию | [L80](../../../../scripts/chromaseed_architecture_scale.py#L80) |
| `base_model` | FunctionDef | См. реализацию | [L148](../../../../scripts/chromaseed_architecture_scale.py#L148) |
| `Predictor` | ClassDef | Independent FP64 NumPy implementation of the exported single-model consumer. | [L152](../../../../scripts/chromaseed_architecture_scale.py#L152) |
| `predict` | FunctionDef | См. реализацию | [L225](../../../../scripts/chromaseed_architecture_scale.py#L225) |
| `predict_torch` | FunctionDef | См. реализацию | [L233](../../../../scripts/chromaseed_architecture_scale.py#L233) |
| `sampling` | FunctionDef | См. реализацию | [L255](../../../../scripts/chromaseed_architecture_scale.py#L255) |
| `fit` | FunctionDef | См. реализацию | [L267](../../../../scripts/chromaseed_architecture_scale.py#L267) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>specs · L36–54</summary>

```python
def specs(variant):
    if variant not in VARIANTS:
        raise ValueError("unregistered architecture")
    if variant == "pool5m":
        return [("token1", 18, 512), ("mlp1", 1572, 3072), ("head", 3072, 3)]
    small = variant.endswith("small")
    a, e, h = (32, 24, 48) if small else (384, 256, 1120)
    out = [("token1", 18, a), ("token2", a, e)]
    if variant.startswith("patch"):
        h1, h2, h3 = (96, 64, 48) if small else (1792, 1536, 1024)
        return out + [("mlp1", 36 + e, h1), ("mlp2", h1, h2), ("mlp3", h2, h3), ("head", h3, 3)]
    return out + [
        ("context", 36 + e, h),
        ("query", h + 3, e),
        ("key", e, e),
        ("update1", 2 * h + e + 3, h),
        ("update2", h, h),
        ("head", h, 3),
    ]
```

</details>

<details><summary>capacity · L57–58</summary>

```python
def capacity(variant):
    return 643 + sum((a + (name != "key")) * b for name, a, b in specs(variant))
```

</details>

<details><summary>gate · L61–77</summary>

```python
def gate(scores, family, training):
    probability = scores.sigmoid()
    if family == "soft":
        mask = torch.ones_like(scores)
    elif family == "dynamic":
        fallback = torch.zeros_like(scores).scatter(
            -1, scores.topk(min(4, scores.shape[-1]), dim=-1).indices, 1.0
        )
        hard = torch.maximum(fallback, (scores > 0).to(scores.dtype))
        mask = hard + probability - probability.detach() if training else hard
    else:
        raise ValueError("unknown attention family")
    unnormalized = (scores - scores.amax(-1, keepdim=True)).exp() * mask
    attention = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(1e-12)
    count = (mask.detach() > 0.5).sum(-1)
    penalty = probability.mean((1, 2)) if family == "dynamic" else probability.sum((1, 2)) * 0.0
    return attention, count, penalty
```

</details>

<details><summary>__init__ · L81–101</summary>

```python
def __init__(self, variant, seeds=None):
        super().__init__()
        self.variant = variant
        self.recurrent = variant.startswith(("soft", "dynamic"))
        self.seeds = tuple(s[0] for s in SLOTS) if seeds is None else tuple(seeds)
        self.theta = nn.Parameter(torch.empty(len(self.seeds), capacity(variant) - 643))
        self.layers = {}
        offset = 0
        with torch.no_grad():
            for name, a, b in specs(variant):
                self.layers[name] = _Layer(self, offset, a, b, has_bias=name != "key")
                size = (a + (name != "key")) * b
                for i, seed in enumerate(self.seeds):
                    if name == "head":
                        self.theta[i, offset : offset + size].zero_()
                    else:
                        g = torch.Generator().manual_seed(seed + zlib.crc32(name.encode()))
                        self.theta[i, offset : offset + size].uniform_(
                            -1 / math.sqrt(a), 1 / math.sqrt(a), generator=g
                        )
                offset += size
```

</details>

<details><summary>forward · L103–135</summary>

```python
def forward(self, x, tokens, base):
        if self.variant == "pool5m":
            t = self.layers["token1"](tokens).relu()
            mean = t.mean(-2)
            spread = ((t - mean.unsqueeze(-2)).square().mean(-2) + 1e-6).sqrt()
            h = self.layers["mlp1"](torch.cat((x, mean, spread, t.amax(-2)), dim=-1)).relu()
            pred = base + self.layers["head"](h).tanh()
            return pred.unsqueeze(-2), pred.sum((1, 2)) * 0.0
        t = F.silu(self.layers["token2"](F.silu(self.layers["token1"](tokens))))
        context_input = torch.cat((x, t.mean(-2)), dim=-1)
        if not self.recurrent:
            h = context_input
            for name in ("mlp1", "mlp2", "mlp3"):
                h = F.silu(self.layers[name](h))
            pred = base + self.layers["head"](h).tanh()
            return pred.unsqueeze(-2), pred.sum((1, 2)) * 0.0
        context = F.silu(self.layers["context"](context_input))
        keys = self.layers["key"](t)
        state, pred = context, base
        outputs, penalties = [], []
        family = "dynamic" if self.variant.startswith("dynamic") else "soft"
        for _ in range(4):
            query = self.layers["query"](torch.cat((state, pred), dim=-1))
            scores = (keys * query.unsqueeze(-2)).sum(-1) / math.sqrt(t.shape[-1])
            attention, _, penalty = gate(scores, family, self.training)
            pooled = (t * attention.unsqueeze(-1)).sum(-2)
            u = torch.cat((state, context, pooled, pred), dim=-1)
            update = self.layers["update2"](F.silu(self.layers["update1"](u))).tanh()
            state = 0.5 * state + 0.5 * update
            pred = pred + 0.25 * self.layers["head"](state).tanh()
            outputs.append(pred)
            penalties.append(penalty)
        return torch.stack(outputs, -2), torch.stack(penalties, -1).mean(-1)
```

</details>

<details><summary>export · L137–145</summary>

```python
def export(self, slot, warm, tokens):
        prep = token_normalizers(tokens)
        base = warm[slot // 2] if len(warm) == 3 else warm[slot]
        return dict(
            variant=np.asarray(self.variant),
            theta=self.theta[slot].detach().cpu().numpy().copy(),
            **prep,
            **{"base_" + k: v.copy() for k, v in base.items()},
        )
```

</details>

<details><summary>__init__ · L155–168</summary>

```python
def __init__(self, model):
        self.model = model
        self.variant = str(model["variant"])
        self.base = base_model(model)
        self.layers = {}
        theta = model["theta"].astype(np.float64)
        offset = 0
        for name, a, b in specs(self.variant):
            weight = theta[offset : offset + a * b].reshape(a, b)
            offset += a * b
            bias = np.zeros(b) if name == "key" else theta[offset : offset + b]
            offset += 0 if name == "key" else b
            self.layers[name] = weight, bias
        assert offset == len(theta)
```

</details>

<details><summary>__call__ · L170–222</summary>

```python
def __call__(self, x, tokens, *, all_passes=False):
        single = np.ndim(x) == 1
        x = np.asarray(x, np.float32).reshape(-1, 36)
        tokens = np.asarray(tokens, np.float32).reshape(-1, 64, 18)
        b = self.base
        xx = ((x - b["x_mean"]) / b["x_std"]).astype(np.float64)
        tt = ((tokens - self.model["t_mean"]) / self.model["t_std"]).astype(np.float64)
        base = np.maximum(xx @ b["w0"].astype(float) + b["b0"], 0) @ b["v0"].astype(float) + b["c0"]

        def layer(name, v):
            w, bias = self.layers[name]
            return v @ w + bias

        def silu(v):
            return v * expit(v)

        if self.variant == "pool5m":
            t = np.maximum(layer("token1", tt), 0)
            mean = t.mean(1)
            spread = np.sqrt(((t - mean[:, None]) ** 2).mean(1) + 1e-6)
            h = np.maximum(layer("mlp1", np.concatenate((xx, mean, spread, t.max(1)), -1)), 0)
            outputs = (base + np.tanh(layer("head", h)))[:, None]
        else:
            t = silu(layer("token2", silu(layer("token1", tt))))
            ci = np.concatenate((xx, t.mean(1)), -1)
            if self.variant.startswith("patch"):
                h = ci
                for name in ("mlp1", "mlp2", "mlp3"):
                    h = silu(layer(name, h))
                outputs = (base + np.tanh(layer("head", h)))[:, None]
            else:
                context = silu(layer("context", ci))
                keys = layer("key", t)
                state, pred, outputs = context, base, []
                for _ in range(4):
                    q = layer("query", np.concatenate((state, pred), -1))
                    scores = np.sum(keys * q[:, None], -1) / math.sqrt(t.shape[-1])
                    mask = np.ones_like(scores)
                    if self.variant.startswith("dynamic"):
                        mask = (scores > 0).astype(float)
                        top = np.argsort(scores, axis=-1, kind="stable")[:, -4:]
                        np.put_along_axis(mask, top, 1.0, axis=-1)
                    a = np.exp(scores - scores.max(-1, keepdims=True)) * mask
                    a /= a.sum(-1, keepdims=True)
                    pooled = np.sum(t * a[..., None], 1)
                    u = np.concatenate((state, context, pooled, pred), -1)
                    state = 0.5 * state + 0.5 * np.tanh(layer("update2", silu(layer("update1", u))))
                    pred = pred + 0.25 * np.tanh(layer("head", state))
                    outputs.append(pred)
                outputs = np.stack(outputs, 1)
        outputs = outputs * b["y_std"].astype(float) + b["y_mean"].astype(float)
        result = outputs if all_passes else outputs[:, -1]
        return result[0] if single else result
```

</details>

<details><summary>predict · L225–229</summary>

```python
def predict(model, x, tokens):
    consumer = Predictor(model)
    return np.concatenate(
        [consumer(x[i : i + 16], tokens[i : i + 16]) for i in range(0, len(x), 16)]
    )
```

</details>

<details><summary>fit · L267–428</summary>

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
        or checkpoints[-1] != steps
        or min(checkpoints) < 1
    ):
        raise ValueError("positive ordered checkpoints ending at steps within fixed horizon")
    if engine not in ("eager", "cuda_graph") or (
        engine == "cuda_graph" and not str(device).startswith("cuda")
    ):
        raise ValueError("invalid engine/device")
    setup(device)
    synchronize(device)
    start = time.perf_counter()
    x, tokens, y = np.asarray(x, np.float32), np.asarray(tokens, np.float32), np.asarray(y, float)
    if (
        x.shape != (len(y), 36)
        or tokens.shape != (len(y), 64, 18)
        or y.shape != (len(x), 3)
        or not all(np.isfinite(v).all() for v in (x, tokens, y))
    ):
        raise ValueError("matching finite fit-only arrays required")
    if len(warm) != 3:
        raise ValueError("three matching warm parents required")
    for b in warm:
        for name, v in (("x", x.astype(float)), ("y", y)):
            np.testing.assert_array_equal(b[name + "_mean"], v.mean(0).astype(np.float32))
            np.testing.assert_array_equal(
                b[name + "_std"], np.maximum(v.std(0), 1e-6).astype(np.float32)
            )
    prep, b = token_normalizers(tokens), warm[0]
    xt = torch.as_tensor((x - b["x_mean"]) / b["x_std"], device=device)
    tt = torch.as_tensor((tokens - prep["t_mean"]) / prep["t_std"], device=device)
    yt = torch.as_tensor(((y - b["y_mean"]) / b["y_std"]).astype(np.float32), device=device)
    base = []
    for b in warm:
        w, bias, v, c = [torch.as_tensor(b[k], device=device) for k in ("w0", "b0", "v0", "c0")]
        base.append((xt @ w + bias).relu() @ v + c)
    base = torch.stack(base)
    net = Bank(variant).to(device)
    cpu_indices = sampling(weights, steps)
    indices = torch.as_tensor(cpu_indices, device=device)
    slot_seed = torch.as_tensor([0, 0, 1, 1, 2, 2], device=device)
    rates = np.asarray([s[1] for s in SLOTS], np.float32)
    optimizer = BankAdamW(net.parameters(), rates, weight_decay=0.01, max_norm=5.0)
    base_rates = torch.as_tensor(rates, device=device).clone()
    factors = torch.as_tensor(
        (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * np.arange(steps) / HORIZON))).astype(np.float32),
        device=device,
    )
    corrections = torch.as_tensor(
        np.asarray([[1 - 0.9**i, np.sqrt(1 - 0.999**i)] for i in range(1, steps + 1)], np.float32),
        device=device,
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def iteration(index, factor, correction):
        net.zero_grad(set_to_none=True)
        base_batch = base.gather(1, index[..., None].expand(-1, -1, 3)).index_select(0, slot_seed)
        ids = index.index_select(0, slot_seed)
        out, penalty = net(xt[ids], tt[ids], base_batch)
        losses = (out - yt[ids].unsqueeze(-2)).square().mean((1, 3))
        loss = 0.6 * losses.mean(-1) + 0.4 * losses[:, -1] + 0.001 * penalty
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
                optimizer.lrs.copy_(base_rates)
                optimizer.t = 0
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
    models, trace = {}, []
    for step in range(1, steps + 1):
        if graph is None:
            loss = iteration(indices[:, step - 1], factors[step - 1], corrections[step - 1])
        else:
            graph.replay()
        if step in checkpoints or step % 128 == 0:
            synchronize(device)
            value = loss.detach().cpu().numpy()
            if not np.isfinite(value).all() or not torch.isfinite(net.theta).all():
                raise ValueError("nonfinite AS trajectory")
            item = dict(
                step=step, seconds=time.perf_counter() - start, minibatch_loss=value.tolist()
            )
            trace.append(item)
            if progress:
                progress(item)
        if step in checkpoints:
            models[step] = [net.export(i, warm, tokens) for i in range(6)]
    synchronize(device)
    return models, dict(
        variant=variant,
        steps=steps,
        checkpoints=list(checkpoints),
        horizon=HORIZON,
        batch_size=BATCH,
        slots=list(SLOTS),
        sampling_sha256=array_hash(cpu_indices),
        fit_tokens_sha256=array_hash(tokens),
        setup_seconds=setup_seconds,
        full_bank_seconds=time.perf_counter() - start,
        trace=trace,
        final_learning_rates=optimizer.lrs.detach().cpu().numpy().tolist(),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if str(device).startswith("cuda")
        else None,
        device=device,
        engine=engine,
        trainable_parameters=capacity(variant) - 643,
        deployed_parameters=capacity(variant),
    )
```

</details>
