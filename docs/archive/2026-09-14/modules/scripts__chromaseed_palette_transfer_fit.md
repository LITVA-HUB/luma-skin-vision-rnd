# `scripts/chromaseed_palette_transfer_fit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer_fit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

P3 fitter derived from frozen HR; only the registered initializer is replaced.

No frozen source is edited or monkeypatched. Original-arm trajectories must
reproduce HR bitwise; transferred encoders remain trainable at native fine-tuning.

SHA-256 исходника: `105b2f6f16e31cfa9803ed3a8bc37bd10cd8cd83fd83bd31e05dc96e3c3d17a4`. Строк: **199**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
import torch
from chromaseed_architecture_scale import BATCH, HORIZON, SLOTS, capacity, sampling
from chromaseed_head_range import validate_mode
from chromaseed_palette_encoder import ENCODER_PARAMETERS
from chromaseed_palette_transfer import Bank, validate_encoders
from chromaseed_patch8_fit import array_hash, token_normalizers
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import setup
from skin_local_search_train import synchronize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_palette_transfer_fit.py#L22)

```python
_STREAM = None
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_palette_transfer_fit.py#L25) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L25–199</summary>

```python
def fit(
    x,
    tokens,
    y,
    weights,
    warm,
    variant,
    mode,
    arm,
    encoders,
    steps,
    checkpoints,
    device="cpu",
    engine="eager",
    progress=None,
):
    global _STREAM
    validate_mode(mode)
    validate_encoders(encoders, arm)
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
    net = Bank(variant, mode, arm, encoders, prep["t_mean"], prep["t_std"])
    initial_theta_sha256 = array_hash(net.theta.detach().numpy())
    encoder_digests = list(net.encoder_digests)
    net = net.to(device)
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
                raise ValueError("nonfinite HR trajectory")
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
        head_mode=mode,
        initialization=arm,
        initial_theta_sha256=initial_theta_sha256,
        encoder_digests=encoder_digests,
        transferred_parameters=0 if arm == "original" else ENCODER_PARAMETERS,
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
