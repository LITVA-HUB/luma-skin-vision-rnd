# `scripts/chromaseed_refine.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small independent network banks for the prospective ChromaSeed-R study.

SHA-256 исходника: `a055383293111bc52b2672e71e8cd6201396ba98c73260a1d35b64a104a1156c`. Строк: **260**.

## Зависимости

```python
from __future__ import annotations
import math
import weakref
import zlib
import numpy as np
import torch
from skin_local_search_core import ridge_solve
from torch import nn
from torch.nn import functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_refine.py#L15)

```python
FAMILIES = ("stats_mlp", "patch_mlp", "recur_soft", "recur_top16", "recur_dynamic")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `BankAdamW` | — | [L44](../../../../scripts/chromaseed_refine.py#L44) |
| `_Layer` | — | [L85](../../../../scripts/chromaseed_refine.py#L85) |
| `BankNet` | nn.Module | [L134](../../../../scripts/chromaseed_refine.py#L134) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_preprocessor` | FunctionDef | Caller supplies fit rows only. The native-Lab anchor is not an error head. | [L18](../../../../scripts/chromaseed_refine.py#L18) |
| `transform` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_refine.py#L37) |
| `BankAdamW` | ClassDef | AdamW/gradient clipping independently over the leading network dimension.  A single flat parameter bank avoids one CUDA optimizer launch per small layer. Loss must SUM per-network losses, not average them across the bank. | [L44](../../../../scripts/chromaseed_refine.py#L44) |
| `_Layer` | ClassDef | Views are created at each access so no autograd graph survives a step. | [L85](../../../../scripts/chromaseed_refine.py#L85) |
| `gate_weights` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_refine.py#L116) |
| `BankNet` | ClassDef | См. реализацию | [L134](../../../../scripts/chromaseed_refine.py#L134) |
| `apply_exit_policy` | FunctionDef | Offline equivalent of actual inference stopping; native Lab distance. | [L227](../../../../scripts/chromaseed_refine.py#L227) |
| `predict_one` | FunctionDef | Actually execute only required passes, for batch-one CPU deployment timing. | [L244](../../../../scripts/chromaseed_refine.py#L244) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L51–60</summary>

```python
def __init__(self, parameters, lrs, weight_decay=.01, max_norm=5.):
        self.params = list(parameters)
        if not self.params or any(p.shape[0] != len(lrs) for p in self.params):
            raise ValueError("each parameter must have one leading slot per learning rate")
        self.lrs = torch.as_tensor(lrs, dtype=self.params[0].dtype, device=self.params[0].device)
        if torch.any(self.lrs <= 0):
            raise ValueError("learning rates must be positive")
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.weight_decay, self.max_norm, self.t = weight_decay, max_norm, 0
```

</details>

<details><summary>__init__ · L88–90</summary>

```python
def __init__(self, net, start, incoming, outgoing, has_bias=True):
        self._net, self.start, self.incoming, self.outgoing = weakref.ref(net), start, incoming, outgoing
        self.has_bias = has_bias
```

</details>

<details><summary>__call__ · L108–113</summary>

```python
def __call__(self, x):
        shape = x.shape
        out = torch.bmm(x.reshape(shape[0], -1, self.incoming), self.weight)
        if self.has_bias:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.outgoing)
```

</details>

<details><summary>__init__ · L135–167</summary>

```python
def __init__(self, family, seeds):
        super().__init__()
        if family not in FAMILIES or not seeds:
            raise ValueError("invalid model family or empty bank")
        self.family = family
        self.seeds = tuple(seeds)
        self.recurrent = family.startswith("recur_")
        specs = []
        if family != "stats_mlp":
            specs += [("token1", 18, 32), ("token2", 32, 24)]
        if self.recurrent:
            specs += [("context", 60, 48), ("query", 51, 24), ("key", 24, 24),
                      ("update1", 123, 48), ("update2", 48, 48), ("head", 48, 3)]
        elif family == "patch_mlp":
            specs += [("mlp1", 60, 96), ("mlp2", 96, 64), ("mlp3", 64, 48), ("head", 48, 3)]
        else:
            specs += [("mlp1", 36, 96), ("mlp2", 96, 96), ("mlp3", 96, 48), ("head", 48, 3)]
        # A common key bias cancels in soft attention/top-k, leaving only floating
        # point gradient noise. Omit it consistently in all recurrent controls.
        self.theta = nn.Parameter(torch.empty(len(seeds), sum((a + int(name != "key")) * b for name, a, b in specs)))
        self.layers = {}
        offset = 0
        with torch.no_grad():
            for name, a, b in specs:
                self.layers[name] = _Layer(self, offset, a, b, has_bias=name != "key")
                size = (a + int(name != "key")) * b
                for slot, seed in enumerate(seeds):
                    generator = torch.Generator().manual_seed(seed + zlib.crc32(name.encode()))
                    if name == "head":
                        self.theta[slot, offset:offset + size].zero_()
                    else:
                        self.theta[slot, offset:offset + size].uniform_(-1 / math.sqrt(a), 1 / math.sqrt(a), generator=generator)
                offset += size
```

</details>

<details><summary>forward · L201–214</summary>

```python
def forward(self, x, patches, base):
        token, keys, context = self.encode(x, patches)
        if not self.recurrent:
            out = base + self.layers["head"](token).tanh()
            count = torch.full(out.shape[:2] + (1,), 0. if self.family == "stats_mlp" else float(patches.shape[-2]), device=out.device, dtype=out.dtype)
            return out.unsqueeze(-2), count, out.sum((1, 2)) * 0.
        state, prediction = context, base
        outputs, counts, penalties = [], [], []
        for _ in range(4):
            state, prediction, count, penalty = self.refine(state, prediction, token, keys, context)
            outputs.append(prediction)
            counts.append(count)
            penalties.append(penalty)
        return torch.stack(outputs, dim=-2), torch.stack(counts, dim=-1), torch.stack(penalties, dim=-1).mean(-1)
```

</details>
