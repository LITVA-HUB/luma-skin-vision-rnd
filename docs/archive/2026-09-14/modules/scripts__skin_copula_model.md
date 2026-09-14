# `scripts/skin_copula_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_copula_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Absolute-color patch core plus matched marginal/dependence histogram context.

SHA-256 исходника: `1227d27239ce1f152bb3fbc5f8f029eedec1c9dbdfb78a46f81384cfe5c2d9cd`. Строк: **26**.

## Зависимости

```python
import torch
from torch import nn
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_copula_model.py#L5)

```python
ARMS=['none','rgb_hist','copula','rank_only']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `DistributionColor` | nn.Module | [L8](../../../../scripts/skin_copula_model.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `DistributionColor` | ClassDef | См. реализацию | [L8](../../../../scripts/skin_copula_model.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–16</summary>

```python
def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(896,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        self.distribution=nn.Sequential(nn.Linear(512,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
```

</details>

<details><summary>forward · L18–24</summary>

```python
def forward(self,x):
        h=self.local(x[:,:1152].reshape(-1,64,18))
        distribution=self.distribution(x[:,1152:].clamp_min(0).sqrt())
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0),distribution],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weights=v[...,3].softmax(1);color=(v[...,:3]*weights[...,None]).sum(1)
        return color,v[...,:3],weights
```

</details>
