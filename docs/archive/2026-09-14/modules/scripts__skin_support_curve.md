# `scripts/skin_support_curve.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_support_curve.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-person support and learned patch-information falsifier.

SHA-256 исходника: `96e3eddd1f1281d0695bcd42b91730dcd0e5f1408a1ce6a741e7c87e5c76757c`. Строк: **45**.

## Зависимости

```python
import numpy as np
import torch
from torch import nn
from skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/skin_support_curve.py#L7)

```python
ARMS=('baseline','statistics','pixels')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `SkinRepresentation` | nn.Module | [L30](../../../../scripts/skin_support_curve.py#L30) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `patient_roles` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_support_curve.py#L10) |
| `pixel_patches` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_support_curve.py#L25) |
| `SkinRepresentation` | ClassDef | См. реализацию | [L30](../../../../scripts/skin_support_curve.py#L30) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L31–40</summary>

```python
def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown representation')
        self.arm=arm;self.core=CaptureColor('mixture')
        if arm=='statistics':
            self.adapter=nn.Sequential(nn.Linear(18,64),nn.SiLU(),nn.Linear(64,120),nn.SiLU(),nn.Linear(120,18))
        elif arm=='pixels':
            self.adapter=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.SiLU(),nn.Conv2d(16,24,3,2,1),nn.SiLU(),
                nn.Conv2d(24,32,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,18))
        if arm!='baseline':nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
```

</details>

<details><summary>forward · L42–45</summary>

```python
def forward(self,tokens,rgb):
        if self.arm=='statistics':tokens=tokens+self.adapter(tokens)
        elif self.arm=='pixels':tokens=tokens+self.adapter(pixel_patches(rgb)).reshape(len(tokens),64,18)
        return self.core(tokens)
```

</details>
