# `scripts/skin_pair_invariance.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_pair_invariance.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Single-image skin models with TRAIN-only paired-capture regularization.

SHA-256 исходника: `4cceedf45de2a04c03d6aab7a0c408d851c78c8f4525bf0bc0a2552fea49cf7e`. Строк: **52**.

## Зависимости

```python
import numpy as np
import torch
from torch import nn
from skin_mskcc_vote import PatchVotes
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/skin_pair_invariance.py#L7)

```python
ARMS=['raw','standardized','quotient3','output','vicreg']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PairedColor` | nn.Module | [L45](../../../../scripts/skin_pair_invariance.py#L45) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pair_indices` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_pair_invariance.py#L10) |
| `nuisance_transform` | FunctionDef | Within-site covariance on mean token channels; never uses camera labels. | [L19](../../../../scripts/skin_pair_invariance.py#L19) |
| `vicreg_loss` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_pair_invariance.py#L35) |
| `PairedColor` | ClassDef | См. реализацию | [L45](../../../../scripts/skin_pair_invariance.py#L45) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L46–48</summary>

```python
def __init__(self,target_std,arm):
        super().__init__();self.backbone=PatchVotes(target_std,0)
        self.projection=nn.Linear(512,64) if arm=='vicreg' else None
```

</details>

<details><summary>forward · L50–52</summary>

```python
def forward(self,x,features=False):
        y,_,context=self.backbone(x,details=True)
        return (y,context) if features else y
```

</details>
