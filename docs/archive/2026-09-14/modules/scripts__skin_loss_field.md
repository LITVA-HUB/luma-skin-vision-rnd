# `scripts/skin_loss_field.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_loss_field.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Candidate skin-color loss fields using real training-reference dictionaries.

SHA-256 исходника: `e497415c3ec985e76741d5b5b08a1d46e31988b482bcc834ab4eaf35facda3c4`. Строк: **39**.

## Зависимости

```python
import numpy as np
import torch
from torch import nn
from skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/skin_loss_field.py#L7)

```python
ARMS=['direct','soft_ce','risk_simplex','risk_affine']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `LossFieldImage` | CaptureColor | [L27](../../../../scripts/skin_loss_field.py#L27) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `risk_embedding` | FunctionDef | Full Gram factor; preserves uniform candidate-risk MSE, not a PCA truncation. | [L10](../../../../scripts/skin_loss_field.py#L10) |
| `affine_weights` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_loss_field.py#L23) |
| `LossFieldImage` | ClassDef | См. реализацию | [L27](../../../../scripts/skin_loss_field.py#L27) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L28–30</summary>

```python
def __init__(self,atoms):
        super().__init__('mixture')
        self.atom_head=nn.Linear(256,atoms)
```

</details>

<details><summary>forward · L32–39</summary>

```python
def forward(self,x):
        h=self.local(x);context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        hidden=self.votes[1](self.votes[0](torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1)))
        v=self.votes[2](hidden);weights=v[...,12].softmax(1)
        hypotheses=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        gate=self.gate(context);point=(hypotheses*gate.softmax(1)[...,None]).sum(1)
        atoms=self.atom_head((hidden*weights[...,None]).sum(1))
        return point,gate,hypotheses,atoms
```

</details>
