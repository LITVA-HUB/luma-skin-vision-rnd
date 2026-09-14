# `scripts/skin_local_teacher_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_teacher_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Identical RGB mixture heads with aligned/global/shuffled frozen descriptors.

SHA-256 исходника: `0b7b2be3924a4cf46a05138654b2fafbed8d169b26f6ae49b06f6d407637f348`. Строк: **34**.

## Зависимости

```python
import torch
from torch import nn
from skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_local_teacher_model.py#L6)

```python
ARMS = ['plain', 'aligned', 'global', 'shuffled']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `LocalTeacherColor` | CaptureColor | [L9](../../../../scripts/skin_local_teacher_model.py#L9) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `LocalTeacherColor` | ClassDef | См. реализацию | [L9](../../../../scripts/skin_local_teacher_model.py#L9) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L10–15</summary>

```python
def __init__(self, arm):
        if arm not in ARMS: raise ValueError('Unknown local teacher arm')
        super().__init__('mixture'); self.arm = arm
        self.adapter = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 256))
        self.register_buffer('teacher_mean', torch.zeros(384))
        self.register_buffer('teacher_std', torch.ones(384))
```

</details>

<details><summary>forward · L26–34</summary>

```python
def forward(self, x):
        h = self.local(x[..., :18].contiguous())
        if self.arm != 'plain': h = h+.5*torch.tanh(self.adapter(self.teacher_tokens(x)))
        context = self.context(torch.cat([h.mean(1), h.amax(1), h.std(1, correction=0)], 1))
        v = self.votes(torch.cat([h, context[:, None].expand(-1, h.shape[1], -1)], -1))
        weights = v[..., 12].softmax(1)
        hypotheses = (v[..., :12].reshape(len(x), h.shape[1], 4, 3)*weights[:, :, None, None]).sum(1)
        logits = self.gate(context); gate = logits.softmax(1)
        return (hypotheses*gate[..., None]).sum(1), logits, hypotheses
```

</details>
