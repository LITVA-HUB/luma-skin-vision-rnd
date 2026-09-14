# `scripts/skin_material_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched image color heads with a fixed measured-material decoder.

SHA-256 исходника: `5272f541db995025094b53084f3c64030e5b6692fc903057f65e6370992f10fc`. Строк: **52**.

## Зависимости

```python
import torch
from torch import nn
from pathlib import Path
import sys
from scripts.skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_material_model.py#L9)

```python
ARMS=['direct','tangent','material','tangent_residual','material_residual']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `MaterialImage` | CaptureColor | [L12](../../../../scripts/skin_material_model.py#L12) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `MaterialImage` | ClassDef | См. реализацию | [L12](../../../../scripts/skin_material_model.py#L12) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L13–24</summary>

```python
def __init__(self,arm,prior,target_mean,target_std):
        super().__init__('mixture')
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        old=self.votes[-1];self.votes[-1]=nn.Linear(256,45)
        with torch.no_grad():
            self.votes[-1].weight[32:].copy_(old.weight)
            self.votes[-1].bias[32:].copy_(old.bias)
        for name in ('mu','basis','matrix','white','base','jacobian'):
            self.register_buffer(name,torch.as_tensor(prior[name],dtype=torch.float32).clone())
        self.register_buffer('target_mean',torch.as_tensor(target_mean,dtype=torch.float32).clone())
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32).clone())
```

</details>

<details><summary>forward · L38–52</summary>

```python
def forward(self,x):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        w=v[...,44].softmax(1)
        z=(v[...,:32].reshape(len(x),h.shape[1],4,8)*w[:,:,None,None]).sum(1)
        free=(v[...,32:44].reshape(len(x),h.shape[1],4,3)*w[:,:,None,None]).sum(1)
        if self.arm=='direct':hypotheses=free
        else:
            native=self.base+z@self.jacobian if self.arm.startswith('tangent') else self.material_decode(z)
            hypotheses=(native-self.target_mean)/self.target_std
            if self.arm.endswith('_residual'):hypotheses=hypotheses+free
        logits=self.gate(context);gate=logits.softmax(1)
        prediction=(hypotheses*gate[...,None]).sum(1)
        return prediction,logits,hypotheses,free
```

</details>
