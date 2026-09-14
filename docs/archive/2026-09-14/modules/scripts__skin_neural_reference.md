# `scripts/skin_neural_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_neural_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Capacity-matched residual and differentiable local-reference color adapters.

SHA-256 исходника: `6305a09d725ec45672ab5b90191f1a74abb4d637e7249f0db128abb123cd7ed4`. Строк: **72**.

## Зависимости

```python
import torch
from torch import nn
from skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_neural_reference.py#L6)

```python
ARMS=('residual','mean','affine')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ColorAdapter` | nn.Module | [L37](../../../../scripts/skin_neural_reference.py#L37) |
| `DeployedColor` | nn.Module | [L57](../../../../scripts/skin_neural_reference.py#L57) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `core_features` | FunctionDef | См. реализацию | [L9](../../../../scripts/skin_neural_reference.py#L9) |
| `reference_correction` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_neural_reference.py#L18) |
| `ColorAdapter` | ClassDef | См. реализацию | [L37](../../../../scripts/skin_neural_reference.py#L37) |
| `DeployedColor` | ClassDef | Prepared features in; native Lab out. Only fixed training memory is stored. | [L57](../../../../scripts/skin_neural_reference.py#L57) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L38–44</summary>

```python
def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown color adapter')
        self.arm=arm
        self.features=nn.Sequential(nn.Linear(551,256),nn.SiLU(),nn.Linear(256,192),nn.SiLU(),nn.Linear(192,16),nn.Tanh())
        self.output=nn.Linear(16,3)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
```

</details>

<details><summary>forward · L51–54</summary>

```python
def forward(self,x,bank,bank_residual,allowed):
        z=self.features(x)
        bank_z=None if self.arm=='residual' else self.features(bank)
        return self.from_embedding(z,bank_z,bank_residual,allowed)
```

</details>

<details><summary>__init__ · L59–64</summary>

```python
def __init__(self,arm,bank_size):
        super().__init__();self.core=CaptureColor('mixture');self.adapter=ColorAdapter(arm)
        for name,size in [('feature_mean',548),('feature_std',548),('target_mean',3),('target_std',3)]:
            self.register_buffer(name,torch.zeros(size))
        self.register_buffer('bank_z',torch.zeros(bank_size,16))
        self.register_buffer('bank_residual',torch.zeros(bank_size,3))
```

</details>

<details><summary>forward · L66–72</summary>

```python
def forward(self,tokens,color):
        p,c=core_features(self.core,tokens)
        x=torch.cat([(torch.cat([c,color],1)-self.feature_mean)/self.feature_std,p],1)
        z=self.adapter.features(x)
        allowed=torch.ones(len(x),len(self.bank_z),dtype=torch.bool,device=x.device)
        correction=self.adapter.from_embedding(z,self.bank_z,self.bank_residual,allowed)
        return (p+correction)*self.target_std+self.target_mean
```

</details>
