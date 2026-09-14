# `scripts/skin_crossfit_correction.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_crossfit_correction.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Stable-color stacked correction with subject-excluded encoder controls.

SHA-256 исходника: `b4e2f5ec2c61ce671df15e200377efc2b08985b63ac5be662c146886ed25ce75`. Строк: **52**.

## Зависимости

```python
import numpy as np
import torch
from torch import nn
from skin_support_curve import SkinRepresentation
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/skin_crossfit_correction.py#L7)

```python
ARMS=('in_full','in_matched','out_person')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `StableHead` | nn.Module | [L32](../../../../scripts/skin_crossfit_correction.py#L32) |
| `StableColorModel` | nn.Module | [L41](../../../../scripts/skin_crossfit_correction.py#L41) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `inner_folds` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_crossfit_correction.py#L10) |
| `choose_predictions` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_crossfit_correction.py#L22) |
| `features` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_crossfit_correction.py#L28) |
| `StableHead` | ClassDef | См. реализацию | [L32](../../../../scripts/skin_crossfit_correction.py#L32) |
| `StableColorModel` | ClassDef | One prepared image, one core and one correction head; no reference bank. | [L41](../../../../scripts/skin_crossfit_correction.py#L41) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L33–36</summary>

```python
def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(39,384),nn.SiLU(),nn.Linear(384,384),nn.SiLU(),nn.Linear(384,64),nn.SiLU(),nn.Linear(64,3),nn.Tanh())
        nn.init.zeros_(self.net[-2].weight);nn.init.zeros_(self.net[-2].bias)
```

</details>

<details><summary>forward · L38–38</summary>

```python
def forward(self,x):return self.net(x)
```

</details>

<details><summary>__init__ · L43–46</summary>

```python
def __init__(self):
        super().__init__();self.base=SkinRepresentation('baseline');self.head=StableHead()
        for name,size in [('color_mean',36),('color_std',36),('target_mean',3),('target_std',3)]:
            self.register_buffer(name,torch.zeros(size))
```

</details>

<details><summary>forward · L48–52</summary>

```python
def forward(self,tokens,color):
        p=self.base(tokens,None)[0]
        native=p*self.target_std+self.target_mean
        x=torch.cat([(color-self.color_mean)/self.color_std,(native-self.target_mean)/self.target_std],1)
        return native+self.head(x)*self.target_std
```

</details>
