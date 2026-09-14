# `scripts/skin_nuisance_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_nuisance_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Causal controls for the training-only latent branch; no new inference block.

SHA-256 исходника: `23327fd50287eb52851e6c52c8a9077cf8e4a79470b29bf072012cd4b900e232`. Строк: **39**.

## Зависимости

```python
import torch
from skin_spatial_model import SpatialColor,screened_diffusion
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_nuisance_model.py#L5)

```python
ARMS=['learned','fixed_grid','global','bias']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `NuisanceColor` | SpatialColor | [L18](../../../../scripts/skin_nuisance_model.py#L18) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `control_residual` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_nuisance_model.py#L8) |
| `NuisanceColor` | ClassDef | См. реализацию | [L18](../../../../scripts/skin_nuisance_model.py#L18) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L19–21</summary>

```python
def __init__(self,control):
        if control not in ARMS:raise ValueError(control)
        super().__init__('graph3');self.control=control
```

</details>

<details><summary>forward · L23–33</summary>

```python
def forward(self,x):
        if not self.training:
            previous=self.steps;self.steps=0
            try:return super().forward(x)
            finally:self.steps=previous
        if self.control=='learned':return super().forward(x)
        h=self.local(x);h=h+.5*torch.tanh(self.relation(control_residual(h,self.control,self.adjacency)))
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weight=v[...,3].softmax(1);color=(v[...,:3]*weight[...,None]).sum(1)
        return color,v[...,:3],weight
```

</details>
