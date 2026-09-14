# `scripts/skin_train_branch_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_train_branch_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Spatial branch used only during fitting; deployment always uses the plain core.

SHA-256 исходника: `5fa94f3c7954ce9537819e1c97340255ee5b80a87ee1ade5f516f18e7c7667c7`. Строк: **26**.

## Зависимости

```python
import torch
from skin_spatial_model import SpatialColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_train_branch_model.py#L5)

```python
ARMS=['graph_always','conv_always','graph_drop','conv_drop']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `TrainingBranchColor` | SpatialColor | [L8](../../../../scripts/skin_train_branch_model.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `TrainingBranchColor` | ClassDef | См. реализацию | [L8](../../../../scripts/skin_train_branch_model.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–12</summary>

```python
def __init__(self,strategy):
        if strategy not in ARMS:raise ValueError(strategy)
        super().__init__(strategy.split('_')[0]+'3')
        self.strategy=strategy
```

</details>

<details><summary>forward · L14–21</summary>

```python
def forward(self,x):
        previous=self.steps
        use=self.training
        if use and self.strategy.endswith('_drop'):
            use=bool(torch.rand(())>=.5)
        self.steps=3 if use else 0
        try:return super().forward(x)
        finally:self.steps=previous
```

</details>
