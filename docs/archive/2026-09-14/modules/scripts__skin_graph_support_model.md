# `scripts/skin_graph_support_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_graph_support_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Graph receives original images only; shared core also learns observed bags.

SHA-256 исходника: `f6eee31c6937d2a6a8791a73107ec0b66f18ead33086a5b1f63b140b63112f3f`. Строк: **18**.

## Зависимости

```python
from skin_spatial_model import SpatialColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 4](../../../../scripts/skin_graph_support_model.py#L4)

```python
ARMS=('plain_raw','plain_paired','graph_raw','graph_paired')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `GraphSupportColor` | SpatialColor | [L7](../../../../scripts/skin_graph_support_model.py#L7) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `GraphSupportColor` | ClassDef | См. реализацию | [L7](../../../../scripts/skin_graph_support_model.py#L7) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L8–9</summary>

```python
def __init__(self):
        super().__init__('graph3')
```

</details>

<details><summary>forward · L11–15</summary>

```python
def forward(self,x,branch=False):
        previous=self.steps
        self.steps=3 if self.training and branch else 0
        try:return super().forward(x)
        finally:self.steps=previous
```

</details>
