# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/drop_path.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/drop_path.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a`. Строк: **34**.

## Зависимости

```python
from torch import nn
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `DropPath` | nn.Module | [L26](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/drop_path.py#L26) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `drop_path` | FunctionDef | См. реализацию | [L14](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/drop_path.py#L14) |
| `DropPath` | ClassDef | Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). | [L26](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/drop_path.py#L26) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L29–31</summary>

```python
def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
```

</details>

<details><summary>forward · L33–34</summary>

```python
def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

</details>
