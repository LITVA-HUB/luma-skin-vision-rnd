# `scripts/cc_v6_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v6_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Canonical-frame correction evidence. Known frame normalization; unverified combination.

SHA-256 исходника: `b5a93339f7d64b4d68af5ae4ee6aab1a871bcb317c532e2b370e9c1df68ae3a4`. Строк: **48**.

## Зависимости

```python
import torch
from cc_v5_model import CorrectionEvidenceNet, _illuminant
from luma_skin_vision.cc.v2 import channel_anchor
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `CanonicalEvidenceNet` | CorrectionEvidenceNet | [L8](../../../../scripts/cc_v6_model.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `CanonicalEvidenceNet` | ClassDef | См. реализацию | [L8](../../../../scripts/cc_v6_model.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–13</summary>

```python
def __init__(self, mode="transport", frame="sog"):
        super().__init__(mode)
        if frame not in {"none", "sog"}:
            raise ValueError("frame must be none or sog")
        self.frame = frame
```

</details>
