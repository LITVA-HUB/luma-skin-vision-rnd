# `src/luma_skin_vision/face/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/face/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Optional user-supplied YuNet ONNX adapter. No weights downloaded implicitly.

SHA-256 исходника: `3eda1d5dde0c90f902336706a1489025a31cabe680567bc4ee41373ebff12c6c`. Строк: **29**.

## Зависимости

```python
from pathlib import Path
import numpy as np
from luma_skin_vision.data import sha256
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `YuNet` | — | [L10](../../../../src/luma_skin_vision/face/__init__.py#L10) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `YuNet` | ClassDef | См. реализацию | [L10](../../../../src/luma_skin_vision/face/__init__.py#L10) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L11–16</summary>

```python
def __init__(self, model_path, expected_sha256):
        import cv2

        if sha256(model_path) != expected_sha256:
            raise ValueError("detector checksum mismatch")
        self.detector = cv2.FaceDetectorYN.create(str(Path(model_path)), "", (320, 320), 0.9)
```

</details>
