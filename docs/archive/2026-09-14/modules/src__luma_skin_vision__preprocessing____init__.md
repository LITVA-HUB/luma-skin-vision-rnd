# `src/luma_skin_vision/preprocessing/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/preprocessing/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `681a7b4142d5149baf545974e775ec847bbcaf7c53ef9384b0b8817076b8450b`. Строк: **70**.

## Зависимости

```python
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms, ImageOps
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../src/luma_skin_vision/preprocessing/__init__.py#L7)

```python
VERSION = "srgb-exif-v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `decode_image` | FunctionDef | См. реализацию | [L10](../../../../src/luma_skin_vision/preprocessing/__init__.py#L10) |
| `quality_gate` | FunctionDef | Provisional acquisition thresholds, NOT calibrated error/quality probabilities. | [L35](../../../../src/luma_skin_vision/preprocessing/__init__.py#L35) |
| `region_tensor` | FunctionDef | См. реализацию | [L56](../../../../src/luma_skin_vision/preprocessing/__init__.py#L56) |
