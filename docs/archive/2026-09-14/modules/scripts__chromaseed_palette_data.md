# `scripts/chromaseed_palette_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Spatial measured patches and fixed photometric simulations for encoder pretraining.

SHA-256 исходника: `5a952cff61131b8edb93f3485617d47846b406d346d5f789d389aa4d3837d467`. Строк: **188**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/chromaseed_palette_data.py#L13)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/chromaseed_palette_data.py#L14)

```python
OUT = Path('D:/Luma-RnD/chromaseed_palette_pretrain_v1')
```

[Строка 15](../../../../scripts/chromaseed_palette_data.py#L15)

```python
P1 = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_palette_v1')
```

[Строка 16](../../../../scripts/chromaseed_palette_data.py#L16)

```python
P1_PALETTE_SHA = '9e5bb25250258fb79a547b3ac6dc93792d1808967fe5842e21b42f57a6cc2112'
```

[Строка 17](../../../../scripts/chromaseed_palette_data.py#L17)

```python
PROTOCOL = ROOT / 'docs/research/chromaseed_palette_pretrain_v1_protocol.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_palette_data.py#L20) |
| `read` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_palette_data.py#L25) |
| `save` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_palette_data.py#L29) |
| `patch_tokens` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_palette_data.py#L35) |
| `spatial_centres` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_palette_data.py#L46) |
| `view_parameters` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_palette_data.py#L60) |
| `rendered_views` | FunctionDef | См. реализацию | [L71](../../../../scripts/chromaseed_palette_data.py#L71) |
| `freeze` | FunctionDef | См. реализацию | [L82](../../../../scripts/chromaseed_palette_data.py#L82) |
| `verify` | FunctionDef | См. реализацию | [L108](../../../../scripts/chromaseed_palette_data.py#L108) |
| `prepare` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_palette_data.py#L116) |
