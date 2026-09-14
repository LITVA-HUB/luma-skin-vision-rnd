# `scripts/skin_face_transfer_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_transfer_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Role-safe Seg2 sources and matched sampling; no model or CUDA initialization.

SHA-256 исходника: `b5932e3c4a8b6c78c5c0202094148135039e3c21cdaa56b358b211041c6c25e7`. Строк: **218**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/skin_face_transfer_data.py#L11)

```python
DATA_ROOT = Path('D:/Luma-RnD/data_growth_2026_09_14')
```

[Строка 12](../../../../scripts/skin_face_transfer_data.py#L12)

```python
ROLES = ('train', 'validation', 'test')
```

[Строка 13](../../../../scripts/skin_face_transfer_data.py#L13)

```python
SOURCES = ('lapa', 'celeba')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ArraySplit` | — | [L57](../../../../scripts/skin_face_transfer_data.py#L57) |
| `ImageFilesSplit` | — | [L82](../../../../scripts/skin_face_transfer_data.py#L82) |
| `_Cycle` | — | [L159](../../../../scripts/skin_face_transfer_data.py#L159) |
| `PairedSampler` | — | [L177](../../../../scripts/skin_face_transfer_data.py#L177) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_face_transfer_data.py#L16) |
| `digest` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_face_transfer_data.py#L20) |
| `checked` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_face_transfer_data.py#L25) |
| `integer_ids` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_face_transfer_data.py#L30) |
| `targets` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_face_transfer_data.py#L42) |
| `ArraySplit` | ClassDef | См. реализацию | [L57](../../../../scripts/skin_face_transfer_data.py#L57) |
| `ImageFilesSplit` | ClassDef | Original LaPa TEST files, opened only by an explicitly authorized evaluator. | [L82](../../../../scripts/skin_face_transfer_data.py#L82) |
| `load_split` | FunctionDef | См. реализацию | [L109](../../../../scripts/skin_face_transfer_data.py#L109) |
| `_Cycle` | ClassDef | См. реализацию | [L159](../../../../scripts/skin_face_transfer_data.py#L159) |
| `PairedSampler` | ClassDef | См. реализацию | [L177](../../../../scripts/skin_face_transfer_data.py#L177) |
| `assemble_batch` | FunctionDef | См. реализацию | [L201](../../../../scripts/skin_face_transfer_data.py#L201) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L58–73</summary>

```python
def __init__(self, source, role, rgb, labels, indices, encoding, metadata=None):
        if source not in SOURCES or role not in ROLES or encoding not in ('lapa11', 'binary'):
            raise ValueError('Unknown source, role or encoding')
        if (rgb.ndim != 4 or rgb.shape[-1] != 3 or labels.shape != rgb.shape[:-1]
                or rgb.dtype != np.uint8 or labels.dtype != np.uint8
                or min(rgb.shape[1:3]) < 16 or any(n % 16 for n in rgb.shape[1:3])):
            raise ValueError('Expected matching uint8 RGB/label arrays with spatial multiples of 16')
        self.source, self.role, self.encoding = source, role, encoding
        self.rgb, self.labels = rgb, labels
        self.indices = integer_ids(indices, len(rgb), unique=True)
        self.indices.setflags(write=False)
        self.allowed = np.zeros(len(rgb), dtype=bool)
        self.allowed[self.indices] = True
        self.metadata = metadata
        if metadata is not None and set(metadata) != set(self.indices):
            raise ValueError('Metadata and permitted rows differ')
```

</details>

<details><summary>__init__ · L84–92</summary>

```python
def __init__(self, records):
        self.source, self.role = 'lapa', 'test'
        self.records = records
        self.indices = np.arange(len(records), dtype=np.int64)
        self.indices.setflags(write=False)
        self.metadata = {i:dict(source='lapa', role='test', original_id=r['stem'],
                               group='lapa:'+r['conservative_source_group'],
                               source_sha256=r['components']['images']['sha256'])
                         for i,r in enumerate(records)}
```

</details>

<details><summary>__init__ · L160–163</summary>

```python
def __init__(self, ids, seed, stream):
        self.ids = integer_ids(ids, unique=True)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
        self.order, self.offset = self.rng.permutation(self.ids), 0
```

</details>

<details><summary>__init__ · L178–186</summary>

```python
def __init__(self, lapa_ids, celeba_ids, seed, batch_size=32):
        if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError('Nonnegative 32-bit seed required')
        if not isinstance(batch_size, int) or batch_size < 2 or batch_size % 2:
            raise ValueError('Positive even batch size required')
        self.half = batch_size // 2
        self.anchor = _Cycle(lapa_ids, seed, 0)
        self.extra = {'lapa_only': _Cycle(lapa_ids, seed, 1), 'lapa_celeba': _Cycle(celeba_ids, seed, 2)}
        self.step, self.cross_stream_duplicate_presentations = 0, 0
```

</details>
