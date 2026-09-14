# `scripts/skin_uminho_acquire.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_uminho_acquire.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Original licensed UMINHO metadata and one preassigned TRAIN reflectance cube.

SHA-256 исходника: `a3003bea2e646c9533de80ad2f2bc01d0737b11c4ec34a7bd6d232b0e2a03354`. Строк: **69**.

## Зависимости

```python
import hashlib,json,urllib.request
from pathlib import Path
from skin_mskcc_data import ROOT,sha
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_uminho_acquire.py#L6)

```python
RAW=ROOT/'data/public/uminho_hsfd_v1'
```

[Строка 7](../../../../scripts/skin_uminho_acquire.py#L7)

```python
OUT=ROOT/'docs/data/provenance/uminho_hsfd_v1'
```

[Строка 8](../../../../scripts/skin_uminho_acquire.py#L8)

```python
IDS=[25598670,25594026,25599159,25599150,25599156,25599153,25637814,25638543,25638546]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fetch` | FunctionDef | См. реализацию | [L11](../../../../scripts/skin_uminho_acquire.py#L11) |
| `download` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_uminho_acquire.py#L15) |
| `main` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_uminho_acquire.py#L33) |
