# `scripts/skin_relational_probe_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_relational_probe_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Relational signal versus actual skin accuracy, with scope-separated controls.

SHA-256 исходника: `bf9db65d6d64157886f952b3695ef0754a2bf6de76bbcb1a0038b225ecec5e68`. Строк: **130**.

## Зависимости

```python
import csv,json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from skin_relational_probe_run import OUT
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_relational_probe_report.py#L12) |
