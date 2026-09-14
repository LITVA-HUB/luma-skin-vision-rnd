# `src/luma_skin_vision/gates.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/gates.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Require explicit measured-protocol review before fitting non-synthetic targets.

SHA-256 исходника: `5eb6b0e016cc3a5df5a03cbd11e3002df40a018937d35442b53524a8f406dc69`. Строк: **41**.

## Зависимости

```python
import json
import math
from pathlib import Path
from luma_skin_vision.data import sha256
from luma_skin_vision.evaluation import repeatability
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `measurement_gate` | FunctionDef | См. реализацию | [L11](../../../../src/luma_skin_vision/gates.py#L11) |
