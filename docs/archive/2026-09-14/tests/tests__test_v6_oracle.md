# `tests/test_v6_oracle.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_v6_oracle.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `3428c76085bef937cb4545123cd6b17a290bebe529aae6d78b3d06676e29351d`. Строк: **22**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from cc_v4_experiment import action_rgb, oracle_errors
from cc_v6_oracle_audit import corrected_oracle
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_oracle_search_clamps_residual_not_absolute_coordinates` | FunctionDef | См. реализацию | [L11](../../../../tests/test_v6_oracle.py#L11) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_oracle_search_clamps_residual_not_absolute_coordinates` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_oracle_search_clamps_residual_not_absolute_coordinates():
    point=np.array([[.15,0.]])
    anchor=np.array([[1.95,0.]])
    trajectory=np.tile(point[:,None],(1,4,1))
    gt=action_rgb(point+anchor+np.array([[.12,0.]]))
    actual=corrected_oracle(point,trajectory,anchor,gt)
    old=oracle_errors(point+anchor,trajectory+anchor[:,None],gt)
    assert actual[0][0] < 1e-6
    # First-stage clamped points differ; later stages include the base point.
    # Test the complete candidate coordinates, not merely a lucky minimum.
    assert actual[2].max() > 2
    assert old[0][0] > .05
```

</details>
