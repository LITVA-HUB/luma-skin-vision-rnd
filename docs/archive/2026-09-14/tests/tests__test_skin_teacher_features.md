# `tests/test_skin_teacher_features.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_teacher_features.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d41633c13d97217371752e2df9921e439eb9fbff2194b2124684fcc769eaa225`. Строк: **22**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from skin_teacher_features import preprocess
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_constant_rgb_has_exact_declared_normalization` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_teacher_features.py#L10) |
| `test_input_contract_rejects_float_or_wrong_shape` | FunctionDef | См. реализацию | [L18](../../../../tests/test_skin_teacher_features.py#L18) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_constant_rgb_has_exact_declared_normalization` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_constant_rgb_has_exact_declared_normalization():
    rgb = np.broadcast_to(np.array([0, 128, 255], dtype=np.uint8), (2, 128, 128, 3)).copy()
    x = preprocess(rgb)
    expected = (torch.tensor([0., 128/255, 1])-torch.tensor([.485, .456, .406]))/torch.tensor([.229, .224, .225])
    assert x.shape == (2, 3, 224, 224)
    torch.testing.assert_close(x, expected[None, :, None, None].expand_as(x), atol=1e-6, rtol=0)
```

</details>

### `test_input_contract_rejects_float_or_wrong_shape` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_input_contract_rejects_float_or_wrong_shape():
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 128, 128, 3)))
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 128, 128), dtype=np.uint8))
```

</details>
