# `tests/test_chromaseed_condensed_exact.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_condensed_exact.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `352406b48f374984f0bbc9227eaf5fa841af057eded25e7634a186eed077b78a`. Строк: **40**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_condensed_lower_median_equals_frozen_blocked_definition` | FunctionDef | См. реализацию | [L11](../../../../tests/test_chromaseed_condensed_exact.py#L11) |
| `test_condensed_fit_preserves_unseen_query_predictions` | FunctionDef | См. реализацию | [L29](../../../../tests/test_chromaseed_condensed_exact.py#L29) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_condensed_lower_median_equals_frozen_blocked_definition` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('kind', ['random', 'duplicates', 'large_offset', 'constant', 'single'])
def test_condensed_lower_median_equals_frozen_blocked_definition(kind):
    from chromaseed_condensed_exact import condensed_width
    from chromaseed_kernel import median_width
    rng = np.random.default_rng(314)
    x = rng.normal(size=(139, 36))
    if kind == "duplicates":
        x = np.repeat(x[:17], 8, axis=0)
    elif kind == "large_offset":
        x = np.repeat(x[:17], 8, axis=0) + 1e6
    elif kind == "constant":
        x = np.ones_like(x)
    elif kind == "single":
        x = x[:1]
    actual, info = condensed_width(x)
    assert actual == pytest.approx(median_width(x), abs=1e-12, rel=1e-12)
    assert info["pair_count"] == len(x) * (len(x) - 1) // 2
```

</details>

### `test_condensed_fit_preserves_unseen_query_predictions` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_condensed_fit_preserves_unseen_query_predictions():
    from chromaseed_condensed_exact import fit_condensed
    from chromaseed_fast_kernel import fit_one
    from chromaseed_kernel import predict_kernel
    rng = np.random.default_rng(138)
    x, y = rng.normal(size=(171, 36)).astype(np.float32), rng.normal(size=(171, 3))
    w = rng.uniform(.2, 2, size=len(x))
    expected, _ = fit_one(x, y, w, "column_exact", 64, 17, 1, 1)
    actual, _ = fit_condensed(x, y, w, 64, 17, 1, 1)
    q = rng.normal(size=(21, 36)).astype(np.float32)
    np.testing.assert_array_equal(actual["centers"], expected["centers"])
    np.testing.assert_allclose(predict_kernel(actual, q), predict_kernel(expected, q), rtol=0, atol=2e-6)
```

</details>
