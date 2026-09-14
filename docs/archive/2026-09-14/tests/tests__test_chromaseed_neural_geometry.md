# `tests/test_chromaseed_neural_geometry.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_neural_geometry.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent hat-trace and invariance tests for effective neural capacity.

SHA-256 исходника: `3ba5bb6af2d845a4f7aead5f2af1edd3c00395552c5d5baac3f2a3096f5529d5`. Строк: **65**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `core` | FunctionDef | См. реализацию | [L13](../../../../tests/test_chromaseed_neural_geometry.py#L13) |
| `problem` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_neural_geometry.py#L20) |
| `test_df_matches_full_hat_trace` | FunctionDef | См. реализацию | [L37](../../../../tests/test_chromaseed_neural_geometry.py#L37) |
| `test_intercept_is_exempt_and_hidden_shift_changes_nothing` | FunctionDef | См. реализацию | [L47](../../../../tests/test_chromaseed_neural_geometry.py#L47) |
| `test_invalid_metric_and_nonfinite_design_are_rejected` | FunctionDef | См. реализацию | [L58](../../../../tests/test_chromaseed_neural_geometry.py#L58) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_df_matches_full_hat_trace` · L37

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('coupled', [False, True])
@pytest.mark.parametrize('alpha', [0.1, 10, 1000])
def test_df_matches_full_hat_trace(coupled, alpha):
    spectrum, degrees = core()
    z, w, m, a, bias, multiplier = problem(coupled)
    g = a.T @ a
    penalty = np.diag(np.r_[np.zeros(bias), np.full(g.shape[0] - bias, alpha)])
    expected = multiplier * np.trace(np.linalg.solve(g + penalty, g))
    assert degrees(spectrum(z, w, m), alpha) == pytest.approx(expected, abs=1e-9)
```

</details>

### `test_intercept_is_exempt_and_hidden_shift_changes_nothing` · L47

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('coupled', [False, True])
def test_intercept_is_exempt_and_hidden_shift_changes_nothing(coupled):
    spectrum, degrees = core()
    z, w, m, _, _, _ = problem(coupled)
    original = spectrum(z, w, m)
    shifted = z.copy()
    shifted[:, 1:] += np.arange(5) * 7
    assert degrees(spectrum(shifted, w, m), 10) == pytest.approx(degrees(original, 10), abs=1e-9)
    assert degrees(original, 1e15) == pytest.approx(3, abs=1e-9)
    assert degrees(spectrum(np.ones((19, 1)), w, m), 10) == pytest.approx(3)
```

</details>

### `test_invalid_metric_and_nonfinite_design_are_rejected` · L58

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_metric_and_nonfinite_design_are_rejected():
    spectrum, _ = core()
    z, w, _, _, _, _ = problem(False)
    with pytest.raises(ValueError):
        spectrum(z, w, -np.repeat(np.eye(3)[None], len(z), axis=0))
    z[0, 1] = np.nan
    with pytest.raises(ValueError):
        spectrum(z, w)
```

</details>
