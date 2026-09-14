# `tests/test_skin_spectral_probe.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_spectral_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `354cb01f3edd363fe30704bf5bba36f81fc1588720668f8a36556a3aa81e15e4`. Строк: **39**.

## Зависимости

```python
import numpy as np
from scripts.skin_spectral_probe_v1 import project_spectra, smooth_illumination_match
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_linear_projection_exact_in_spanned_affine_space` | FunctionDef | См. реализацию | [L5](../../../../tests/test_skin_spectral_probe.py#L5) |
| `test_log_projection_preserves_positive_multiplicative_family` | FunctionDef | См. реализацию | [L13](../../../../tests/test_skin_spectral_probe.py#L13) |
| `test_polynomial_illumination_counterexample_and_direction` | FunctionDef | См. реализацию | [L23](../../../../tests/test_skin_spectral_probe.py#L23) |
| `test_projection_does_not_refit_on_query` | FunctionDef | См. реализацию | [L33](../../../../tests/test_skin_spectral_probe.py#L33) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_linear_projection_exact_in_spanned_affine_space` · L5

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_linear_projection_exact_in_spanned_affine_space():
    rng = np.random.default_rng(101)
    basis = rng.normal(size=(2, 33))
    train = 3 + rng.normal(size=(20, 2)) @ basis
    query = 3 + rng.normal(size=(7, 2)) @ basis
    np.testing.assert_allclose(project_spectra(train, query, 2, 'linear'), query, atol=1e-12)
```

</details>

### `test_log_projection_preserves_positive_multiplicative_family` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_log_projection_preserves_positive_multiplicative_family():
    rng = np.random.default_rng(102)
    basis = rng.normal(size=(2, 33)) * .1
    train = np.exp(-2 + rng.normal(size=(20, 2)) @ basis)
    query = np.exp(-2 + rng.normal(size=(7, 2)) @ basis)
    actual = project_spectra(train, query, 2, 'log')
    assert (actual > 0).all()
    np.testing.assert_allclose(actual, query, atol=1e-12)
```

</details>

### `test_polynomial_illumination_counterexample_and_direction` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_polynomial_illumination_counterexample_and_direction():
    x = np.linspace(-1, 1, 33)
    first = .2 + .03 * np.sin(x * 5)
    illuminant = np.exp(.3 + .1*x - .07*x*x)
    second = first / illuminant
    fitted, corrected = smooth_illumination_match(first, second, 2)
    np.testing.assert_allclose(fitted, illuminant, atol=1e-12)
    np.testing.assert_allclose(corrected, first, atol=1e-12)
```

</details>

### `test_projection_does_not_refit_on_query` · L33

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_projection_does_not_refit_on_query():
    rng = np.random.default_rng(103)
    train = rng.uniform(.1, .5, (30, 33))
    query = rng.uniform(.1, .5, (4, 33))
    a = project_spectra(train, query, 3, 'log')
    b = project_spectra(train, np.concatenate([query, query*8]), 3, 'log')[:4]
    np.testing.assert_allclose(a, b, atol=1e-14)
```

</details>
