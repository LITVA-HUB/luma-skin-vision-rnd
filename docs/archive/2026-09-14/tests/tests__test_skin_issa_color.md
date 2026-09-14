# `tests/test_skin_issa_color.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_issa_color.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `46093e7d19bd81057aba12d4da2410099a52aaa3b7f199893d7f45c94f9e2886`. Строк: **18**.

## Зависимости

```python
import numpy as np
import pytest
from scripts.skin_issa_color import spectral_xyz, source_lab
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_percent_reflectance_scale_and_white_are_explicit` | FunctionDef | См. реализацию | [L6](../../../../tests/test_skin_issa_color.py#L6) |
| `test_missing_spectral_samples_cannot_silently_be_zero_filled` | FunctionDef | См. реализацию | [L15](../../../../tests/test_skin_issa_color.py#L15) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_percent_reflectance_scale_and_white_are_explicit` · L6

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_percent_reflectance_scale_and_white_are_explicit():
    cmf = np.array([[.2,.1,.7],[.5,.8,.3],[.1,.4,.2]])
    spd = np.array([20.,70.,10.])
    white = spectral_xyz(np.full((1,3), 100.), cmf, spd)[0]
    xyz = spectral_xyz(np.full((1,3), 50.), cmf, spd)
    np.testing.assert_allclose(xyz[0], .5*white, atol=1e-12)
    np.testing.assert_allclose(source_lab(xyz,white)[0], [116*np.cbrt(.5)-16,0,0], atol=1e-12)
```

</details>

### `test_missing_spectral_samples_cannot_silently_be_zero_filled` · L15

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_missing_spectral_samples_cannot_silently_be_zero_filled():
    with pytest.raises(ValueError, match='finite'):
        spectral_xyz(np.array([[2.,np.nan]]),np.ones((2,3)),np.ones(2))
    np.testing.assert_allclose(source_lab(np.zeros((1,3)),np.ones(3)*100),np.zeros((1,3)),atol=1e-12)
```

</details>
