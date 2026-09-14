# `tests/test_skin_he_xyz.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_he_xyz.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d5ed0045b449c09941ca6f8ca4e4f09bffd85aa6ad1159b2ba6cf8115a5b60bc`. Строк: **36**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from cc_v7_external_audit import angle, percentile
from skin_he_xyz import inputs, metrics, model
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_linear_calibration_recovers_known_cross_channel_mapping` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_he_xyz.py#L12) |
| `test_root_features_preserve_scalar_exposure` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_he_xyz.py#L19) |
| `test_xyz_error_is_coordinate_error_with_brightness_sensitivity` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_he_xyz.py#L24) |
| `test_independent_atan2_handles_parallel_and_perpendicular_vectors` | FunctionDef | См. реализацию | [L33](../../../../tests/test_skin_he_xyz.py#L33) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_linear_calibration_recovers_known_cross_channel_mapping` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_linear_calibration_recovers_known_cross_channel_mapping():
    x = np.random.default_rng(17).uniform(.1, 1, (30, 3))
    mapping = np.array([[3, 1, .2], [.1, 2, .3], [.5, .2, 4]])
    fitted = model("linear3").fit(x[:20], x[:20] @ mapping)
    np.testing.assert_allclose(fitted.predict(x[20:]), x[20:] @ mapping, atol=1e-12)
```

</details>

### `test_root_features_preserve_scalar_exposure` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_root_features_preserve_scalar_exposure():
    x = np.array([[.2, .3, .4], [.1, .8, .3]])
    np.testing.assert_allclose(inputs(x*7, "root2"), inputs(x, "root2")*7)
```

</details>

### `test_xyz_error_is_coordinate_error_with_brightness_sensitivity` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_xyz_error_is_coordinate_error_with_brightness_sensitivity():
    gt = np.array([[10., 20., 30.], [10., 20., 30.]])
    pred = gt + [3, 4, 0]
    report = metrics(pred, gt)
    assert report["xyz_rmse"] == pytest.approx(5/np.sqrt(3))
    assert report["median_euclidean_xyz"] == 5
    assert metrics(gt*2, gt)["xyz_rmse"] > 0
```

</details>

### `test_independent_atan2_handles_parallel_and_perpendicular_vectors` · L33

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_independent_atan2_handles_parallel_and_perpendicular_vectors():
    assert angle([1, 2, 3], [2, 4, 6]) == 0
    assert angle([1, 0, 0], [0, 1, 0]) == 90
    assert percentile([0., 2., 6., 8.], .25) == 1.5
```

</details>
