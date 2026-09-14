# `tests/test_chromaseed_affine_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_affine_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

QR compression preserves a separately constructed augmented coupled objective.

SHA-256 исходника: `9e83dec3a40e99e13178d59d3bf52ac391d474041b1cc89ea9651041562e9521`. Строк: **24**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_affine_reference import qr_ridge
from chromaseed_perceptual_reference import augmented_svd
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_augmented_qr_svd_matches_full_coupled_svd` | FunctionDef | См. реализацию | [L15](../../../../tests/test_chromaseed_affine_reference.py#L15) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_augmented_qr_svd_matches_full_coupled_svd` · L15

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0])
def test_augmented_qr_svd_matches_full_coupled_svd(alpha):
    rng = np.random.default_rng(917372)
    design = rng.normal(size=(171, 11))
    design[:, -1] = design[:, 0] + 1e-9 * design[:, 2]
    y = rng.normal(size=(171, 3))
    weights = rng.uniform(0.001, 2, 171)
    metric = np.array([[1.4, 0.2, -0.3], [0.2, 0.5, 0.07], [-0.3, 0.07, 1.2]])
    expected = augmented_svd(design, y, np.broadcast_to(metric, (171, 3, 3)), weights, alpha)
    actual = qr_ridge(design, y, weights, metric, alpha)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
```

</details>
