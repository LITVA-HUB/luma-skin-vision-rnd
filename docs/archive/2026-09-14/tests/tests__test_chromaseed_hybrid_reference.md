# `tests/test_chromaseed_hybrid_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_hybrid_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent dense/SVD/QR reconstruction of H's staged learning.

SHA-256 исходника: `bb0f265ced3ff1214abead8605ce42b597cff470057760b743ac612923c27576`. Строк: **44**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_hybrid import fit_bank, model_id, predict
from chromaseed_hybrid_reference import Basis, Geometry
from chromaseed_hybrid_reference import predict as reference_predict
from test_chromaseed_projection import toy
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_independent_staged_fit_and_predictions` | FunctionDef | См. реализацию | [L17](../../../../tests/test_chromaseed_hybrid_reference.py#L17) |
| `test_reference_support_and_uniform_are_different_learned_paths` | FunctionDef | См. реализацию | [L40](../../../../tests/test_chromaseed_hybrid_reference.py#L40) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_independent_staged_fit_and_predictions` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('two', [True, False])
def test_independent_staged_fit_and_predictions(two):
    data = toy(two)
    models, _ = fit_bank(*data, rank=4)
    b = Basis(Geometry(*data), 17, rank=4)
    for loss in ("norm", "perceptual"):
        for kind, alpha, rho, power in (
            ("raw", 0.1, 0.0, 0),
            ("projected", 1.0, 1.0, 0),
            ("blend", 0.1, 0.5, 0),
            ("uniform", 10.0, 0.25, 0),
            ("support", 1.0, 1.0, 1),
            ("support", 0.1, 0.5, 4),
        ):
            m = models[model_id(loss, 17, kind, alpha, rho, power)]
            independent = b.make(loss, kind, alpha, rho, power)
            np.testing.assert_allclose(
                reference_predict(m, data[0]), predict(m, data[0]), atol=2e-8, rtol=0
            )
            np.testing.assert_allclose(
                reference_predict(independent, data[0]), predict(m, data[0]), atol=0.001, rtol=0
            )
```

</details>

### `test_reference_support_and_uniform_are_different_learned_paths` · L40

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_reference_support_and_uniform_are_different_learned_paths():
    data = toy()
    b = Basis(Geometry(*data), 17, rank=4)
    a, c = b.make("norm", "uniform", 0.1, 0.5, 0), b.make("norm", "support", 0.1, 0.5, 4)
    assert np.max(np.abs(a["latent_coefficient"] - c["latent_coefficient"])) > 1e-4
```

</details>
