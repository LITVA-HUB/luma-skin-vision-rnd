# `tests/test_chromaseed_weak_ridge.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_weak_ridge.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `58f8c7deb778941920703f18471291e013f6fcd03da362101735db0df6e55202`. Строк: **52**.

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
| `data` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_weak_ridge.py#L10) |
| `test_original_alpha_payloads_are_unchanged` | FunctionDef | См. реализацию | [L19](../../../../tests/test_chromaseed_weak_ridge.py#L19) |
| `test_weak_penalty_unseen_predictions_match_independent_reference` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed_weak_ridge.py#L31) |
| `test_expanded_grid_includes_exact_controls_and_zero_aliases` | FunctionDef | См. реализацию | [L43](../../../../tests/test_chromaseed_weak_ridge.py#L43) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_original_alpha_payloads_are_unchanged` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['norm_mse', 'constant_de2', 'local_de2', 'local_irls', 'midpoint_irls'])
def test_original_alpha_payloads_are_unchanged(family):
    from chromaseed_perceptual import fit_single as original
    from chromaseed_weak_ridge import ALPHAS, fit_single
    x, y, w, _ = data()
    steps = 4 if family.endswith("irls") else (0 if family == "norm_mse" else 1)
    expected, _ = original(x, y, w, family, 17, 1, 0, steps, rank=18)
    actual, _ = fit_single(x, y, w, family, 17, 1, ALPHAS.index(.1), steps, rank=18)
    for field in actual:
        np.testing.assert_array_equal(actual[field], expected[field])
```

</details>

### `test_weak_penalty_unseen_predictions_match_independent_reference` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['norm_mse', 'constant_de2', 'local_de2', 'local_irls', 'midpoint_irls'])
def test_weak_penalty_unseen_predictions_match_independent_reference(family):
    from chromaseed_kernel import predict_kernel
    from chromaseed_perceptual_audit import reference_readouts
    from chromaseed_weak_ridge import fit_single
    x, y, w, query = data()
    steps = 4 if family.endswith("irls") else (0 if family == "norm_mse" else 1)
    model, _ = fit_single(x, y, w, family, 17, 1, 0, steps, rank=18)
    coefficients, _ = reference_readouts(model, x, y, w, family, .0001, steps)
    reference = {**model, "coefficient": coefficients[steps]}
    np.testing.assert_allclose(predict_kernel(model, query), predict_kernel(reference, query), rtol=0, atol=.0002)
```

</details>

### `test_expanded_grid_includes_exact_controls_and_zero_aliases` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_expanded_grid_includes_exact_controls_and_zero_aliases():
    from chromaseed_weak_ridge import ALPHAS, key
    from chromaseed_weak_ridge_train import candidate_grid
    assert ALPHAS == (.0001, .0003, .001, .003, .01, .03, .1, 1., 10.)
    assert len(list(candidate_grid("norm_mse"))) == 27
    grid = list(candidate_grid("local_irls"))
    assert len(grid) == 108
    for wi, ai, step in grid:
        if step == 0:
            assert key("local_irls", 17, wi, ai, step) == key("norm_mse", 17, wi, ai, 0)
```

</details>
