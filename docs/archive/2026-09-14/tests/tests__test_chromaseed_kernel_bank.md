# `tests/test_chromaseed_kernel_bank.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_kernel_bank.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `271f44ff5fdda62567179ae3a34fff1209ddc14df2d333a56d1c82931b276d3e`. Строк: **34**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_small_bank_has_registered_configs_finite_predictions_and_valid_bounds` | FunctionDef | См. реализацию | [L9](../../../../tests/test_chromaseed_kernel_bank.py#L9) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_small_bank_has_registered_configs_finite_predictions_and_valid_bounds` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_small_bank_has_registered_configs_finite_predictions_and_valid_bounds():
    from chromaseed_kernel import AdaptiveKernel, predict_kernel
    from chromaseed_kernel_bank import (
        evaluate_bank,
        fit_bank,
        flatten_bank,
        make_adaptive,
        model_id,
    )

    rng = np.random.default_rng(4073)
    x, y = rng.normal(size=(38, 36)).astype(np.float32), rng.normal(size=(38, 3)) * [10, 4, 6] + [50, 4, 10]
    models, diagnostics, receipt = fit_bank(x, y, np.ones(len(x)), exact_backend="cpu")
    assert len(models) == receipt["readout_configurations"] == 369
    query = rng.normal(size=(7, 36)).astype(np.float32)
    predictions, violation = evaluate_bank(models, diagnostics, receipt, query, np.arange(7))
    assert violation <= 1e-6
    for key in list(models)[::37]:
        np.testing.assert_allclose(predictions[f"pred__{key}"], predict_kernel(models[key], query), rtol=0, atol=1e-10)
    arrays = flatten_bank(models, diagnostics)
    adaptive = AdaptiveKernel(make_adaptive(arrays, 1, 0, 17, 0.))
    expected = predictions[f"pred__{model_id('project_rpchol', 128, 17, 1, 0)}"]
    for i in range(7):
        prediction, used, bound, met = adaptive.predict_one(query[i])
        np.testing.assert_allclose(prediction, expected[i], rtol=0, atol=1e-10)
        assert used <= 38 and not met and np.isfinite(bound)
```

</details>
