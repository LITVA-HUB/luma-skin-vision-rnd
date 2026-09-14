# `tests/test_chromaseed_frozen.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_frozen.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `c410ae761e11a7e463aadbbf0e0bf7ae5356c1fa06e9e816bcc06f7fb9ace85d`. Строк: **36**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_frozen_head_matches_independent_weighted_ridge_and_preserves_basis` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_frozen.py#L10) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_frozen_head_matches_independent_weighted_ridge_and_preserves_basis` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_frozen_head_matches_independent_weighted_ridge_and_preserves_basis():
    from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD, ChromaSeed, pack_model, predict
    from chromaseed_frozen import fit_readout

    torch.manual_seed(29)
    source = pack_model(ChromaSeed())
    rng = np.random.default_rng(9751)
    x = rng.uniform(.1, .9, (91, 36)).astype(np.float32)
    y = rng.normal([52, 8, 15], [12, 4, 5], (91, 3))
    w = rng.uniform(.2, 2, 91)
    alpha = 1.0
    model, _ = fit_readout(source, x, y, w, alpha)
    np.testing.assert_array_equal(model["hidden_w"], source["hidden_w"])
    np.testing.assert_array_equal(model["hidden_b"], source["hidden_b"])
    z = (x.astype(float) - X_MEAN) / X_STD
    pre = z @ source["hidden_w"].astype(float).T + source["hidden_b"]
    h = pre / (1 + np.exp(-pre))
    f = np.column_stack((z, h))
    mean, std = f.mean(0), np.maximum(f.std(0), 1e-6)
    design = np.column_stack((np.ones(len(x)), (f - mean) / std))
    penalty = np.eye(101) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ (w[:, None] * design) + penalty,
                           design.T @ (w[:, None] * ((y - Y_MEAN) / Y_STD)))
    expected = (design @ beta) * Y_STD + Y_MEAN
    np.testing.assert_allclose(predict(model, x), expected, atol=1e-3, rtol=0)
    assert sum(v.nbytes for v in model.values() if v.dtype.kind == "f") == 10996
```

</details>
