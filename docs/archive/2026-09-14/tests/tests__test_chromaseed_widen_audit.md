# `tests/test_chromaseed_widen_audit.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_widen_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent active-neuron contractions agree with the executable consumer.

SHA-256 исходника: `8dc1039351cd52b9c5440a05a1c4b57a3a63027ad5c0909b85982c6a60a91f5a`. Строк: **34**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_patch8_fit import token_normalizers
from chromaseed_widen import Bank, predict
from chromaseed_widen_audit import direct, local_transform
from test_chromaseed_patch8 import fixture, independent_tokens
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_independent_contraction_with_all_branches_active` | FunctionDef | См. реализацию | [L17](../../../../tests/test_chromaseed_widen_audit.py#L17) |
| `test_independent_affine_transform_matches_pixel_statistics` | FunctionDef | См. реализацию | [L26](../../../../tests/test_chromaseed_widen_audit.py#L26) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_independent_contraction_with_all_branches_active` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('variant', ['tiny', 'm31', 'm61', 'm111', 'm832'])
def test_independent_contraction_with_all_branches_active(variant):
    x, t, _, warm = fixture(5)
    m = Bank(warm, variant).export(0, warm, token_normalizers(t))
    rng = np.random.default_rng(235)
    m["g"][:] = rng.normal(0, 0.02, m["g"].shape).astype(np.float32)
    m["v0"][:] = rng.normal(0, 0.03, m["v0"].shape).astype(np.float32)
    np.testing.assert_allclose(direct(m, x, t), predict(m, x, t), rtol=0, atol=2e-8)
```

</details>

### `test_independent_affine_transform_matches_pixel_statistics` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_independent_affine_transform_matches_pixel_statistics():
    rgb = np.random.default_rng(273).uniform(size=(128, 128, 3))
    tokens = independent_tokens(rgb).astype(np.float32)[None]
    anchor = np.array([0.2, 0.6, 0.8])
    dose = 64 / 255
    expected = independent_tokens(rgb + dose * (anchor - rgb)).astype(np.float32)
    np.testing.assert_allclose(
        local_transform(tokens, dose, anchor)[0], expected, rtol=0, atol=6e-8
    )
```

</details>
