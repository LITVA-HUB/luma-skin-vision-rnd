# `tests/test_chromaseed_widen_early.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_widen_early.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

The optimization study must preserve original trajectories and paired canaries.

SHA-256 исходника: `698287d1c6d38834bfc1b8aaf4a9e56b4425e9110cd14643fdf39e740c6c3282`. Строк: **68**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from chromaseed_widen import fit as old_fit
from chromaseed_widen import rate_factors
from chromaseed_widen_early_fit import fit
from test_chromaseed_patch8 import fixture
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `same` | FunctionDef | См. реализацию | [L17](../../../../tests/test_chromaseed_widen_early.py#L17) |
| `test_original_rate_cpu_preserves_weights_and_all_receipt_streams` | FunctionDef | См. реализацию | [L23](../../../../tests/test_chromaseed_widen_early.py#L23) |
| `test_lower_rate_is_applied_without_affecting_high_rate_canaries` | FunctionDef | См. реализацию | [L35](../../../../tests/test_chromaseed_widen_early.py#L35) |
| `test_largest_cuda_original_prefix_and_paired_canary_identity` | FunctionDef | См. реализацию | [L49](../../../../tests/test_chromaseed_widen_early.py#L49) |
| `test_unknown_rate_cannot_silently_change_bank_coordinates` | FunctionDef | См. реализацию | [L65](../../../../tests/test_chromaseed_widen_early.py#L65) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_original_rate_cpu_preserves_weights_and_all_receipt_streams` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_original_rate_cpu_preserves_weights_and_all_receipt_streams():
    x, t, y, warm = fixture()
    args = (x, t, y, np.ones(len(x)), warm, "m31")
    a, ia = old_fit(*args, 8, (0, 4, 8))
    b, ib = fit(*args, 0.0001, 8, (0, 4, 8))
    for step in a:
        for slot in range(6):
            same(a[step][slot], b[step][slot])
    assert ia["sampling_sha256"] == ib["sampling_sha256"]
    assert ia["slots"] == ib["slots"] and ib["schedule_horizon"] == 8192
```

</details>

### `test_lower_rate_is_applied_without_affecting_high_rate_canaries` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lower_rate_is_applied_without_affecting_high_rate_canaries():
    x, t, y, warm = fixture()
    args = (x, t, y, np.ones(len(x)), warm, "m31")
    a, _ = fit(*args, 0.0001, 8, (0, 4, 8))
    b, info = fit(*args, 0.00001, 8, (0, 4, 8))
    for step in a:
        for slot in (1, 3, 5):
            same(a[step][slot], b[step][slot])
    assert np.any(a[8][0]["w0"] != b[8][0]["w0"])
    rates = np.tile(np.array([0.00001, 0.001], np.float32), 3) * rate_factors(8)[-1]
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), rates)
```

</details>

### `test_largest_cuda_original_prefix_and_paired_canary_identity` · L49

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA absent')
def test_largest_cuda_original_prefix_and_paired_canary_identity():
    x, t, y, warm = fixture(6)
    args = (x, t, y, np.ones(len(x)), warm, "m832")
    a, _ = old_fit(*args, 16, (0, 8, 16), "cuda", "cuda_graph")
    b, _ = fit(*args, 0.0001, 16, (0, 8, 16), "cuda", "cuda_graph")
    c, _ = fit(*args, 0.00003, 16, (0, 8, 16), "cuda", "cuda_graph")
    d, _ = fit(*args, 0.00003, 8, (0, 8), "cuda", "cuda_graph")
    for step in a:
        for slot in range(6):
            same(a[step][slot], b[step][slot])
            if slot % 2:
                same(b[step][slot], c[step][slot])
    for slot in range(6):
        same(c[8][slot], d[8][slot])
```

</details>

### `test_unknown_rate_cannot_silently_change_bank_coordinates` · L65

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_unknown_rate_cannot_silently_change_bank_coordinates():
    x, t, y, warm = fixture()
    with pytest.raises(ValueError):
        fit(x, t, y, np.ones(len(x)), warm, "m31", 0.0002, 4, (0, 4))
```

</details>
