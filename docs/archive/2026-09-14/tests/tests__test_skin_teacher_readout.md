# `tests/test_skin_teacher_readout.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_teacher_readout.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `cc7e9dcc3bd1e33c64a176aee4628f648f478b7285c3d6499cf7e7ff3dc72f0f`. Строк: **32**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from skin_teacher_readout import fit_blocks, transform_blocks, site_weights, source_permutation
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_scaler_uses_fit_columns_and_equalizes_block_energy` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_teacher_readout.py#L9) |
| `test_site_weight_does_not_multiply_repeated_views` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_teacher_readout.py#L19) |
| `test_shuffles_stay_within_subset_and_are_repeatable` | FunctionDef | См. реализацию | [L26](../../../../tests/test_skin_teacher_readout.py#L26) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_scaler_uses_fit_columns_and_equalizes_block_energy` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_scaler_uses_fit_columns_and_equalizes_block_energy():
    x = np.array([[1., 10], [3., 10]])
    scaler = fit_blocks([x])
    z = transform_blocks([x], scaler)
    np.testing.assert_allclose(z, [[-1/np.sqrt(2), 0], [1/np.sqrt(2), 0]])
    v = transform_blocks([np.array([[101., 20.]])], scaler)
    np.testing.assert_allclose(v, [[99/np.sqrt(2), 10/np.sqrt(2)]])
    assert scaler[0]['mean'].tolist() == [2, 10]
```

</details>

### `test_site_weight_does_not_multiply_repeated_views` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_site_weight_does_not_multiply_repeated_views():
    w = site_weights(np.array(['a', 'a', 'a', 'b']))
    assert w.mean() == pytest.approx(1)
    assert w[:3].sum() == pytest.approx(w[3])
    np.testing.assert_allclose(w, [2/3, 2/3, 2/3, 2])
```

</details>

### `test_shuffles_stay_within_subset_and_are_repeatable` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_shuffles_stay_within_subset_and_are_repeatable():
    p = source_permutation(17, 11, 'fit')
    np.testing.assert_array_equal(np.sort(p), np.arange(17))
    np.testing.assert_array_equal(p, source_permutation(17, 11, 'fit'))
    assert not np.array_equal(p, source_permutation(17, 11, 'selection'))
    with pytest.raises(ValueError):
        source_permutation(17, 11, 'test')
```

</details>
