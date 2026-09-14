# `tests/test_skin_copula.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_copula.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `ed4d098f49b790f36077ca5d53ce1369f03cfcb4b845ebd9bdff76b5209f2cd7`. Строк: **31**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_copula_data import rank_channels,joint_hist
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_rank_context_exactly_invariant_to_strict_channel_maps_with_ties` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_copula.py#L8) |
| `test_ties_and_constant_channel_have_midrank` | FunctionDef | См. реализацию | [L15](../../../../tests/test_skin_copula.py#L15) |
| `test_histogram_matches_counts_and_preserves_mass` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_copula.py#L22) |
| `test_channel_mixing_is_not_claimed_invariant` | FunctionDef | См. реализацию | [L28](../../../../tests/test_skin_copula.py#L28) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_rank_context_exactly_invariant_to_strict_channel_maps_with_ties` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_rank_context_exactly_invariant_to_strict_channel_maps_with_ties():
    x=np.random.default_rng(81).integers(0,256,(32,32,3))/255
    transformed=.03+.85*x**np.array([.6,1.4,2.2])
    np.testing.assert_array_equal(rank_channels(x),rank_channels(transformed))
    np.testing.assert_array_equal(joint_hist(rank_channels(x)),joint_hist(rank_channels(transformed)))
```

</details>

### `test_ties_and_constant_channel_have_midrank` · L15

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_ties_and_constant_channel_have_midrank():
    x=np.array([[0.,0.,1.],[0.,1.,1.],[1.,2.,1.],[2.,3.,1.]])
    y=rank_channels(x)
    np.testing.assert_array_equal(y[:,0],[.25,.25,.625,.875])
    np.testing.assert_array_equal(y[:,2],[.5,.5,.5,.5])
```

</details>

### `test_histogram_matches_counts_and_preserves_mass` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_histogram_matches_counts_and_preserves_mass():
    x=np.array([[0.,0.,0.],[1.,1.,1.],[.25,.5,.75],[.25,.5,.75]])
    h=joint_hist(x)
    assert h.shape==(512,) and h.sum()==1 and h[0]==.25 and h[-1]==.25 and h[2*64+4*8+6]==.5
```

</details>

### `test_channel_mixing_is_not_claimed_invariant` · L28

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_channel_mixing_is_not_claimed_invariant():
    x=np.random.default_rng(82).uniform(size=(64,64,3))
    matrix=np.array([[.8,.1,.1],[.15,.7,.15],[.1,.2,.7]])
    assert not np.array_equal(joint_hist(rank_channels(x)),joint_hist(rank_channels(x@matrix.T)))
```

</details>
