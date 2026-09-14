# `tests/test_chromaseed_selection_stability.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_selection_stability.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `44acb95265617374e368318961824f98c55a3ac06bfeb01441e148461ef02006`. Строк: **88**.

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
| `test_people_not_images_or_prediction_ensemble` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_selection_stability.py#L10) |
| `test_stratified_counts_preserve_camera_people_and_are_reproducible` | FunctionDef | См. реализацию | [L26](../../../../tests/test_chromaseed_selection_stability.py#L26) |
| `configs` | FunctionDef | См. реализацию | [L39](../../../../tests/test_chromaseed_selection_stability.py#L39) |
| `test_exact_ties_follow_registered_order` | FunctionDef | См. реализацию | [L46](../../../../tests/test_chromaseed_selection_stability.py#L46) |
| `test_hand_computable_deletion_bootstrap_and_paired_gaps` | FunctionDef | См. реализацию | [L54](../../../../tests/test_chromaseed_selection_stability.py#L54) |
| `test_one_person_can_change_selection_without_refitting` | FunctionDef | См. реализацию | [L72](../../../../tests/test_chromaseed_selection_stability.py#L72) |
| `test_invalid_scores_fail` | FunctionDef | См. реализацию | [L84](../../../../tests/test_chromaseed_selection_stability.py#L84) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_people_not_images_or_prediction_ensemble` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_people_not_images_or_prediction_ensemble():
    from chromaseed_selection_stability import person_losses

    from luma_skin_vision.color import delta_e00

    target = np.tile([50., 0., 0.], (4, 1))
    prediction = np.stack([target.copy(), target.copy()])
    prediction[0, :, 0] += [10., 10., 10., 0.]
    prediction[1, :, 0] -= [10., 10., 10., 0.]
    actual = person_losses(prediction, target, np.array([7, 7, 7, 9]))
    expected_first = np.mean([delta_e00(prediction[s, :1], target[:1])[0] for s in range(2)])
    np.testing.assert_allclose(actual, [expected_first, 0.], rtol=0, atol=1e-12)
    assert actual.mean() > 0.  # Averaged predictions would falsely give zero.
    assert actual.mean() != pytest.approx(expected_first * .75)
```

</details>

### `test_stratified_counts_preserve_camera_people_and_are_reproducible` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_stratified_counts_preserve_camera_people_and_are_reproducible():
    from chromaseed_selection_stability import bootstrap_counts

    camera = np.array(['a', 'b', 'b', 'a', 'b'])
    counts = bootstrap_counts(camera, 1000, 19)
    assert counts.shape == (1000, 5)
    assert counts.dtype.kind in 'iu'
    np.testing.assert_array_equal(counts[:, camera == 'a'].sum(axis=1), 2)
    np.testing.assert_array_equal(counts[:, camera == 'b'].sum(axis=1), 3)
    np.testing.assert_array_equal(counts, bootstrap_counts(camera, 1000, 19))
    assert np.any(counts > 1)
```

</details>

### `test_exact_ties_follow_registered_order` · L46

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_exact_ties_follow_registered_order():
    from chromaseed_selection_stability import winner_indices

    scores = np.ones((2, 4))
    scores[1, 0] -= 1e-12  # Do not collapse near ties.
    np.testing.assert_array_equal(winner_indices(scores, configs()), [3, 0])
```

</details>

### `test_hand_computable_deletion_bootstrap_and_paired_gaps` · L54

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_hand_computable_deletion_bootstrap_and_paired_gaps():
    from chromaseed_selection_stability import diagnose

    losses = np.array([[0., 0., 6.], [3., 3., 3.]])
    cfg = [dict(steps=0, alpha=.01, width_factor=1.), dict(steps=1, alpha=.1, width_factor=1.)]
    counts = np.array([[3, 0, 0], [0, 0, 3], [1, 1, 1], [0, 3, 0]])
    result, trace = diagnose(losses, cfg, counts, parent_index=1)
    assert result['original_index'] == 0
    np.testing.assert_array_equal(trace['deletion_winners'], [0, 0, 0])
    np.testing.assert_array_equal(trace['bootstrap_winners'], [0, 1, 0, 0])
    np.testing.assert_array_equal(trace['bootstrap_original_ranks'], [1, 2, 1, 1])
    np.testing.assert_array_equal(trace['gap_vs_parent'], [-3., 3., -1., -3.])
    assert result['bootstrap_original_frequency'] == .75
    assert result['bootstrap_weak_alpha_frequency'] == .75
    assert result['deletion_switches'] == 0
    assert result['paired_vs_parent']['mean'] == -1.
```

</details>

### `test_one_person_can_change_selection_without_refitting` · L72

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_one_person_can_change_selection_without_refitting():
    from chromaseed_selection_stability import diagnose

    losses = np.array([[0., 5., 5.], [4., 3., 3.]])
    cfg = [dict(steps=0, alpha=.01, width_factor=1.), dict(steps=1, alpha=.1, width_factor=1.)]
    result, trace = diagnose(losses, cfg, np.array([[1, 1, 1]]), parent_index=1)
    assert result['original_index'] == 0
    assert result['deletion_switches'] == 1
    np.testing.assert_array_equal(trace['deletion_winners'], [1, 0, 0])
```

</details>

### `test_invalid_scores_fail` · L84

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', [np.array([[np.nan, 2.0], [1.0, 2.0]]), np.ones((2, 1)), np.ones((1, 3))])
def test_invalid_scores_fail(bad):
    from chromaseed_selection_stability import diagnose

    with pytest.raises(ValueError):
        diagnose(bad, configs()[:2], np.ones((2, bad.shape[1]), dtype=int), parent_index=0)
```

</details>
