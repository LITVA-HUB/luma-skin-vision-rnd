# `tests/test_skin_pixel_experiment.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_pixel_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `ff6ee5f57873fd3a2adad16659f42a4880da0afd5266cb2aef22540dab435c80`. Строк: **32**.

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
| `test_confusion_matrix_and_imbalanced_accuracy` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_pixel_experiment.py#L10) |
| `test_winner_uses_validation_balanced_accuracy_only` | FunctionDef | См. реализацию | [L20](../../../../tests/test_skin_pixel_experiment.py#L20) |
| `test_metric_boundary_rejects_bad_probabilities` | FunctionDef | См. реализацию | [L27](../../../../tests/test_skin_pixel_experiment.py#L27) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_confusion_matrix_and_imbalanced_accuracy` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_confusion_matrix_and_imbalanced_accuracy():
    from skin_pixel_experiment import metrics
    m = metrics(np.array([1, 1, 0, 0, 0, 0]), np.array([.9, .2, .6, .1, .2, .3]), .5)
    assert m['tp'] == 1 and m['fn'] == 1 and m['fp'] == 1 and m['tn'] == 3
    assert m['accuracy'] == pytest.approx(4/6)
    assert m['balanced_accuracy'] == pytest.approx(.625)
    assert m['skin_precision'] == pytest.approx(.5)
    assert m['skin_recall'] == pytest.approx(.5)
```

</details>

### `test_winner_uses_validation_balanced_accuracy_only` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_winner_uses_validation_balanced_accuracy_only():
    from skin_pixel_experiment import choose
    items = [dict(id='a', validation=dict(balanced_accuracy=.8), test=dict(accuracy=1.0)),
             dict(id='b', validation=dict(balanced_accuracy=.9), test=dict(accuracy=0.0))]
    assert choose(items)['id'] == 'b'
```

</details>

### `test_metric_boundary_rejects_bad_probabilities` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_metric_boundary_rejects_bad_probabilities():
    from skin_pixel_experiment import metrics
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([np.nan, .8]), .5)
    with pytest.raises(ValueError):
        metrics(np.array([0, 1]), np.array([0, 1.1]), .5)
```

</details>
