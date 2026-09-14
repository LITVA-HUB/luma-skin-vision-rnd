# `tests/test_chromaseed_perceptual_protocol.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_perceptual_protocol.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `316572ab7a2e10b98574f6a9952ff5fa55927dc4f583d14e08fd7c98c17ef59b`. Строк: **21**.

## Зависимости

```python
import sys
from pathlib import Path
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_iterative_selection_includes_unchanged_control_without_new_model` | FunctionDef | См. реализацию | [L7](../../../../tests/test_chromaseed_perceptual_protocol.py#L7) |
| `test_selection_prefers_fewer_corrections_on_equal_inner_quality` | FunctionDef | См. реализацию | [L18](../../../../tests/test_chromaseed_perceptual_protocol.py#L18) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_iterative_selection_includes_unchanged_control_without_new_model` · L7

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_iterative_selection_includes_unchanged_control_without_new_model():
    from chromaseed_perceptual import key
    from chromaseed_perceptual_train import candidate_grid
    grid = list(candidate_grid("midpoint_irls"))
    assert len(grid) == 36
    zero = [item for item in grid if item[2] == 0]
    assert len(zero) == 9
    for wi, ai, steps in zero:
        assert key("midpoint_irls", 17, wi, ai, steps) == key("norm_mse", 17, wi, ai, 0)
```

</details>

### `test_selection_prefers_fewer_corrections_on_equal_inner_quality` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_prefers_fewer_corrections_on_equal_inner_quality():
    from chromaseed_perceptual_train import choose_candidate
    candidates = [dict(person_mean=4., steps=s, alpha=a, width_factor=.5) for s, a in [(16, .1), (1, 1.), (0, 10.)]]
    assert choose_candidate(candidates)["steps"] == 0
```

</details>
