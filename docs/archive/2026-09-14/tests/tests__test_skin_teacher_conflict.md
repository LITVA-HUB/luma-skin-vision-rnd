# `tests/test_skin_teacher_conflict.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_teacher_conflict.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `8711f7da4587540c9b9a265e057a88f2473644bf5ede2294e579e7d8045b35ea`. Строк: **23**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
import torch
from skin_teacher_conflict import gradient_relation
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_opposing_auxiliary_step_increases_primary_loss_to_first_order` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_teacher_conflict.py#L9) |
| `test_unused_auxiliary_parameters_are_zero_not_conflict` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_teacher_conflict.py#L19) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_opposing_auxiliary_step_increases_primary_loss_to_first_order` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_opposing_auxiliary_step_increases_primary_loss_to_first_order():
    x = torch.tensor(.5, requires_grad=True)
    color = torch.autograd.grad(x.square(), x, retain_graph=True)
    aux = torch.autograd.grad(.1*(x-2).square(), x)
    r = gradient_relation(color, aux)
    assert r['cosine'] == pytest.approx(-1)
    assert r['weighted_aux_to_color_norm_ratio'] == pytest.approx(.3)
    assert r['color_derivative_along_negative_aux_gradient'] == pytest.approx(.3)
```

</details>

### `test_unused_auxiliary_parameters_are_zero_not_conflict` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_unused_auxiliary_parameters_are_zero_not_conflict():
    r = gradient_relation([torch.tensor([1., 2.])], [None])
    assert r['cosine'] is None
    assert r['weighted_aux_to_color_norm_ratio'] == 0
    assert r['color_derivative_along_negative_aux_gradient'] == 0
```

</details>
