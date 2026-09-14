# `tests/test_skin_shared_bias.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_shared_bias.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `9af4fc7f38bdbd4b23a4c6eb2d17ff82772d1b9e704d6ecc602214ec9e9c44da`. Строк: **32**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
import torch
from skin_shared_bias import paired_objective
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_opposing_errors_cancel_only_in_mean_objective` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_shared_bias.py#L9) |
| `test_common_bias_cost_identical_in_every_arm` | FunctionDef | См. реализацию | [L21](../../../../tests/test_skin_shared_bias.py#L21) |
| `test_reference_and_pair_layout_are_checked` | FunctionDef | См. реализацию | [L27](../../../../tests/test_skin_shared_bias.py#L27) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_opposing_errors_cancel_only_in_mean_objective` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_opposing_errors_cancel_only_in_mean_objective():
    p = torch.tensor([[1., 2, 3], [-1., -2, -3]], dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(p)
    ordinary = paired_objective(p, y, 'individual')
    assert ordinary.item() == pytest.approx(14/3)
    assert paired_objective(p, y, 'shared_only').item() == 0
    assert paired_objective(p, y, 'shared_half').item() == pytest.approx(7/3)
    assert paired_objective(p, y, 'consistency').item() == pytest.approx(28/3)
    paired_objective(p, y, 'shared_half').backward()
    torch.testing.assert_close(p.grad, p.detach()/6)
```

</details>

### `test_common_bias_cost_identical_in_every_arm` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_common_bias_cost_identical_in_every_arm():
    p = torch.full((4, 3), 2., dtype=torch.float64)
    y = torch.zeros_like(p)
    assert all(paired_objective(p, y, arm).item() == 4 for arm in ['individual', 'shared_only', 'shared_half', 'consistency'])
```

</details>

### `test_reference_and_pair_layout_are_checked` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_reference_and_pair_layout_are_checked():
    p = torch.zeros(4, 3); y = p.clone(); y[2, 0] = 1
    with pytest.raises(ValueError, match='reference'):
        paired_objective(p, y, 'shared_only')
    with pytest.raises(ValueError, match='even'):
        paired_objective(p[:3], p[:3], 'individual')
```

</details>
