# `tests/test_skin_train_branch.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_train_branch.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `0165b2d3e53f3e7ca5a56a2d072909fda61e9b86d16dc71d9a997a782e945667`. Строк: **29**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
from skin_train_branch_model import TrainingBranchColor, ARMS
from skin_spatial_model import SpatialColor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_all_strategies_deploy_exactly_plain_for_same_weights` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_train_branch.py#L9) |
| `test_always_branch_matches_parent_training` | FunctionDef | См. реализацию | [L18](../../../../tests/test_skin_train_branch.py#L18) |
| `test_evaluation_does_not_consume_drop_rng_or_leave_steps_changed` | FunctionDef | См. реализацию | [L26](../../../../tests/test_skin_train_branch.py#L26) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_all_strategies_deploy_exactly_plain_for_same_weights` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_all_strategies_deploy_exactly_plain_for_same_weights():
    x=torch.rand(2,64,18)
    torch.manual_seed(51);plain=SpatialColor('plain').eval()
    for arm in ARMS:
        torch.manual_seed(51);m=TrainingBranchColor(arm).eval()
        torch.testing.assert_close(m(x)[0],plain(x)[0],atol=0,rtol=0)
        assert m.active_parameters()==924932
```

</details>

### `test_always_branch_matches_parent_training` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_always_branch_matches_parent_training():
    x=torch.rand(2,64,18)
    for kind in ['graph','conv']:
        torch.manual_seed(52);parent=SpatialColor(kind+'3').train()
        torch.manual_seed(52);m=TrainingBranchColor(kind+'_always').train()
        torch.testing.assert_close(m(x)[0],parent(x)[0],atol=0,rtol=0)
```

</details>

### `test_evaluation_does_not_consume_drop_rng_or_leave_steps_changed` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_evaluation_does_not_consume_drop_rng_or_leave_steps_changed():
    torch.manual_seed(53);m=TrainingBranchColor('graph_drop').eval();x=torch.rand(2,64,18)
    before=torch.get_rng_state().clone();steps=m.steps;m(x)
    assert torch.equal(torch.get_rng_state(),before) and m.steps==steps
```

</details>
