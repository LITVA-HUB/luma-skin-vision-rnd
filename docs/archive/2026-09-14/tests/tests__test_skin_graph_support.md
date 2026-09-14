# `tests/test_skin_graph_support.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_graph_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5db3cfe1653e7d40d11598b2f932bde838a2e58c58baab09b4edc66f76f82d8d`. Строк: **22**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
from skin_graph_support_model import GraphSupportColor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_inference_is_identical_core_for_all_branch_flags` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_graph_support.py#L8) |
| `test_training_branch_receives_gradient_only_when_requested` | FunctionDef | См. реализацию | [L16](../../../../tests/test_skin_graph_support.py#L16) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_inference_is_identical_core_for_all_branch_flags` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_inference_is_identical_core_for_all_branch_flags():
    torch.manual_seed(17);model=GraphSupportColor().eval();x=torch.rand(2,64,18)
    a=model(x,branch=False)[0];b=model(x,branch=True)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    perm=torch.randperm(64)
    torch.testing.assert_close(a,model(x[:,perm],branch=True)[0],rtol=1e-5,atol=1e-7)
```

</details>

### `test_training_branch_receives_gradient_only_when_requested` · L16

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_training_branch_receives_gradient_only_when_requested():
    model=GraphSupportColor().train();x=torch.rand(2,64,18)
    model(x,branch=False)[0].square().sum().backward()
    assert model.relation.weight.grad is None
    model.zero_grad(set_to_none=True);model(x,branch=True)[0].square().sum().backward()
    assert model.relation.weight.grad is not None and model.relation.weight.grad.abs().sum()>0
    assert model.steps==3
```

</details>
