# `tests/test_skin_nuisance.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_nuisance.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `316889db2efa546800a569796f9b198edc2fba0fceca5eee009e33d446abe44c`. Строк: **30**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
from skin_nuisance_model import NuisanceColor, ARMS, control_residual
from skin_spatial_model import SpatialColor,grid_adjacency
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_every_control_has_exact_plain_inference` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_nuisance.py#L9) |
| `test_learned_control_matches_original_training_operator` | FunctionDef | См. реализацию | [L18](../../../../tests/test_skin_nuisance.py#L18) |
| `test_fixed_operators_preserve_constants_and_global_is_permutation_equivariant` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_nuisance.py#L24) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_every_control_has_exact_plain_inference` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_every_control_has_exact_plain_inference():
    x=torch.rand(2,64,18)
    torch.manual_seed(71);plain=SpatialColor('plain').eval()
    for arm in ARMS:
        torch.manual_seed(71);m=NuisanceColor(arm).eval()
        torch.testing.assert_close(m(x)[0],plain(x)[0],atol=0,rtol=0)
        assert m.active_parameters()==924932
```

</details>

### `test_learned_control_matches_original_training_operator` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_learned_control_matches_original_training_operator():
    torch.manual_seed(72);m=NuisanceColor('learned').train()
    torch.manual_seed(72);original=SpatialColor('graph3').train();x=torch.rand(2,64,18)
    torch.testing.assert_close(m(x)[0],original(x)[0],atol=0,rtol=0)
```

</details>

### `test_fixed_operators_preserve_constants_and_global_is_permutation_equivariant` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fixed_operators_preserve_constants_and_global_is_permutation_equivariant():
    torch.manual_seed(901);h=torch.randn(2,64,9,dtype=torch.float64);constant=h[:,:1].expand_as(h);adj=grid_adjacency().double()
    for kind in ['fixed_grid','global']:
        torch.testing.assert_close(control_residual(constant,kind,adj),torch.zeros_like(h),atol=1e-6,rtol=0)
    perm=torch.randperm(64)
    torch.testing.assert_close(control_residual(h[:,perm],'global',adj),control_residual(h,'global',adj)[:,perm],atol=1e-6,rtol=0)
    assert torch.equal(control_residual(h,'bias',adj),torch.full_like(h,.05))
```

</details>
