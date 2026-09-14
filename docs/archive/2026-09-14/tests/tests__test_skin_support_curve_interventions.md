# `tests/test_skin_support_curve_interventions.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_support_curve_interventions.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d12cf50f28db29d8a5f76e7891b0e54db72f937749c137d0fc935c1d6559d58f`. Строк: **20**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
from skin_support_curve_interventions import intervene
from skin_support_curve import pixel_patches
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_shuffle_preserves_each_patch_color_multiset` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_support_curve_interventions.py#L9) |
| `test_mean_changes_only_within_patch_distribution` | FunctionDef | См. реализацию | [L16](../../../../tests/test_skin_support_curve_interventions.py#L16) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_shuffle_preserves_each_patch_color_multiset` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_shuffle_preserves_each_patch_color_multiset():
    x=torch.arange(128*128*3).reshape(1,128,128,3).float()
    original=pixel_patches(x).flatten(2);changed=pixel_patches(intervene(x,'shuffle')).flatten(2)
    assert torch.equal(original.sort(2).values,changed.sort(2).values)
    assert not torch.equal(original,changed)
```

</details>

### `test_mean_changes_only_within_patch_distribution` · L16

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_mean_changes_only_within_patch_distribution():
    x=torch.rand(2,128,128,3)
    original=pixel_patches(x);changed=pixel_patches(intervene(x,'mean'))
    torch.testing.assert_close(original.mean((2,3)),changed.mean((2,3)),rtol=1e-6,atol=1e-7)
    assert changed.std((2,3)).max()==0
```

</details>
