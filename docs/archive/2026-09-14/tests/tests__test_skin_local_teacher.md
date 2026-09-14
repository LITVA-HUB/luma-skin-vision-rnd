# `tests/test_skin_local_teacher.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_teacher.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `c0d450165cedf3d3cc00f9d4dc8b02795621d3a91062d3eb295d628845398f04`. Строк: **44**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from skin_local_teacher_features import pool_tokens, image_permutation
from skin_local_teacher_model import LocalTeacherColor
from skin_capture_model import CaptureColor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_pool_keeps_exact_two_by_two_spatial_correspondence` | FunctionDef | См. реализацию | [L11](../../../../tests/test_skin_local_teacher.py#L11) |
| `test_per_image_control_is_stable_and_preserves_every_token` | FunctionDef | См. реализацию | [L20](../../../../tests/test_skin_local_teacher.py#L20) |
| `test_no_teacher_matches_original_mixture_exactly` | FunctionDef | См. реализацию | [L27](../../../../tests/test_skin_local_teacher.py#L27) |
| `test_alignment_modes_have_identical_parameter_shapes_and_raw_rgb` | FunctionDef | См. реализацию | [L36](../../../../tests/test_skin_local_teacher.py#L36) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_pool_keeps_exact_two_by_two_spatial_correspondence` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pool_keeps_exact_two_by_two_spatial_correspondence():
    grid = (torch.arange(16)[:, None]*100+torch.arange(16)[None, :]).float()
    tokens = grid.reshape(1, 256, 1).expand(2, -1, 384)
    pooled = pool_tokens(tokens)
    expected = torch.tensor([[100*(2*r+.5)+(2*c+.5) for c in range(8)] for r in range(8)]).flatten()
    assert pooled.shape == (2, 64, 384)
    torch.testing.assert_close(pooled[0, :, 0], expected, atol=0, rtol=0)
```

</details>

### `test_per_image_control_is_stable_and_preserves_every_token` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_per_image_control_is_stable_and_preserves_every_token():
    a = image_permutation('example-a'); b = image_permutation('example-b')
    np.testing.assert_array_equal(np.sort(a), np.arange(64))
    np.testing.assert_array_equal(a, image_permutation('example-a'))
    assert not np.array_equal(a, b)
```

</details>

### `test_no_teacher_matches_original_mixture_exactly` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_no_teacher_matches_original_mixture_exactly():
    torch.manual_seed(17); base = CaptureColor('mixture').eval()
    torch.manual_seed(17); model = LocalTeacherColor('plain').eval()
    x = torch.rand(2, 64, 402)
    with torch.no_grad():
        for a, b in zip(base(x[..., :18].contiguous()), model(x), strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
```

</details>

### `test_alignment_modes_have_identical_parameter_shapes_and_raw_rgb` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_alignment_modes_have_identical_parameter_shapes_and_raw_rgb():
    states = []
    for arm in ['plain', 'aligned', 'global', 'shuffled']:
        torch.manual_seed(17); model = LocalTeacherColor(arm); states.append(model.state_dict())
    assert all(all(torch.equal(states[0][k], v) for k, v in s.items()) for s in states)
    torch.manual_seed(5); x = torch.randn(2, 64, 402)
    model = LocalTeacherColor('global'); global_tokens = model.teacher_tokens(x)
    torch.testing.assert_close(global_tokens[:, 0], global_tokens[:, -1], atol=0, rtol=0)
    torch.testing.assert_close(global_tokens[:, 0], x[..., 18:].mean(1), atol=0, rtol=0)
```

</details>
