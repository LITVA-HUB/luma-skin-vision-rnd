# `tests/test_skin_face_segment.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_segment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `ffc74b1d2b44c11b43fe92c7dad1e0dd208873fd12d78e1ae0624ac054f31767`. Строк: **50**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_segmentation_shape_gradients_and_own_state_reload` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_face_segment.py#L12) |
| `test_skin_targets_retain_nose_and_exclude_other_classes` | FunctionDef | См. реализацию | [L30](../../../../tests/test_skin_face_segment.py#L30) |
| `test_overlap_scores_use_confusion_counts_and_defined_empty_case` | FunctionDef | См. реализацию | [L38](../../../../tests/test_skin_face_segment.py#L38) |
| `test_deployment_parameter_budget` | FunctionDef | См. реализацию | [L47](../../../../tests/test_skin_face_segment.py#L47) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_segmentation_shape_gradients_and_own_state_reload` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_segmentation_shape_gradients_and_own_state_reload():
    from skin_face_segment import SkinUNet, skin_loss
    torch.manual_seed(123)
    model = SkinUNet(width=4)
    x = torch.rand(2,3,32,48)
    y = torch.randint(0,2,(2,1,32,48)).float()
    output = model(x)
    assert output.shape == y.shape
    loss = skin_loss(output,y)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = SkinUNet(width=4)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone(x),output,rtol=0,atol=0)
    with pytest.raises(ValueError, match='multiple'):
        model(torch.rand(1,3,33,32))
```

</details>

### `test_skin_targets_retain_nose_and_exclude_other_classes` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_skin_targets_retain_nose_and_exclude_other_classes():
    from skin_face_segment import binary_mask
    target = binary_mask(np.arange(11,dtype=np.uint8))
    assert target.tolist() == [0,1,0,0,0,0,1,0,0,0,0]
    with pytest.raises(ValueError, match='labels'):
        binary_mask(np.array([12]))
```

</details>

### `test_overlap_scores_use_confusion_counts_and_defined_empty_case` · L38

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_overlap_scores_use_confusion_counts_and_defined_empty_case():
    from skin_face_segment import scores
    r = scores(tp=8,fp=2,fn=4,tn=10)
    assert r['iou'] == pytest.approx(8/14)
    assert r['dice'] == pytest.approx(16/22)
    assert r['precision'] == .8
    assert scores(0,0,0,10)['iou'] == 1
```

</details>

### `test_deployment_parameter_budget` · L47

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_deployment_parameter_budget():
    from skin_face_segment import SkinUNet
    count = sum(p.numel() for p in SkinUNet().parameters())
    assert 4_000_000 < count < 5_000_000
```

</details>
