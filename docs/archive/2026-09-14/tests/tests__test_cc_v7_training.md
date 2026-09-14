# `tests/test_cc_v7_training.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v7_training.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5c05ad3693b1de5f3542fbcd1074b9388c04e8ec49f83fc1872e78be4a602552`. Строк: **46**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
from cc_v7_experiment import augment_batch, distillation_loss, point_loss
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_sensor_and_native_share_exposure_flip_and_native_target_orientation` | FunctionDef | См. реализацию | [L10](../../../../tests/test_cc_v7_training.py#L10) |
| `test_distillation_gradient_does_not_flow_into_teacher` | FunctionDef | См. реализацию | [L24](../../../../tests/test_cc_v7_training.py#L24) |
| `test_point_loss_accepts_rounded_normalized_boundary_and_has_finite_gradient` | FunctionDef | См. реализацию | [L34](../../../../tests/test_cc_v7_training.py#L34) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_sensor_and_native_share_exposure_flip_and_native_target_orientation` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_sensor_and_native_share_exposure_flip_and_native_target_orientation():
    x=torch.arange(2*3*4*4,dtype=torch.float32).reshape(2,3,4,4)+1
    gt=torch.ones(2,3)
    def draw(sensor):
        return augment_batch(x,gt,torch.Generator().manual_seed(8),torch.Generator().manual_seed(9),sensor)
    native=draw(False)
    sensor=draw(True)
    torch.testing.assert_close(native[2],sensor[2],rtol=0,atol=0)
    # The same sensor transform must explain both the input and its GT label.
    expected=torch.einsum("nij,njhw->nihw",sensor[3],native[0])
    torch.testing.assert_close(sensor[0],expected)
    torch.testing.assert_close(sensor[1],torch.nn.functional.normalize(torch.einsum("nij,nj->ni",sensor[3],gt),dim=-1))
```

</details>

### `test_distillation_gradient_does_not_flow_into_teacher` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_distillation_gradient_does_not_flow_into_teacher():
    student=torch.randn(2,16,384,requires_grad=True)
    teacher=torch.randn(2,16,384,requires_grad=True)
    loss=distillation_loss(student,teacher)
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    torch.testing.assert_close(distillation_loss(student,student),torch.tensor(0.),atol=1e-6,rtol=0)
```

</details>

### `test_point_loss_accepts_rounded_normalized_boundary_and_has_finite_gradient` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_point_loss_accepts_rounded_normalized_boundary_and_has_finite_gradient():
    z=torch.stack((torch.full((10001,),2.),torch.full((10001,),-2.),torch.linspace(-2,2,10001)),1)
    pred=torch.nn.functional.normalize(z.exp(),dim=-1).requires_grad_()
    action=pred.log()[:,0]-pred.log()[:,1]
    assert (action>4).any()  # FP32 normalize/log roundtrip can exceed 4 by one ULP.
    gt=torch.ones_like(pred)
    loss=point_loss(pred,gt)
    ratio=(gt/pred).double()
    neutral=torch.ones_like(ratio)
    exact=torch.atan2(torch.linalg.cross(ratio,neutral).norm(dim=-1),(ratio*neutral).sum(-1)).mean()*180/torch.pi
    torch.testing.assert_close(loss.double(),exact,atol=1e-5,rtol=0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
```

</details>
