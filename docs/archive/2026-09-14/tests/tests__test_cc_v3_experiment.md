# `tests/test_cc_v3_experiment.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v3_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-role isolation and training-loss checks for the V3 experiment.

SHA-256 исходника: `4672d8dd65d17be77ea57658c572141f9073bae4c5f38dfad0962aedd14ffa15`. Строк: **53**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from cc_v3_experiment import point_objective, source_rows, validation_key
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_source_rows_excludes_every_nonfitting_role` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_v3_experiment.py#L14) |
| `test_point_loss_uses_invalid_raw_gradient_and_valid_geometry` | FunctionDef | См. реализацию | [L27](../../../../tests/test_cc_v3_experiment.py#L27) |
| `test_invalid_predictions_cannot_win_checkpoint_selection` | FunctionDef | См. реализацию | [L39](../../../../tests/test_cc_v3_experiment.py#L39) |
| `test_untrained_epoch_metrics_are_unmeasured` | FunctionDef | См. реализацию | [L45](../../../../tests/test_cc_v3_experiment.py#L45) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_source_rows_excludes_every_nonfitting_role` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_source_rows_excludes_every_nonfitting_role():
    rows = [
        {"id": str(i), "group": str(i), "subset": role}
        for i, role in enumerate(["test", "train", "risk", "val", "cal", "train"])
    ]
    selected, train, val = source_rows(rows)
    assert selected.tolist() == [1, 3, 5]
    assert train.tolist() == [0, 2] and val.tolist() == [1]
    rows[3]["group"] = rows[1]["group"]
    with pytest.raises(ValueError, match="group"):
        source_rows(rows)
```

</details>

### `test_point_loss_uses_invalid_raw_gradient_and_valid_geometry` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_point_loss_uses_invalid_raw_gradient_and_valid_geometry():
    raw = torch.tensor([[-0.1, 0.7, 0.5], [0.3, 0.7, 0.5]], requires_grad=True)
    gt = torch.tensor([[0.4, 0.6, 0.5], [0.4, 0.6, 0.5]])
    task, positive = point_objective(raw, gt)
    (task + 10 * positive).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0, 0] < 0
    corrected = gt[1].numpy() / raw[1].detach().numpy()
    angle = np.degrees(np.arccos(corrected.sum() / (np.sqrt(3) * np.linalg.norm(corrected))))
    actual, _ = point_objective(raw[1:].detach(), gt[1:])
    assert actual.item() == pytest.approx(angle, abs=1e-4)
```

</details>

### `test_invalid_predictions_cannot_win_checkpoint_selection` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_predictions_cannot_win_checkpoint_selection():
    assert validation_key(2.0, 1.0) < validation_key(1.0, 0.98)
    assert validation_key(2.0, 1.0) < validation_key(3.0, 1.0)
    assert np.isinf(validation_key(float("nan"), 1.0))
```

</details>

### `test_untrained_epoch_metrics_are_unmeasured` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_untrained_epoch_metrics_are_unmeasured():
    from cc_v3_experiment import training_summary

    assert training_summary(np.zeros(3), 0) == {
        "train_reproduction": None,
        "train_nll": None,
        "train_positivity": None,
    }
    assert training_summary(np.array([4.0, 8.0, 2.0]), 2)["train_reproduction"] == 2.0
```

</details>
