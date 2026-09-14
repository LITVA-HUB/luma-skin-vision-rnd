# `tests/test_chromaseed_training.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_training.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `1ede7707dd330b4d99b5cf323d45fecafe3b1627c48de64fb4552b93e41efec0`. Строк: **42**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_training_replays_identical_init_batches_and_reduces_loss` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_training.py#L10) |
| `test_checkpoint_and_weight_contract_rejects_invalid_inputs` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed_training.py#L31) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_training_replays_identical_init_batches_and_reduces_loss` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_training_replays_identical_init_batches_and_reduces_loss():
    from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD, ChromaSeed, pack_model, predict
    from chromaseed_train import train_trace

    torch.set_num_threads(1)
    torch.manual_seed(17)
    initial = pack_model(ChromaSeed())
    rng = np.random.default_rng(167)
    x = (X_MEAN + rng.normal(0, .2, (48, 36)) * X_STD).astype(np.float32)
    target = (Y_MEAN + np.column_stack((x[:, 0], x[:, 1], x[:, 2])) * Y_STD).astype(np.float32)
    weights = np.ones(48)
    models, receipt = train_trace(initial, x, target, weights, 17, .003, 80, (20, 80), "cpu")
    again, _ = train_trace(initial, x, target, weights, 17, .003, 80, (20, 80), "cpu")
    np.testing.assert_array_equal(predict(models[80], x), predict(again[80], x))
    original_error = np.mean(((predict(initial, x) - target) / Y_STD) ** 2)
    assert receipt["checkpoints"]["80"]["weighted_normalized_mse"] < original_error / 10
    assert set(models) == {20, 80}
    assert receipt["steps"] == 80 and receipt["batch_size"] == 256
    assert receipt["checkpoints"]["80"]["seconds"] >= receipt["checkpoints"]["20"]["seconds"]
```

</details>

### `test_checkpoint_and_weight_contract_rejects_invalid_inputs` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_checkpoint_and_weight_contract_rejects_invalid_inputs():
    import pytest
    from chromaseed import ChromaSeed, pack_model
    from chromaseed_train import train_trace

    initial = pack_model(ChromaSeed())
    x = np.ones((6, 36), np.float32)
    y = np.zeros((6, 3), np.float32)
    with pytest.raises(ValueError):
        train_trace(initial, x, y, np.zeros(6), 17, .001, 8, (8,), "cpu")
    with pytest.raises(ValueError):
        train_trace(initial, x, y, np.ones(6), 17, .001, 8, (9,), "cpu")
```

</details>
