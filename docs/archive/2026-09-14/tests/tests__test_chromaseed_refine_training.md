# `tests/test_chromaseed_refine_training.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_refine_training.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `16af05eff838051787c08032eb8f9dd46f63b5fcae4122b84df6e1d8d4b53080`. Строк: **55**.

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
| `test_bank_batches_replay_across_learning_rates_and_preserve_weighting` | FunctionDef | См. реализацию | [L11](../../../../tests/test_chromaseed_refine_training.py#L11) |
| `test_small_training_improves_nonlinear_fit_and_replays_exactly` | FunctionDef | См. реализацию | [L21](../../../../tests/test_chromaseed_refine_training.py#L21) |
| `test_cuda_graph_matches_eager_updates_and_checkpoint_budget` | FunctionDef | См. реализацию | [L42](../../../../tests/test_chromaseed_refine_training.py#L42) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_bank_batches_replay_across_learning_rates_and_preserve_weighting` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bank_batches_replay_across_learning_rates_and_preserve_weighting():
    from chromaseed_refine_train import sampling_indices

    weights = np.array([1., 3., 6.])
    sampled = sampling_indices(weights, [17, 29, 17], 256)
    np.testing.assert_array_equal(sampled[0], sampled[2])
    assert not np.array_equal(sampled[0], sampled[1])
    np.testing.assert_allclose(np.bincount(sampled[0].ravel(), minlength=3) / sampled[0].size, [.1, .3, .6], atol=.015)
```

</details>

### `test_small_training_improves_nonlinear_fit_and_replays_exactly` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_small_training_improves_nonlinear_fit_and_replays_exactly():
    from chromaseed_refine_train import predict_bank, train_bank

    torch.set_num_threads(1)
    rng = np.random.default_rng(57)
    x = rng.normal(size=(72, 36)).astype(np.float32)
    patches = rng.normal(size=(72, 64, 18)).astype(np.float32)
    y = np.column_stack((50 + 5 * x[:, 0] ** 2, 4 + 3 * x[:, 1] ** 2, 10 + 4 * x[:, 2] ** 2)).astype(np.float32)
    models, receipt = train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003), (29, .003)], 80, (1, 80), "cpu")
    early, _ = predict_bank(models[1], x, patches, "cpu")
    late, _ = predict_bank(models[80], x, patches, "cpu")
    assert np.mean((late[:, :, -1] - y) ** 2) < .7 * np.mean((early[:, :, -1] - y) ** 2)
    again, _ = train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003), (29, .003)], 80, (1, 80), "cpu")
    np.testing.assert_array_equal(models[80]["theta"], again[80]["theta"])
    assert receipt["steps"] == 80 and receipt["batch_size"] == 64
    with pytest.raises(ValueError):
        train_bank(x, patches, y, np.ones(len(x)), "stats_mlp", [(17, .003)], 8, (9,), "cpu")
```

</details>

### `test_cuda_graph_matches_eager_updates_and_checkpoint_budget` · L42

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph equivalence requires a CUDA GPU')
@pytest.mark.parametrize('family', ['stats_mlp', 'patch_mlp', 'recur_soft', 'recur_top16', 'recur_dynamic'])
def test_cuda_graph_matches_eager_updates_and_checkpoint_budget(family):
    from chromaseed_refine_train import setup, train_bank

    setup("cuda")
    rng = np.random.default_rng(351)
    x = rng.normal(size=(48, 36)).astype(np.float32)
    patches = rng.normal(size=(48, 64, 18)).astype(np.float32)
    y = (rng.normal(size=(48, 3)) * [10, 3, 5] + [50, 4, 10]).astype(np.float32)
    arguments = (x, patches, y, np.ones(len(x)), family, [(17, .0003), (29, .001), (17, .003)], 64, (1, 16, 64), "cuda")
    eager, _ = train_bank(*arguments, engine="eager")
    graph, receipt = train_bank(*arguments, engine="cuda_graph")
    for checkpoint in (1, 16, 64):
        np.testing.assert_allclose(graph[checkpoint]["theta"], eager[checkpoint]["theta"], atol=2e-6, rtol=2e-5)
    assert receipt["steps"] == 64 and receipt["engine"] == "cuda_graph"
```

</details>
