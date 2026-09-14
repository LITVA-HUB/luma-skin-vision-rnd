# `tests/test_chromaseed_neural_blocks.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_neural_blocks.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Retained training must preserve block identities, noise and local credit.

SHA-256 исходника: `2a9848743e9336b2350b17e759f7d9b7902684f70c52bb6fdf9d3ccc9893e458`. Строк: **133**.

## Зависимости

```python
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `mod` | FunctionDef | См. реализацию | [L15](../../../../tests/test_chromaseed_neural_blocks.py#L15) |
| `data` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_neural_blocks.py#L20) |
| `test_initialization_uses_original_block_and_seed` | FunctionDef | См. реализацию | [L30](../../../../tests/test_chromaseed_neural_blocks.py#L30) |
| `test_noise_retains_original_k_coordinates_not_shortened_rng_shape` | FunctionDef | См. реализацию | [L43](../../../../tests/test_chromaseed_neural_blocks.py#L43) |
| `test_first_local_block_keeps_trainable_noise_input_rows` | FunctionDef | См. реализацию | [L53](../../../../tests/test_chromaseed_neural_blocks.py#L53) |
| `test_raw_export_rejects_wrong_original_identity` | FunctionDef | См. реализацию | [L64](../../../../tests/test_chromaseed_neural_blocks.py#L64) |
| `test_invalid_or_coupled_subset_is_rejected` | FunctionDef | См. реализацию | [L87](../../../../tests/test_chromaseed_neural_blocks.py#L87) |
| `test_cpu_subset_training_matches_full_and_np_export` | FunctionDef | См. реализацию | [L93](../../../../tests/test_chromaseed_neural_blocks.py#L93) |
| `test_cuda_graph_reset_and_full_subset_semantics` | FunctionDef | См. реализацию | [L114](../../../../tests/test_chromaseed_neural_blocks.py#L114) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_initialization_uses_original_block_and_seed` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family,j,indices', [('local2', 1, [0]), ('local4', 3, [0, 1, 2]), ('blind4', 3, [2])])
def test_initialization_uses_original_block_and_seed(family, j, indices):
    from chromaseed_local_denoise import Bank

    m = mod()
    actual = m.RetainedBank(family, [29, 17], j)
    full = Bank(family, [29, 17])
    np.testing.assert_array_equal(actual.block_indices, indices)
    np.testing.assert_array_equal(
        actual.theta.detach().numpy().reshape(2, len(indices), -1),
        full.theta.detach().numpy().reshape(2, full.k, -1)[:, indices],
    )
```

</details>

### `test_noise_retains_original_k_coordinates_not_shortened_rng_shape` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_noise_retains_original_k_coordinates_not_shortened_rng_shape():
    from chromaseed_local_denoise_fit import noise_sequences

    m = mod()
    wanted = noise_sequences([17, 29, 17], 7, 4)[:, :, [0, 1]]
    actual = m.retained_noise([17, 29, 17], 7, 4, [0, 1])
    np.testing.assert_array_equal(actual, wanted)
    assert not np.array_equal(actual, noise_sequences([17, 29, 17], 7, 2))
```

</details>

### `test_first_local_block_keeps_trainable_noise_input_rows` · L53

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_first_local_block_keeps_trainable_noise_input_rows():
    m = mod()
    net = m.RetainedBank("local4", [17], 1)
    net.local(torch.ones(1, 64, 36), torch.ones(1, 1, 64, 3)).square().mean().backward()
    assert net.d == 39
    assert torch.count_nonzero(net.theta.grad[0, : 39 * 16].reshape(39, 16)[36:]) > 0
```

</details>

### `test_raw_export_rejects_wrong_original_identity` · L64

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('field,value', [('prefix', np.array(1.5)), ('block_indices', np.array([1], np.uint8))])
def test_raw_export_rejects_wrong_original_identity(field, value):
    from chromaseed_local_denoise import preprocessor

    m = mod()
    x, y, _ = data()
    raw = m.RetainedBank("local4", [17], 1).export(0, preprocessor(x, y))
    raw[field] = value
    with pytest.raises(ValueError):
        m.to_prefix(raw)
```

</details>

### `test_invalid_or_coupled_subset_is_rejected` · L87

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family,j', [('e2e4', 1), ('e2e4', 4), ('plain', 1), ('local4', 0), ('local4', 5), ('local4', 1.5), ('local4', True)])
def test_invalid_or_coupled_subset_is_rejected(family, j):
    with pytest.raises(ValueError):
        mod().RetainedBank(family, [17], j)
```

</details>

### `test_cpu_subset_training_matches_full_and_np_export` · L93

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family,j', [('local2', 1), ('local4', 2), ('blind4', 3)])
def test_cpu_subset_training_matches_full_and_np_export(family, j):
    from chromaseed_local_denoise_fit import fit as full_fit
    from chromaseed_neural_prefix_numpy import export_prefix, predict

    m = mod()
    x, y, w = data()
    full, _ = full_fit(x, y, w, family, [(17, 0.003), (29, 0.001)], 12, (4, 12))
    subset, _ = m.fit(x, y, w, family, [(17, 0.003), (29, 0.001)], j, 12, (4, 12))
    for step in (4, 12):
        for slot in range(2):
            raw = subset[step][slot]
            np.testing.assert_allclose(
                raw["theta"], full[step][slot]["theta"][raw["block_indices"]], rtol=2e-6, atol=2e-6
            )
            expected = export_prefix(full[step][slot], j)
            actual = m.to_prefix(raw)
            np.testing.assert_allclose(predict(actual, x), predict(expected, x), rtol=0, atol=0.002)
```

</details>

### `test_cuda_graph_reset_and_full_subset_semantics` · L114

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('family,j', [('local4', 1), ('local4', 3), ('blind4', 2)])
def test_cuda_graph_reset_and_full_subset_semantics(family, j):
    from chromaseed_local_denoise_fit import fit as full_fit

    m = mod()
    x, y, w = data()
    full, _ = full_fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], 64, (64,), "cuda", "cuda_graph"
    )
    graph, _ = m.fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], j, 64, (64,), "cuda", "cuda_graph"
    )
    eager, _ = m.fit(
        x, y, w, family, [(17, 0.003), (29, 0.001), (43, 0.003)], j, 64, (64,), "cuda", "eager"
    )
    for slot in range(3):
        raw = graph[64][slot]
        np.testing.assert_allclose(
            raw["theta"], full[64][slot]["theta"][raw["block_indices"]], rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(raw["theta"], eager[64][slot]["theta"], rtol=2e-6, atol=2e-6)
```

</details>
