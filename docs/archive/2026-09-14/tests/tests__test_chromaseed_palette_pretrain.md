# `tests/test_chromaseed_palette_pretrain.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_palette_pretrain.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5781cb8b69ff67f20f11266e06703e3b6862d3c83341cf2173a7e3315b0ea9a0`. Строк: **70**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_patch_token_layout_matches_native_image_pipeline` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_palette_pretrain.py#L10) |
| `test_spatial_sampling_requires_whole_patch_and_fails_without_enough` | FunctionDef | См. реализацию | [L23](../../../../tests/test_chromaseed_palette_pretrain.py#L23) |
| `test_encoder_transfer_preserves_raw_input_function_and_as_initialization` | FunctionDef | См. реализацию | [L34](../../../../tests/test_chromaseed_palette_pretrain.py#L34) |
| `test_cpu_fitter_uses_paired_inputs_and_distinct_aligned_targets` | FunctionDef | См. реализацию | [L57](../../../../tests/test_chromaseed_palette_pretrain.py#L57) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_patch_token_layout_matches_native_image_pipeline` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_patch_token_layout_matches_native_image_pipeline():
    from chromaseed_palette_data import patch_tokens
    from skin_mskcc_pixels import features

    rng = np.random.default_rng(814)
    patch = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    image = np.tile(patch, (8, 8, 1))
    expected = features(image)['tokens'][0]
    np.testing.assert_allclose(patch_tokens(patch[None])[0], expected, atol=1e-14, rtol=0)
    constant = np.full((1, 16, 16, 3), 160, np.uint8)
    np.testing.assert_allclose(patch_tokens(constant)[0, 12:], 0, atol=1e-14)
```

</details>

### `test_spatial_sampling_requires_whole_patch_and_fails_without_enough` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_spatial_sampling_requires_whole_patch_and_fails_without_enough():
    from chromaseed_palette_data import spatial_centres

    mask = np.ones((64, 64), bool)
    mask[22:24, 22:24] = False
    centres = np.array([[1, 1], [24, 24], [40, 40], [45, 45]])
    np.testing.assert_array_equal(spatial_centres(mask, centres, cap=2), centres[2:])
    with pytest.raises(ValueError):
        spatial_centres(mask, centres, cap=3)
```

</details>

### `test_encoder_transfer_preserves_raw_input_function_and_as_initialization` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_encoder_transfer_preserves_raw_input_function_and_as_initialization():
    import torch
    from chromaseed_architecture_scale import Bank
    from chromaseed_palette_encoder import EncoderBank, numpy_encode, transplant

    torch.set_num_threads(1)
    encoder = EncoderBank()
    parent = Bank('patch5m')
    prefix = 105856
    np.testing.assert_array_equal(encoder.theta.detach().numpy()[:, :prefix],
                                  parent.theta.detach().numpy()[:, :prefix])
    rng = np.random.default_rng(91)
    mean, std = rng.normal(size=18), rng.uniform(.5, 2, 18)
    native_mean, native_std = rng.normal(size=18), rng.uniform(.5, 2, 18)
    model = encoder.export(0, mean, std)
    updated = transplant(model, native_mean, native_std)
    x = rng.normal(size=(13, 18))
    a, b = numpy_encode(model, x), numpy_encode(updated, x)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-6)
    actual = encoder.encode(torch.tensor(((x-mean)/std)[None].repeat(6, 0), dtype=torch.float32))[0]
    np.testing.assert_allclose(a, actual.detach().numpy(), atol=2e-6, rtol=1e-6)
```

</details>

### `test_cpu_fitter_uses_paired_inputs_and_distinct_aligned_targets` · L57

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cpu_fitter_uses_paired_inputs_and_distinct_aligned_targets():
    import torch
    from chromaseed_palette_pretrain import fit_arrays

    torch.set_num_threads(1)
    rng = np.random.default_rng(155)
    x = rng.normal(size=(32, 18)).astype(np.float32)
    target = rng.normal(size=(32, 36))
    _, state, info = fit_arrays(x, target, target[::-1].copy(), steps=2)
    assert info['steps'] == 2 and info['cuda_initialized'] is False
    assert state['sampling'].shape == (3, 2, 128)
    np.testing.assert_array_equal(state['initial_theta'][0], state['initial_theta'][1])
    assert not np.array_equal(state['theta'][0], state['theta'][1])
    assert np.isfinite(state['theta']).all()
```

</details>
