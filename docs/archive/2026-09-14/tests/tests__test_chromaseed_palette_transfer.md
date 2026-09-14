# `tests/test_chromaseed_palette_transfer.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_palette_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Numerical checks for the native initialization intervention; CPU only.

SHA-256 исходника: `35fed77e8751910bfcac7ab0ff7a1374384535d394eea2cbeee440c7d0f62a8d`. Строк: **161**.

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
| `encoders` | FunctionDef | См. реализацию | [L13](../../../../tests/test_chromaseed_palette_transfer.py#L13) |
| `test_transfer_changes_only_encoder_and_preserves_raw_local_function` | FunctionDef | См. реализацию | [L27](../../../../tests/test_chromaseed_palette_transfer.py#L27) |
| `test_transfer_rejects_wrong_seed_arm_or_incompatible_architecture` | FunctionDef | См. реализацию | [L53](../../../../tests/test_chromaseed_palette_transfer.py#L53) |
| `test_original_fit_and_prefix_are_bitwise_hr` | FunctionDef | См. реализацию | [L69](../../../../tests/test_chromaseed_palette_transfer.py#L69) |
| `test_transferred_encoder_finetunes_and_keeps_the_original_sampling` | FunctionDef | См. реализацию | [L92](../../../../tests/test_chromaseed_palette_transfer.py#L92) |
| `test_heads_are_fixed_from_inner_per_architecture_choice` | FunctionDef | См. реализацию | [L121](../../../../tests/test_chromaseed_palette_transfer.py#L121) |
| `test_full_tensor_export_keeps_nonnumeric_provenance_and_parameter_count` | FunctionDef | См. реализацию | [L146](../../../../tests/test_chromaseed_palette_transfer.py#L146) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_transfer_changes_only_encoder_and_preserves_raw_local_function` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_transfer_changes_only_encoder_and_preserves_raw_local_function():
    from chromaseed_head_range import Bank as OriginalBank
    from chromaseed_palette_encoder import ENCODER_PARAMETERS, numpy_encode
    from chromaseed_palette_transfer import Bank
    from torch.nn import functional as F

    rng = np.random.default_rng(10203)
    tokens = rng.uniform(0, 1, (3, 64, 18)).astype(np.float32)
    mean = tokens.astype(float).mean((0, 1)).astype(np.float32)
    std = tokens.astype(float).std((0, 1)).astype(np.float32)
    source = encoders("aligned")
    net = Bank("dynamic5m", "wide", "aligned", source, mean, std)
    old = OriginalBank("dynamic5m", "wide")
    np.testing.assert_array_equal(
        net.theta.detach().numpy()[:, ENCODER_PARAMETERS:],
        old.theta.detach().numpy()[:, ENCODER_PARAMETERS:],
    )
    inputs = torch.from_numpy((tokens - mean) / std)[None].expand(6, -1, -1, -1)
    with torch.no_grad():
        actual = F.silu(net.layers["token2"](F.silu(net.layers["token1"](inputs)))).numpy()
    for slot in range(6):
        expected = numpy_encode(source[slot // 2], tokens)
        np.testing.assert_allclose(actual[slot], expected, atol=2e-5, rtol=1e-6)
    assert not torch.cuda.is_initialized()
```

</details>

### `test_transfer_rejects_wrong_seed_arm_or_incompatible_architecture` · L53

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_transfer_rejects_wrong_seed_arm_or_incompatible_architecture():
    from chromaseed_palette_transfer import Bank

    args = ("patch5m", "linear", "aligned", encoders("aligned"), np.zeros(18), np.ones(18))
    wrong = list(args)
    wrong[3] = encoders("shuffled")
    with pytest.raises(ValueError):
        Bank(*wrong)
    wrong = list(args)
    wrong[3] = list(reversed(wrong[3]))
    with pytest.raises(ValueError):
        Bank(*wrong)
    with pytest.raises(ValueError):
        Bank("pool5m", *args[1:])
```

</details>

### `test_original_fit_and_prefix_are_bitwise_hr` · L69

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_original_fit_and_prefix_are_bitwise_hr():
    from chromaseed_head_range_fit import fit as old_fit
    from chromaseed_palette_transfer_fit import fit
    from test_chromaseed_architecture_scale import fixture

    x, tokens, y, warm = fixture()
    weights = np.linspace(0.2, 2, len(y))
    old, reference = old_fit(x, tokens, y, weights, warm, "patch5m", "linear", 4, (2, 4))
    new, info = fit(x, tokens, y, weights, warm, "patch5m", "linear", "original", None, 4, (2, 4))
    prefix, _ = fit(x, tokens, y, weights, warm, "patch5m", "linear", "original", None, 2, (2,))
    for step in (2, 4):
        for a, b in zip(old[step], new[step], strict=True):
            assert a.keys() == b.keys()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
    for a, b in zip(new[2], prefix[2], strict=True):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert info["sampling_sha256"] == reference["sampling_sha256"]
    assert info["final_learning_rates"] == reference["final_learning_rates"]
    assert not torch.cuda.is_initialized()
```

</details>

### `test_transferred_encoder_finetunes_and_keeps_the_original_sampling` · L92

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_transferred_encoder_finetunes_and_keeps_the_original_sampling():
    from chromaseed_palette_encoder import ENCODER_PARAMETERS
    from chromaseed_palette_transfer import Bank
    from chromaseed_palette_transfer_fit import fit
    from test_chromaseed_architecture_scale import fixture

    x, tokens, y, warm = fixture()
    weights = np.ones(len(y))
    mean = tokens.astype(float).mean((0, 1)).astype(np.float32)
    std = tokens.astype(float).std((0, 1)).astype(np.float32)
    source = encoders("aligned")
    initial = Bank("soft5m", "wide", "aligned", source, mean, std).theta.detach().numpy().copy()
    models, info = fit(x, tokens, y, weights, warm, "soft5m", "wide", "aligned", source, 4, (2, 4))
    prefix, other = fit(x, tokens, y, weights, warm, "soft5m", "wide", "aligned", source, 2, (2,))
    for slot in range(6):
        assert (
            np.count_nonzero(
                initial[slot, :ENCODER_PARAMETERS] != models[4][slot]["theta"][:ENCODER_PARAMETERS]
            )
            > 0
        )
        assert str(models[4][slot]["palette_arm"]) == "aligned"
        for key in prefix[2][slot]:
            np.testing.assert_array_equal(prefix[2][slot][key], models[2][slot][key])
    assert info["initial_theta_sha256"] == other["initial_theta_sha256"]
    assert info["transferred_parameters"] == ENCODER_PARAMETERS
    assert not torch.cuda.is_initialized()
```

</details>

### `test_heads_are_fixed_from_inner_per_architecture_choice` · L121

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_heads_are_fixed_from_inner_per_architecture_choice():
    from chromaseed_palette_transfer_run import ROLES, VARIANTS, resolve_heads

    selection = dict(
        roles={
            r: dict(
                policies=dict(
                    per_architecture={
                        v: dict(variant=v + "__wide", architecture=v, head_mode="wide")
                        for v in VARIANTS
                    }
                )
            )
            for r in ROLES
        }
    )
    heads = resolve_heads(selection)
    assert all(set(h.values()) == {"wide"} for h in heads.values())
    selection["roles"]["mixed"]["policies"]["per_architecture"]["patch5m"]["variant"] = (
        "pool5m__wide"
    )
    with pytest.raises(ValueError):
        resolve_heads(selection)
```

</details>

### `test_full_tensor_export_keeps_nonnumeric_provenance_and_parameter_count` · L146

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_full_tensor_export_keeps_nonnumeric_provenance_and_parameter_count():
    from chromaseed_architecture_scale import capacity
    from chromaseed_palette_transfer import Bank, encoder_digest
    from test_chromaseed_architecture_scale import fixture

    _, tokens, _, warm = fixture()
    mean = tokens.astype(float).mean((0, 1)).astype(np.float32)
    std = tokens.astype(float).std((0, 1)).astype(np.float32)
    source = encoders("shuffled")
    model = Bank("patch5m", "linear", "shuffled", source, mean, std).export(2, warm, tokens)
    assert str(model["palette_encoder_digest"]) == encoder_digest(source[1])
    assert str(model["palette_arm"]) == "shuffled"
    assert (
        sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc")
        == 4 * capacity("patch5m") + 458
    )
```

</details>
