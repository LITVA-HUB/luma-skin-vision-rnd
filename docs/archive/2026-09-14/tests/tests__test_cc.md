# `tests/test_cc.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `89748048833aa2db8f9d178894821f38ce0d622a5e238347cabe191671efd052`. Строк: **95**.

## Зависимости

```python
import numpy as np
import pytest
from luma_skin_vision.cc.core import (
    angular,
    experts,
    linearize,
    reproduction,
    selective_curve,
    summarize,
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_angular_scale_invariance_and_orthogonal` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc.py#L14) |
| `test_reproduction_is_corrected_neutral_not_recovery` | FunctionDef | См. реализацию | [L21](../../../../tests/test_cc.py#L21) |
| `test_summary_and_coverage_are_actual_accepted_counts` | FunctionDef | См. реализацию | [L29](../../../../tests/test_cc.py#L29) |
| `test_linear_loader_subtracts_black_not_srgb_gamma` | FunctionDef | См. реализацию | [L40](../../../../tests/test_cc.py#L40) |
| `test_model_contract_and_finite_gradient` | FunctionDef | См. реализацию | [L52](../../../../tests/test_cc.py#L52) |
| `test_capture_day_partition_is_stable` | FunctionDef | См. реализацию | [L68](../../../../tests/test_cc.py#L68) |
| `test_risk_fitting_cannot_see_test_errors` | FunctionDef | См. реализацию | [L75](../../../../tests/test_cc.py#L75) |
| `test_data_cache_fingerprint_detects_target_change` | FunctionDef | См. реализацию | [L88](../../../../tests/test_cc.py#L88) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_angular_scale_invariance_and_orthogonal` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_angular_scale_invariance_and_orthogonal():
    assert angular([[1, 0, 0]], [[0, 1, 0]])[0] == pytest.approx(90)
    assert angular([[1, 2, 3]], [[2, 4, 6]])[0] < 1e-5
    with pytest.raises(ValueError):
        angular([[0, 0, 0]], [[1, 1, 1]])
```

</details>

### `test_reproduction_is_corrected_neutral_not_recovery` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_reproduction_is_corrected_neutral_not_recovery():
    gt = np.array([[1.0, 2, 4]])
    pred = np.array([[2.0, 2, 2]])
    expected = np.degrees(np.arccos(3.5 / (np.sqrt(3) * np.sqrt(5.25))))
    assert reproduction(pred, gt)[0] == pytest.approx(expected)
    assert reproduction(gt * 3, gt)[0] < 1e-5
```

</details>

### `test_summary_and_coverage_are_actual_accepted_counts` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_summary_and_coverage_are_actual_accepted_counts():
    result = summarize(np.arange(1, 101))
    assert result["median"] == 50.5
    assert result["best25"] == 13
    assert result["worst25"] == 88
    curve = selective_curve(np.arange(1, 11), np.arange(1, 11), [str(i) for i in range(10)])
    assert curve["fixed"]["80"]["n"] == 8
    assert curve["fixed"]["80"]["mean"] == 4.5
    assert curve["fixed"]["95"]["coverage"] == 0.9
```

</details>

### `test_linear_loader_subtracts_black_not_srgb_gamma` · L40

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_linear_loader_subtracts_black_not_srgb_gamma():
    image = np.full((10, 10, 3), [3048, 4048, 6048], dtype=np.uint16)
    x = linearize(image, black=2048, white=15000)
    np.testing.assert_allclose(x[0, 0], np.array([1000, 2000, 4000]) / (15000 - 2048))
    est = experts(x)
    for e in est[:3]:
        assert angular(e[None], [[1, 2, 4]])[0] < 1e-5
    assert np.isfinite(est).all()
    with pytest.raises(ValueError):
        linearize(image.astype(np.uint8))
```

</details>

### `test_model_contract_and_finite_gradient` · L52

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_model_contract_and_finite_gradient():
    torch = pytest.importorskip("torch")
    from luma_skin_vision.cc.model import CompactCC, reproduction_loss

    torch.set_num_threads(2)
    for mixture in [False, True]:
        model = CompactCC(mixture).eval()
        pred, context = model(torch.rand(2, 3, 64, 64), torch.ones(2, 4, 3) / np.sqrt(3))
        assert pred.shape == (2, 3) and context.shape == (2, 64)
        assert (pred > 0).all()
        loss = reproduction_loss(pred, torch.ones(2, 3))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
```

</details>

### `test_capture_day_partition_is_stable` · L68

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_capture_day_partition_is_stable():
    from luma_skin_vision.cc.data import partition

    assert partition("2020:06:22") == partition("2020:06:22")
    assert partition("2020:06:22") in {"train", "val", "risk", "cal"}
```

</details>

### `test_risk_fitting_cannot_see_test_errors` · L75

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_risk_fitting_cannot_see_test_errors():
    from luma_skin_vision.cc.benchmark import apply_risk, fit_risk

    rng = np.random.default_rng(1)
    f = rng.normal(size=(60, 7))
    e = rng.uniform(0, 20, 60)
    a = fit_risk(f, e, np.arange(30), np.arange(30, 40))
    changed = e.copy()
    changed[40:] = 180
    b = fit_risk(f, changed, np.arange(30), np.arange(30, 40))
    np.testing.assert_array_equal(apply_risk(f, a), apply_risk(f, b))
```

</details>

### `test_data_cache_fingerprint_detects_target_change` · L88

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_data_cache_fingerprint_detects_target_change(tmp_path):
    from luma_skin_vision.cc.benchmark import data_hashes

    for name in ["cube.npz", "cube_manifest.json", "sony.npz", "sony_manifest.json"]:
        (tmp_path / name).write_bytes(b"original")
    original = data_hashes(tmp_path)
    (tmp_path / "cube.npz").write_bytes(b"changed GT")
    assert data_hashes(tmp_path) != original
```

</details>
