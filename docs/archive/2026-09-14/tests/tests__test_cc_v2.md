# `tests/test_cc_v2.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU engineering tests; these are not real-data performance measurements.

SHA-256 исходника: `f36beff4e625cafba6fa214886479334bab0420d9aa4bed7b20e8d98c53d0ee7`. Строк: **279**.

## Зависимости

```python
import importlib
import importlib.util
import json
from argparse import Namespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `module` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_v2.py#L14) |
| `cpu_threads` | FunctionDef | См. реализацию | [L21](../../../../tests/test_cc_v2.py#L21) |
| `test_unnormalized_channel_anchor_is_gain_homogeneous_with_masks` | FunctionDef | См. реализацию | [L29](../../../../tests/test_cc_v2.py#L29) |
| `test_model_positive_normalized_prediction_and_finite_gradients` | FunctionDef | См. реализацию | [L42](../../../../tests/test_cc_v2.py#L42) |
| `test_nontrivial_residual_is_equivariant_and_reproduction_error_unchanged` | FunctionDef | См. реализацию | [L56](../../../../tests/test_cc_v2.py#L56) |
| `test_matched_modes_have_identical_capacity_and_feature_columns` | FunctionDef | См. реализацию | [L89](../../../../tests/test_cc_v2.py#L89) |
| `test_degenerate_inputs_are_finite_and_explicitly_invalid` | FunctionDef | См. реализацию | [L104](../../../../tests/test_cc_v2.py#L104) |
| `test_input_validation_rejects_negative_nonfinite_and_integer_pixels` | FunctionDef | См. реализацию | [L118](../../../../tests/test_cc_v2.py#L118) |
| `test_gain_augmentation_transforms_gt_and_pixels_without_clipping` | FunctionDef | См. реализацию | [L130](../../../../tests/test_cc_v2.py#L130) |
| `make_cache` | FunctionDef | См. реализацию | [L146](../../../../tests/test_cc_v2.py#L146) |
| `args` | FunctionDef | См. реализацию | [L165](../../../../tests/test_cc_v2.py#L165) |
| `test_runner_reuses_splits_and_binds_data_fingerprints` | FunctionDef | См. реализацию | [L182](../../../../tests/test_cc_v2.py#L182) |
| `test_camera_split_keeps_target_camera_out_of_all_source_roles` | FunctionDef | См. реализацию | [L198](../../../../tests/test_cc_v2.py#L198) |
| `test_cpu_train_uses_only_source_and_snapshots_exact_code` | FunctionDef | См. реализацию | [L218](../../../../tests/test_cc_v2.py#L218) |
| `test_prediction_binds_checkpoint_emits_features_and_refuses_overwrite` | FunctionDef | См. реализацию | [L238](../../../../tests/test_cc_v2.py#L238) |
| `test_prediction_rejects_changed_live_source` | FunctionDef | См. реализацию | [L272](../../../../tests/test_cc_v2.py#L272) |

## Все тестовые определения (12)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_unnormalized_channel_anchor_is_gain_homogeneous_with_masks` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('p', [1, 6])
def test_unnormalized_channel_anchor_is_gain_homogeneous_with_masks(p):
    v2 = module()
    torch.manual_seed(83)
    x = torch.rand(2, 3, 12, 17, dtype=torch.float64) + 0.1
    x[:, :, :3, :6] = 0
    gains = torch.exp(torch.randn(2, 3, dtype=x.dtype))
    anchor = v2.channel_anchor(x, p)
    torch.testing.assert_close(anchor, x.pow(p).mean((-2, -1)).pow(1 / p))
    torch.testing.assert_close(v2.channel_anchor(x * gains[:, :, None, None], p), anchor * gains)
```

</details>

### `test_model_positive_normalized_prediction_and_finite_gradients` · L42

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('mode', ['direct', 'gw', 'sog'])
@pytest.mark.parametrize('backbone', ['small', 'large'])
def test_model_positive_normalized_prediction_and_finite_gradients(mode, backbone):
    v2 = module()
    model = v2.CompactResidualCC(mode, backbone).eval()
    assert torch.count_nonzero(model.illuminant.weight) == 0
    pred, context = model(torch.rand(2, 3, 32, 32))
    assert context.shape == (2, 64)
    assert torch.all(pred > 0) and torch.isfinite(pred).all()
    torch.testing.assert_close(pred.norm(dim=-1), torch.ones(2))
    loss = (pred - F.normalize(torch.rand_like(pred) + 0.1, dim=-1)).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
```

</details>

### `test_nontrivial_residual_is_equivariant_and_reproduction_error_unchanged` · L56

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('mode', ['gw', 'sog'])
def test_nontrivial_residual_is_equivariant_and_reproduction_error_unchanged(mode):
    v2 = module()
    from luma_skin_vision.cc.core import reproduction

    torch.manual_seed(71)
    model = v2.CompactResidualCC(mode).eval()
    with torch.no_grad():
        model.illuminant.weight.normal_(0, 0.8)
        model.illuminant.bias.copy_(torch.tensor([0.3, -0.2, 0.1]))
    x = torch.rand(2, 3, 64, 64) + 0.02
    x[:, :, -8:, -8:] = 0
    gt = F.normalize(torch.rand(2, 3) + 0.2, dim=-1)
    with torch.no_grad():
        pred, context = model(x)
        features = v2.risk_features_invariant(x, pred, context)
        for _ in range(3):
            gain = torch.exp(torch.rand(2, 3) * 2 - 1)
            gained = x * gain[:, :, None, None]
            gp, gc = model(gained)
            torch.testing.assert_close(gp, F.normalize(pred * gain, dim=-1), rtol=3e-5, atol=2e-6)
            torch.testing.assert_close(gc, context, rtol=3e-5, atol=2e-6)
            gf = v2.risk_features_invariant(gained, gp, gc)
            torch.testing.assert_close(gf["combined"], features["combined"], rtol=3e-5, atol=2e-6)
            np.testing.assert_allclose(
                reproduction(gp.numpy(), (gt * gain).numpy()),
                reproduction(pred.numpy(), gt.numpy()),
                atol=2e-4,
            )
        assert not torch.allclose(
            pred, F.normalize(v2.channel_anchor(x, 1 if mode == "gw" else 6), dim=-1)
        )
```

</details>

### `test_matched_modes_have_identical_capacity_and_feature_columns` · L89

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_matched_modes_have_identical_capacity_and_feature_columns():
    v2 = module()
    counts = [
        sum(p.numel() for p in v2.CompactResidualCC(mode).parameters())
        for mode in ["direct", "gw", "sog"]
    ]
    assert len(set(counts)) == 1
    x = torch.rand(2, 3, 33, 41)
    p, c = v2.CompactResidualCC("gw").eval()(x)
    f = v2.risk_features_invariant(x, p, c)
    assert f["cheap"].shape == (2, len(v2.CHEAP_FEATURE_COLUMNS))
    assert f["combined"].shape == (2, 64 + len(v2.CHEAP_FEATURE_COLUMNS))
```

</details>

### `test_degenerate_inputs_are_finite_and_explicitly_invalid` · L104

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('mode', ['direct', 'gw', 'sog'])
def test_degenerate_inputs_are_finite_and_explicitly_invalid(mode):
    v2 = module()
    x = torch.zeros(2, 3, 32, 32)
    x[1, 0] = 1
    model = v2.CompactResidualCC(mode).eval()
    p, c = model(x)
    assert torch.isfinite(p).all() and torch.all(p > 0)
    features = v2.risk_features_invariant(x, p, c)
    assert not features["valid"].any()
    assert all(torch.isfinite(features[key]).all() for key in ["cheap", "combined", "context"])
    p.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
```

</details>

### `test_input_validation_rejects_negative_nonfinite_and_integer_pixels` · L118

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_input_validation_rejects_negative_nonfinite_and_integer_pixels():
    v2 = module()
    for x in [
        torch.full((1, 3, 32, 32), -1.0),
        torch.full((1, 3, 32, 32), float("nan")),
        torch.ones(1, 3, 32, 32, dtype=torch.int32),
    ]:
        with pytest.raises((ValueError, TypeError, AssertionError)):
            v2.channel_anchor(x)
```

</details>

### `test_gain_augmentation_transforms_gt_and_pixels_without_clipping` · L130

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('magnitude', [0, 0.7])
def test_gain_augmentation_transforms_gt_and_pixels_without_clipping(magnitude):
    v2 = module()
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5) + 1
    gt = F.normalize(torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]), dim=-1)
    aug, target, meta = v2.gain_augment(
        x, gt, magnitude=magnitude, generator=torch.Generator().manual_seed(91)
    )
    expected = x * meta["gains"][:, :, None, None] * meta["exposure"][:, None, None, None]
    expected = torch.where(meta["flipped"][:, None, None, None], expected.flip(-1), expected)
    torch.testing.assert_close(aug, expected)
    torch.testing.assert_close(target, F.normalize(gt * meta["gains"], dim=-1))
    assert aug.max() > 4
    if magnitude == 0:
        torch.testing.assert_close(meta["gains"], torch.ones(2, 3))
```

</details>

### `test_runner_reuses_splits_and_binds_data_fingerprints` · L182

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_runner_reuses_splits_and_binds_data_fingerprints(tmp_path):
    runner = module("v2_experiment")
    from luma_skin_vision.cc.benchmark import indices

    rows = make_cache(tmp_path / "data")
    actual = runner.split_indices(rows, "official")
    for key, expected in indices(rows, "official").items():
        np.testing.assert_array_equal(actual[key], expected)
    original = runner.data_fingerprints(tmp_path / "data")
    runner.verify_data(tmp_path / "data", original)
    with (tmp_path / "data/cube_manifest.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="fingerprint|changed"):
        runner.verify_data(tmp_path / "data", original)
```

</details>

### `test_camera_split_keeps_target_camera_out_of_all_source_roles` · L198

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_camera_split_keeps_target_camera_out_of_all_source_roles():
    runner = module("v2_experiment")
    from luma_skin_vision.cc.benchmark import indices

    rows = [
        {"id": str(i), "subset": part, "group": part, "camera": "Canon EOS 550D"}
        for i, part in enumerate(["train", "train", "val", "risk", "cal"])
    ]
    rows.extend(
        [
            {"id": "target", "subset": "train", "group": "target_day", "camera": "Canon EOS 600D"},
            {"id": "overlap", "subset": "test", "group": "train", "camera": "Canon EOS 600D"},
        ]
    )
    actual = runner.split_indices(rows, "camera")
    for key, expected in indices(rows, "camera").items():
        np.testing.assert_array_equal(actual[key], expected)
    assert actual["test"].tolist() == [5]
```

</details>

### `test_cpu_train_uses_only_source_and_snapshots_exact_code` · L218

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cpu_train_uses_only_source_and_snapshots_exact_code(tmp_path):
    runner = module("v2_experiment")
    make_cache(tmp_path / "data", poisoned_test=True)
    out = tmp_path / "run"
    runner.train(args(tmp_path / "data", out))
    config = json.loads((out / "config.json").read_text())
    checkpoint = json.loads((out / "checkpoint_manifest.json").read_text())
    summary = json.loads((out / "training.json").read_text())
    assert checkpoint["source_hash"] == config["source_hash"]
    assert config["source_snapshot"]["source_hash"] == config["source_hash"]
    assert (out / "source_snapshot/src/luma_skin_vision/cc/v2.py").is_file()
    assert len(summary["validation_gain_stress"]) == 4
    for stress in summary["validation_gain_stress"]:
        assert stress["mean_reproduction"] == pytest.approx(summary["best_validation"], abs=2e-4)
    assert not (out / "predictions.npz").exists()
    assert summary["best_validation"] == min(h["val_reproduction"] for h in summary["history"])
    with pytest.raises(FileExistsError):
        runner.train(args(tmp_path / "data", out))
```

</details>

### `test_prediction_binds_checkpoint_emits_features_and_refuses_overwrite` · L238

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prediction_binds_checkpoint_emits_features_and_refuses_overwrite(tmp_path):
    runner = module("v2_experiment")
    make_cache(tmp_path / "data")
    out = tmp_path / "run"
    a = args(tmp_path / "data", out)
    runner.train(a)
    external = tmp_path / "external.npz"
    external_manifest = tmp_path / "external.json"
    np.savez_compressed(
        external, images=np.zeros((2, 3, 32, 32), np.float32), gt=np.full((2, 3), np.nan)
    )
    external_manifest.write_text(json.dumps([{"id": "external0"}, {"id": "external1"}]))
    a.external = [str(external), str(external_manifest)]
    runner.predict(a)
    with np.load(out / "predictions.npz") as predictions:
        assert predictions["pred"].shape == (12, 3)
        assert predictions["context"].shape == (12, 64)
        assert predictions["cheap_features"].shape[0] == 12
        assert predictions["valid"].all()
    with np.load(out / "external_predictions.npz") as predictions:
        assert not predictions["valid"].any()
        assert np.isfinite(predictions["pred"]).all()
    output_manifest = json.loads((out / "predictions_manifest.json").read_text())
    assert output_manifest["fitting_or_error_evaluation"] is False
    assert len(output_manifest["outputs"]["external_predictions.npz"]["input_hashes"]["npz"]) == 64
    with pytest.raises(FileExistsError):
        runner.predict(a)
    a.overwrite = True
    with (out / "model.pt").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint"):
        runner.predict(a)
```

</details>

### `test_prediction_rejects_changed_live_source` · L272

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prediction_rejects_changed_live_source(tmp_path, monkeypatch):
    runner = module("v2_experiment")
    make_cache(tmp_path / "data")
    a = args(tmp_path / "data", tmp_path / "run")
    runner.train(a)
    monkeypatch.setattr(runner, "source_identity", lambda: {"source_hash": "0" * 64})
    with pytest.raises(ValueError, match="Live source"):
        runner.predict(a)
```

</details>
