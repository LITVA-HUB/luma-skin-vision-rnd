# `tests/test_cc_v2_select.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v2_select.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Synthetic-only selector integrity/leakage tests; no real target evaluation.

SHA-256 исходника: `2d80d2f6047109b60f41582c53968298bd2cffa9d58b99a85bd1995ab7448575`. Строк: **261**.

## Зависимости

```python
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.v2_experiment import SCHEMA, data_fingerprints, source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `selector` | FunctionDef | См. реализацию | [L17](../../../../tests/test_cc_v2_select.py#L17) |
| `bounded_cpu` | FunctionDef | См. реализацию | [L26](../../../../tests/test_cc_v2_select.py#L26) |
| `write_predictions` | FunctionDef | См. реализацию | [L31](../../../../tests/test_cc_v2_select.py#L31) |
| `fixture_run` | FunctionDef | Opaque checkpoint bytes are sufficient: selectors never instantiate a CNN. | [L43](../../../../tests/test_cc_v2_select.py#L43) |
| `alter_npz` | FunctionDef | См. реализацию | [L122](../../../../tests/test_cc_v2_select.py#L122) |
| `test_selector_fit_ignores_all_non_risk_cal_values_and_keeps_date_folds_disjoint` | FunctionDef | См. реализацию | [L129](../../../../tests/test_cc_v2_select.py#L129) |
| `test_fit_rejects_predictions_changed_after_model_export` | FunctionDef | См. реализацию | [L162](../../../../tests/test_cc_v2_select.py#L162) |
| `test_fit_rejects_prediction_manifest_for_another_checkpoint` | FunctionDef | См. реализацию | [L169](../../../../tests/test_cc_v2_select.py#L169) |
| `test_fit_rejects_source_predictions_bound_to_a_different_input_cache` | FunctionDef | См. реализацию | [L179](../../../../tests/test_cc_v2_select.py#L179) |
| `test_frozen_policy_always_refuses_invalid_inputs_even_at_full_coverage` | FunctionDef | См. реализацию | [L193](../../../../tests/test_cc_v2_select.py#L193) |
| `test_external_targets_must_match_prediction_export_binding` | FunctionDef | См. реализацию | [L207](../../../../tests/test_cc_v2_select.py#L207) |
| `test_mixed_validity_caps_diagnostic_coverage_and_preserves_metric_units` | FunctionDef | См. реализацию | [L235](../../../../tests/test_cc_v2_select.py#L235) |
| `test_zero_calibration_error_does_not_collapse_risk_ranking` | FunctionDef | См. реализацию | [L256](../../../../tests/test_cc_v2_select.py#L256) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_selector_fit_ignores_all_non_risk_cal_values_and_keeps_date_folds_disjoint` · L129

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selector_fit_ignores_all_non_risk_cal_values_and_keeps_date_folds_disjoint(
    selector, tmp_path
):
    run, data, rows = fixture_run(tmp_path / "clean")
    poisoned, poisoned_data, _ = fixture_run(tmp_path / "poisoned", poison_heldout=True)
    selector.fit(run, data)
    selector.fit(poisoned, poisoned_data)
    normal = json.loads((run / "risk_v2/selection.json").read_text())
    other = json.loads((poisoned / "risk_v2/selection.json").read_text())
    mapping = {r["id"]: r for r in rows}
    assert set(normal["fit_ids"]).isdisjoint(normal["cal_ids"])
    assert len(normal["folds"]) == 5
    visited = []
    for fold in normal["folds"]:
        train = {mapping[normal["fit_ids"][i]]["group"] for i in fold["train"]}
        validation = {mapping[normal["fit_ids"][i]]["group"] for i in fold["validation"]}
        assert train.isdisjoint(validation)
        visited.extend(fold["validation"])
    assert sorted(visited) == list(range(len(normal["fit_ids"])))
    for block in ("context", "cheap", "combined"):
        a, b = normal["heads"][block], other["heads"][block]
        assert a["selected"] == b["selected"]
        assert a["scale"] == b["scale"]
        assert a["candidates"] == b["candidates"]
        assert {c["name"] for c in a["candidates"]} == {
            "ridge1",
            "ridge10",
            "ridge100",
            "hgb3",
            "hgb7",
        }
```

</details>

### `test_fit_rejects_predictions_changed_after_model_export` · L162

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fit_rejects_predictions_changed_after_model_export(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    alter_npz(run / "predictions.npz", "pred", lambda value: value[:, ::-1])
    with pytest.raises(ValueError, match="(?i)prediction|binding|changed"):
        selector.fit(run, data)
```

</details>

### `test_fit_rejects_prediction_manifest_for_another_checkpoint` · L169

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fit_rejects_prediction_manifest_for_another_checkpoint(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint_sha256"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="(?i)checkpoint|binding|prediction"):
        selector.fit(run, data)
```

</details>

### `test_fit_rejects_source_predictions_bound_to_a_different_input_cache` · L179

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fit_rejects_source_predictions_bound_to_a_different_input_cache(selector, tmp_path):
    run, data, rows = fixture_run(tmp_path)
    other = tmp_path / "other_input.npz"
    np.savez_compressed(other, gt=np.full((len(rows), 3), 2.0))
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    source = manifest["outputs"]["predictions.npz"]
    source["input_npz"] = str(other)
    source["input_hashes"]["npz"] = sha256(other)
    write_json(path, manifest)
    with pytest.raises(ValueError, match="(?i)input|dataset|binding|prediction"):
        selector.fit(run, data)
```

</details>

### `test_frozen_policy_always_refuses_invalid_inputs_even_at_full_coverage` · L193

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_frozen_policy_always_refuses_invalid_inputs_even_at_full_coverage(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path, invalid_test=True)
    selector.fit(run, data)
    output = tmp_path / "evaluation.json"
    selector.evaluate(run, data, None, output)
    report = json.loads(output.read_text())["domains"]["source_regression"]
    for record in report.values():
        assert record["invalid_n"] == 4
        for frozen in record["frozen_source_thresholds"].values():
            assert frozen["n"] == 0
            assert frozen["coverage"] == 0
            assert frozen["mean_reproduction"] is None
```

</details>

### `test_external_targets_must_match_prediction_export_binding` · L207

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_external_targets_must_match_prediction_export_binding(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path)
    external = tmp_path / "external.npz"
    external_manifest = tmp_path / "external.json"
    rows = [{"id": f"e{i}", "camera": "synthetic", "group": f"e{i}"} for i in range(3)]
    write_json(external_manifest, rows)
    np.savez_compressed(external, gt=np.ones((3, 3)))
    values = {
        "pred": np.ones((3, 3)) / np.sqrt(3),
        "context": np.zeros((3, 64)),
        "cheap_features": np.zeros((3, 21)),
        "valid": np.ones(3, bool),
        "ids": np.array([r["id"] for r in rows]),
    }
    path = run / "predictions_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"]["external_predictions.npz"] = write_predictions(
        run, "external_predictions.npz", external, external_manifest, values
    )
    write_json(path, manifest)
    selector.fit(run, data)
    alter_npz(external, "gt", lambda value: value * np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="(?i)prediction|fingerprint|binding|changed|input"):
        selector.evaluate(
            run, data, [str(external), str(external_manifest)], tmp_path / "evaluation.json"
        )
```

</details>

### `test_mixed_validity_caps_diagnostic_coverage_and_preserves_metric_units` · L235

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_mixed_validity_caps_diagnostic_coverage_and_preserves_metric_units(selector):
    pred = np.column_stack([np.linspace(0.5, 1.5, 10), np.ones(10), np.ones(10)])
    gt = np.ones_like(pred)
    valid = np.array([False, False] + [True] * 8)
    scores = np.arange(10, dtype=float)
    ids = [f"image{i}" for i in range(10)]
    report = selector.selective_result(pred, gt, scores, ids, np.array([1e8, 1e9]), valid)
    errors = np.asarray(report["errors"])
    assert report["max_supported_coverage"] == 0.8
    assert report["selective"]["order"] == list(range(2, 10))
    assert report["selective"]["coverage"][-1] == 0.8
    assert report["selective"]["fixed"]["80"]["n"] == 8
    assert report["selective"]["fixed"]["80"]["mean"] == pytest.approx(errors[valid].mean())
    assert report["selective"]["fixed"]["60"]["n"] == 6
    assert report["selective"]["fixed"]["60"]["mean"] == pytest.approx(errors[2:8].mean())
    assert report["selective"]["fixed"]["90"]["attainable"] is False
    for frozen in report["frozen_source_thresholds"].values():
        assert frozen["n"] == 8
        assert frozen["coverage"] == 0.8
```

</details>

### `test_zero_calibration_error_does_not_collapse_risk_ranking` · L256

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_calibration_error_does_not_collapse_risk_ranking(selector, tmp_path):
    run, data, _ = fixture_run(tmp_path, perfect_cal=True)
    selector.fit(run, data)
    state = json.loads((run / "risk_v2/selection.json").read_text())
    for head in state["heads"].values():
        assert head["scale"] > 0
```

</details>
