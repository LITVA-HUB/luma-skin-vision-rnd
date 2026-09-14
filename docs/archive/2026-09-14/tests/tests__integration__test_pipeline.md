# `tests/integration/test_pipeline.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/integration/test_pipeline.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `6cf1e30ccc1984149b59d133a3502679f5a4c083833d83b4fc03d29be55480e9`. Строк: **88**.

## Зависимости

```python
import json
import numpy as np
import pytest
from luma_skin_vision.api import analyze
from luma_skin_vision.synthetic import generate
from luma_skin_vision.training import calibrate_run, evaluate_run, load_run, train
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `config` | FunctionDef | См. реализацию | [L11](../../../../tests/integration/test_pipeline.py#L11) |
| `test_train_calibrate_evaluate_load_and_conservative_api` | FunctionDef | См. реализацию | [L30](../../../../tests/integration/test_pipeline.py#L30) |
| `test_preparation_retains_compact_crops_not_full_photos` | FunctionDef | См. реализацию | [L65](../../../../tests/integration/test_pipeline.py#L65) |
| `test_baseline_train_and_test_fingerprint` | FunctionDef | См. реализацию | [L81](../../../../tests/integration/test_pipeline.py#L81) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_train_calibrate_evaluate_load_and_conservative_api` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_train_calibrate_evaluate_load_and_conservative_api(tmp_path):
    manifest = generate(tmp_path / "data", subjects=12, size=32)
    run = train(config(manifest), tmp_path / "runs")
    model, metadata = load_run(run)
    assert metadata["data_kind"] == "SYNTHETIC"
    assert metadata["parameters"] < 10_000_000
    assert metadata["status"] == "COMPLETED"
    oof = json.loads((run / "oof_audit.json").read_text())
    assert all(not set(f["fit_subjects"]) & set(f["held_out_subjects"]) for f in oof)
    calibration = calibrate_run(run)
    assert calibration.domain_validated is False
    report = evaluate_run(run)
    assert report["data_kind"] == "SYNTHETIC"
    assert report["split"] == "test"
    assert np.isfinite(report["metrics"]["mean_delta_e00"])
    import torch

    x = torch.ones(1, 3, 32, 32) * 0.5
    aux = torch.zeros(1, 12)
    model.eval()
    second, _ = load_run(run)
    with torch.no_grad():
        np.testing.assert_array_equal(model(x, aux).numpy(), second(x, aux).numpy())
    result = analyze(run, manifest.parent / "images" / "syn_000_d0_l0_r0.jpg", bbox=[0, 0, 32, 32])
    assert result["status"] == "UNSUPPORTED"
    assert result["measurement"] is None and result["profile_update"]["may_update"] is False
    from luma_skin_vision.export import benchmark_run, export_run

    export_run(run)
    onnx = run / "model.onnx"
    onnx.write_bytes(onnx.read_bytes() + b"stale-artifact")
    with pytest.raises(ValueError, match="ONNX artifact"):
        benchmark_run(run, onnx=True, iterations=5)
```

</details>

### `test_preparation_retains_compact_crops_not_full_photos` · L65

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_preparation_retains_compact_crops_not_full_photos(tmp_path):
    import tracemalloc

    from luma_skin_vision.data import validate_records
    from luma_skin_vision.training import prepare

    manifest = generate(tmp_path, subjects=8, size=256)
    rows = validate_records(manifest)
    tracemalloc.start()
    prepared = prepare(manifest, rows, config(manifest))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert prepared["images"].shape == (128, 3, 32, 32)
    assert peak < 48 * 1024 * 1024
```

</details>

### `test_baseline_train_and_test_fingerprint` · L81

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_baseline_train_and_test_fingerprint(tmp_path):
    manifest = generate(tmp_path / "data", subjects=12, size=32)
    run = train(config(manifest, "baseline_a0"), tmp_path / "runs")
    assert evaluate_run(run)["metrics"]["mean_delta_e00"] >= 0
    with manifest.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate_run(run)
```

</details>
