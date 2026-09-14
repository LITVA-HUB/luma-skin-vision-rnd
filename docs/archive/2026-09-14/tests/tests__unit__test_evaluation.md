# `tests/unit/test_evaluation.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_evaluation.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `4aab813ac55cd11d48f0aa772d2e317daf3a8d7ee998f8f32019102bf5849009`. Строк: **84**.

## Зависимости

```python
import numpy as np
import pytest
from luma_skin_vision.calibration import Calibrator
from luma_skin_vision.evaluation import paired_bootstrap, repeatability, risk_coverage, summarize
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_equal_coverage_and_ties` | FunctionDef | См. реализацию | [L8](../../../../tests/unit/test_evaluation.py#L8) |
| `test_paired_cluster_bootstrap` | FunctionDef | См. реализацию | [L19](../../../../tests/unit/test_evaluation.py#L19) |
| `test_summary_and_repeatability` | FunctionDef | См. реализацию | [L29](../../../../tests/unit/test_evaluation.py#L29) |
| `test_calibration_split_quantile_serialization` | FunctionDef | См. реализацию | [L41](../../../../tests/unit/test_evaluation.py#L41) |
| `test_invalid_calibrator_rejected` | FunctionDef | См. реализацию | [L66](../../../../tests/unit/test_evaluation.py#L66) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_equal_coverage_and_ties` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_equal_coverage_and_ties():
    curve = risk_coverage([4, 1, 3, 2], [4, 1, 3, 2], ["d", "a", "c", "b"], coverages=[0.5, 1.0])
    assert curve[0]["accepted"] == 2 and curve[0]["mean_delta_e00"] == 1.5
    assert curve[1]["mean_delta_e00"] == 2.5
    tied = risk_coverage([1, 9], [0, 0], ["a", "b"], coverages=[0.5])
    reverse = risk_coverage([9, 1], [0, 0], ["b", "a"], coverages=[0.5])
    assert tied == reverse  # Tie break must not inspect true error or input row order.
    with pytest.raises(ValueError):
        risk_coverage([], [], [])
```

</details>

### `test_paired_cluster_bootstrap` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_paired_cluster_bootstrap():
    result = paired_bootstrap([4, 6, 8, 10], [3, 5, 7, 9], ["a", "a", "b", "b"], seed=1, draws=100)
    assert result["mean_difference"] == 1
    assert result["ci95"] == [1, 1]
    assert result["subjects"] == 2
    assert result == paired_bootstrap(
        [4, 6, 8, 10], [3, 5, 7, 9], ["a", "a", "b", "b"], seed=1, draws=100
    )
```

</details>

### `test_summary_and_repeatability` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_summary_and_repeatability():
    y = np.array([[50, 0, 0], [50, 0, 0]])
    metrics = summarize(y, y, ["a", "b"])
    assert metrics["mean_delta_e00"] == 0 and metrics["catastrophic_rate"] == 0
    assert metrics["mae_lab"] == [0, 0, 0]
    assert (
        repeatability([[[50, 0, 0], [50, 0, 0]], [[40, 0, 0], [40, 0, 0]]])["median_pair_delta_e00"]
        == 0
    )
    assert np.isfinite(summarize(y, y, [1, 2])["between_subject_prediction_std_lab"]).all()
```

</details>

### `test_calibration_split_quantile_serialization` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_calibration_split_quantile_serialization(tmp_path):
    c = Calibrator.fit(
        np.arange(20),
        np.arange(20) + 2,
        [f"s{i}" for i in range(20)],
        split="calibration",
        alpha=0.1,
        tolerance=5,
        model_hash="abc",
        data_kind="SYNTHETIC",
    )
    assert c.upper_error([1]).item() == 3
    c.save(tmp_path / "c.json")
    loaded = Calibrator.load(tmp_path / "c.json")
    assert loaded == c
    assert c.domain_validated is False
    with pytest.raises(ValueError, match="calibration"):
        Calibrator.fit([1], [1], ["a"], split="test", model_hash="a", data_kind="SYNTHETIC")
    # Finite sample order exceeds n: no finite guarantee available.
    small = Calibrator.fit(
        [1], [2], ["a"], split="calibration", alpha=0.1, model_hash="a", data_kind="SYNTHETIC"
    )
    assert np.isinf(small.upper_error([1])).all()
```

</details>

### `test_invalid_calibrator_rejected` · L66

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_calibrator_rejected(tmp_path):
    import json

    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            dict(
                schema_version="1.0",
                model_hash="x",
                data_kind="SYNTHETIC",
                alpha=2,
                tolerance=-1,
                residual_quantile=0,
                subject_count=10,
            )
        )
    )
    with pytest.raises(ValueError):
        Calibrator.load(p)
```

</details>
