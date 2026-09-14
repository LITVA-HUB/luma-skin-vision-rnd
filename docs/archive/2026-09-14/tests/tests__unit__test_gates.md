# `tests/unit/test_gates.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_gates.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `273209c39541ab8510476d3ed862a412f9bb84b22f955efc6db42be13684e75e`. Строк: **65**.

## Зависимости

```python
import json
import numpy as np
import pytest
from luma_skin_vision.data import validate_records
from luma_skin_vision.gates import measurement_gate
from luma_skin_vision.synthetic import generate
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_real_training_requires_measurement_gate` | FunctionDef | См. реализацию | [L11](../../../../tests/unit/test_gates.py#L11) |
| `test_subject_macro_cannot_hide_image_coverage_units` | FunctionDef | См. реализацию | [L20](../../../../tests/unit/test_gates.py#L20) |
| `test_real_gate_binds_manifest_and_evidence` | FunctionDef | См. реализацию | [L41](../../../../tests/unit/test_gates.py#L41) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_real_training_requires_measurement_gate` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_real_training_requires_measurement_gate(tmp_path):
    path = generate(tmp_path, subjects=12)
    rows = validate_records(path)
    for row in rows:
        row.data_kind = "INSTRUMENT"
    with pytest.raises(ValueError, match="GATE 1"):
        measurement_gate(path, rows)
```

</details>

### `test_subject_macro_cannot_hide_image_coverage_units` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_subject_macro_cannot_hide_image_coverage_units():
    from luma_skin_vision.evaluation import bootstrap_selective

    target = np.tile([50, 0, 0], (4, 1))
    pred = target.copy()
    pred[:, 0] += 2
    args = (
        pred,
        target,
        target,
        [1, 9, 2, 8],
        [9, 1, 8, 2],
        ["a:l", "a:r", "b:l", "b:r"],
        ["s1", "s1", "s2", "s2"],
    )
    with pytest.raises(ValueError, match="image_ids"):
        bootstrap_selective(*args)
    result = bootstrap_selective(*args, image_ids=["a", "a", "b", "b"], draws=20)
    assert result["ci95"][0] > 0
```

</details>

### `test_real_gate_binds_manifest_and_evidence` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_real_gate_binds_manifest_and_evidence(tmp_path):
    from luma_skin_vision.data import sha256

    path = generate(tmp_path, subjects=12)
    rows = validate_records(path)
    # Metadata-only fixture tests gate requirements, not an instrument dataset.
    for row in rows:
        row.data_kind = "INSTRUMENT"
        row.makeup_protocol = "none"
    approval = {
        "schema_version": "1.0",
        "dataset_hash": sha256(path),
        "decision": "PASS",
        "reviewer_role": "colorimetry_lead",
        "protocol_id": "pilot-v1",
        "repeatability_p95_limit": 1.0,
        "reference_accuracy_reviewed": True,
        "registration_reviewed": True,
    }
    (tmp_path / "measurement_approval.json").write_text(json.dumps(approval))
    assert measurement_gate(path, rows)["decision"] == "PASS"
    approval["dataset_hash"] = "changed"
    (tmp_path / "measurement_approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="GATE 1"):
        measurement_gate(path, rows)
```

</details>
