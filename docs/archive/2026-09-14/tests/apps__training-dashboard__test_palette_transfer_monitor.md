# `apps/training-dashboard/test_palette_transfer_monitor.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../apps/training-dashboard/test_palette_transfer_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exercise persisted completion, lineage, liveness and estimate boundaries.

SHA-256 исходника: `8b8fda34d79d04f2a7604f4ca98e9ce2446c600ad6732daf4995bdc2a362c9e5`. Строк: **229**.

## Зависимости

```python
import hashlib
import json
import os
import sys
from pathlib import Path
import pytest
from monitor import ROLES
from palette_transfer_monitor import ARMS, VARIANTS, snapshot
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `write` | FunctionDef | См. реализацию | [L16](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L16) |
| `fixture` | FunctionDef | См. реализацию | [L23](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L23) |
| `live_job` | FunctionDef | См. реализацию | [L43](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L43) |
| `receipt` | FunctionDef | См. реализацию | [L65](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L65) |
| `test_prepared_is_waiting_not_training_or_accurate` | FunctionDef | См. реализацию | [L91](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L91) |
| `test_parent_and_gpu_gates_require_matching_receipts` | FunctionDef | См. реализацию | [L104](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L104) |
| `test_dead_unknown_or_stale_never_show_live_speed` | FunctionDef | См. реализацию | [L127](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L127) |
| `test_steps_and_reported_counter_do_not_replace_receipts` | FunctionDef | См. реализацию | [L137](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L137) |
| `test_pid_mismatch_and_malformed_json_degrade_without_crashing` | FunctionDef | См. реализацию | [L151](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L151) |
| `test_eta_uses_native_receipts_and_selected_final_lengths` | FunctionDef | См. реализацию | [L163](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L163) |
| `test_matched_hr_times_are_explicitly_provisional` | FunctionDef | См. реализацию | [L188](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L188) |
| `test_complete_needs_all_bank_receipts_and_a_seal` | FunctionDef | См. реализацию | [L200](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L200) |
| `test_absent_registration_does_not_invent_a_study` | FunctionDef | См. реализацию | [L219](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L219) |
| `test_malformed_nested_selection_is_visible_and_does_not_break_api` | FunctionDef | См. реализацию | [L223](../../../../apps/training-dashboard/test_palette_transfer_monitor.py#L223) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_prepared_is_waiting_not_training_or_accurate` · L91

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prepared_is_waiting_not_training_or_accurate(tmp_path):
    root, _, options = fixture(tmp_path)
    row = snapshot(**options)
    assert row["state"] == "waiting_previous"
    assert row["completed_banks"] == 0 and len(row["packages"]) == 72
    assert row["cpu"]["passed"] and row["cpu"]["combinations"] == 27
    assert row["steps_per_second"] is None and row["estimated_remaining_seconds"] is None
    assert row["current"] is None and not row["native_quality_verified"]
    assert len({p["id"] for p in row["packages"]}) == 72
    assert sum(p["stage"] == "inner" for p in row["packages"]) == 54
    assert not any(p["initialization"] == "original" for p in row["packages"])
```

</details>

### `test_parent_and_gpu_gates_require_matching_receipts` · L104

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_parent_and_gpu_gates_require_matching_receipts(tmp_path):
    root, registration, options = fixture(tmp_path)
    source = write(options["hr_root"] / "source_lock.json", {"sources": {}})
    write(options["hr_root"] / "job.json", dict(status="complete", pid=99))
    seal = tmp_path / "docs/benchmarks/chromaseed_head_range_v1/verification.json"
    write(seal, dict(passed=True, source_lock_sha256="wrong"))
    options["probe"] = lambda _: False
    assert snapshot(**options)["state"] == "waiting_previous"
    write(seal, dict(passed=True, source_lock_sha256=source))
    assert snapshot(**options)["state"] == "needs_gpu_check"
    write(
        root / "preflight_cuda.json",
        dict(passed=True, registration_sha256=registration, device="cuda"),
    )
    assert snapshot(**options)["state"] == "ready"
    options["probe"] = lambda _: True
    assert snapshot(**options)["state"] == "waiting_previous"
```

</details>

### `test_dead_unknown_or_stale_never_show_live_speed` · L127

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('alive,now,state', [(False, 1100, 'interrupted'), (None, 1100, 'unknown'), (True, 1400, 'stale')])
def test_dead_unknown_or_stale_never_show_live_speed(tmp_path, alive, now, state):
    root, _, options = fixture(tmp_path)
    live_job(root, options)
    options.update(probe=lambda _: alive, now=now)
    row = snapshot(**options)
    assert row["state"] == state
    assert row["steps_per_second"] is None and row["estimated_remaining_seconds"] is None
    assert not any(p["status"] == "running" for p in row["packages"])
```

</details>

### `test_steps_and_reported_counter_do_not_replace_receipts` · L137

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_steps_and_reported_counter_do_not_replace_receipts(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    row = snapshot(**options)
    assert row["state"] == "training" and row["steps_per_second"] == 20.48
    assert row["completed_banks"] == 0 and row["packages"][0]["status"] == "running"
    assert row["estimated_remaining_seconds"] is None
    receipt(root, source)
    receipt(root, "foreign-source", fold=1)
    row = snapshot(**options)
    assert row["completed_banks"] == 1 and row["packages"][0]["status"] == "completed"
    assert row["warnings"]
```

</details>

### `test_pid_mismatch_and_malformed_json_degrade_without_crashing` · L151

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pid_mismatch_and_malformed_json_degrade_without_crashing(tmp_path):
    root, _, options = fixture(tmp_path)
    _, progress = live_job(root, options)
    write(root / "progress.json", dict(progress, pid=7))
    assert snapshot(**options)["state"] == "unknown"
    (root / "progress.json").write_text("{partial", encoding="utf-8")
    row = snapshot(**options)
    assert row["state"] == "degraded" and row["steps_per_second"] is None
    write(root / "progress.json", [])
    assert snapshot(**options)["warnings"]
```

</details>

### `test_eta_uses_native_receipts_and_selected_final_lengths` · L163

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_eta_uses_native_receipts_and_selected_final_lengths(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    for variant in VARIANTS:
        receipt(root, source, variant=variant, fold=2)
    row = snapshot(**options)
    assert row["estimated_remaining_seconds"] == pytest.approx(69 * 204.8 - 100)
    assert row["final_steps_provisional"] and row["eta_basis"] == "p3"
    selection = dict(
        source_lock_sha256=source,
        roles={
            role: {
                "policies": {
                    "per_pair": {v + "__" + arm: {"step": 128} for v in VARIANTS for arm in ARMS}
                }
            }
            for role in ROLES
        },
    )
    write(root / "selections.json", selection)
    row = snapshot(**options)
    assert not row["final_steps_provisional"]
    assert row["estimated_remaining_seconds"] == pytest.approx(51 * 204.8 + 18 * 12.8 - 100)
```

</details>

### `test_matched_hr_times_are_explicitly_provisional` · L188

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_matched_hr_times_are_explicitly_provisional(tmp_path):
    root, _, options = fixture(tmp_path)
    live_job(root, options)
    for role in ROLES:
        for variant in VARIANTS:
            path = options["hr_root"] / "inner" / role / (variant + "__wide") / "fold0/receipt.json"
            write(path, dict(steps=2048, write_and_prediction_inclusive_seconds=204.8))
    row = snapshot(**options)
    assert row["eta_basis"] == "prior_native"
    assert row["estimated_remaining_seconds"] == pytest.approx(72 * 204.8 - 100)
```

</details>

### `test_complete_needs_all_bank_receipts_and_a_seal` · L200

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_complete_needs_all_bank_receipts_and_a_seal(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    write(root / "job.json", dict(status="complete", pid=42, source_lock_sha256=source))
    options["probe"] = lambda _: False
    assert snapshot(**options)["state"] == "degraded"
    for role in ROLES:
        for variant in VARIANTS:
            for arm in ARMS:
                for fold in [0, 1, 2, None]:
                    receipt(root, source, role, variant, arm, fold)
    assert snapshot(**options)["state"] == "complete"
    seal = tmp_path / "docs/benchmarks/chromaseed_palette_transfer_v1/verification.json"
    write(seal, dict(passed=True, source_lock_sha256="old"))
    assert not snapshot(**options)["native_quality_verified"]
    write(seal, dict(passed=True, source_lock_sha256=source))
    assert snapshot(**options)["state"] == "verified"
```

</details>

### `test_absent_registration_does_not_invent_a_study` · L219

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_absent_registration_does_not_invent_a_study(tmp_path):
    assert snapshot(root=tmp_path, project=tmp_path, hr_root=tmp_path / "hr") is None
```

</details>

### `test_malformed_nested_selection_is_visible_and_does_not_break_api` · L223

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_malformed_nested_selection_is_visible_and_does_not_break_api(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    write(root / "selections.json", dict(source_lock_sha256=source, roles=[]))
    row = snapshot(**options)
    assert row["state"] == "degraded" and row["warnings"]
    json.dumps(row, allow_nan=False)
```

</details>
