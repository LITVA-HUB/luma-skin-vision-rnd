# `tests/test_chromaseed_head_range_verification.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_head_range_verification.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `f2f00efe60ccfd65538a1410166401aa857be91a3d71bc604cc2ca32adf3f2dc`. Строк: **259**.

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
| `test_primary_gate_rejects_nonterminal_or_ambiguous_jobs` | FunctionDef | См. реализацию | [L13](../../../../tests/test_chromaseed_head_range_verification.py#L13) |
| `test_primary_gate_accepts_only_completed_dead_worker` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_head_range_verification.py#L20) |
| `test_independent_selection_keeps_controls_and_stable_ties` | FunctionDef | См. реализацию | [L26](../../../../tests/test_chromaseed_head_range_verification.py#L26) |
| `test_pass_comparison_detects_wrong_intermediate_answer` | FunctionDef | См. реализацию | [L54](../../../../tests/test_chromaseed_head_range_verification.py#L54) |
| `test_result_index_rejects_duplicates_even_if_record_count_is_preserved` | FunctionDef | См. реализацию | [L65](../../../../tests/test_chromaseed_head_range_verification.py#L65) |
| `test_costs_include_failed_partial_without_double_counting_previous_banks` | FunctionDef | См. реализацию | [L73](../../../../tests/test_chromaseed_head_range_verification.py#L73) |
| `test_worker_inventory_excludes_own_wrapper_but_detects_another_fit` | FunctionDef | См. реализацию | [L101](../../../../tests/test_chromaseed_head_range_verification.py#L101) |
| `test_os_process_probe_observes_actual_child_exit` | FunctionDef | См. реализацию | [L128](../../../../tests/test_chromaseed_head_range_verification.py#L128) |
| `test_model_audit_rejects_mistagged_head_and_wrong_native_normalizer` | FunctionDef | См. реализацию | [L145](../../../../tests/test_chromaseed_head_range_verification.py#L145) |
| `test_saved_receipts_cannot_be_silently_replaced` | FunctionDef | См. реализацию | [L161](../../../../tests/test_chromaseed_head_range_verification.py#L161) |
| `test_final_audit_checks_all_passes_of_actual_exported_models` | FunctionDef | См. реализацию | [L173](../../../../tests/test_chromaseed_head_range_verification.py#L173) |

## Все тестовые определения (11)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_primary_gate_rejects_nonterminal_or_ambiguous_jobs` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('status,alive', [('running', False), ('failed', False), ('complete', True), ('complete', None)])
def test_primary_gate_rejects_nonterminal_or_ambiguous_jobs(status, alive):
    from chromaseed_head_range_verification import require_terminal

    with pytest.raises(RuntimeError):
        require_terminal({"status": status, "pid": 123}, probe=lambda _: alive)
```

</details>

### `test_primary_gate_accepts_only_completed_dead_worker` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_primary_gate_accepts_only_completed_dead_worker():
    from chromaseed_head_range_verification import require_terminal

    require_terminal({"status": "complete", "pid": 123}, probe=lambda _: False)
```

</details>

### `test_independent_selection_keeps_controls_and_stable_ties` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_independent_selection_keeps_controls_and_stable_ties():
    from chromaseed_head_range_verification import MODES, PARAMETERS, select_policies

    rows = [
        dict(
            variant=v + "__" + m,
            architecture=v,
            head_mode=m,
            clean=1.0 if m == "unit" else 2.0,
            p90=3.0,
            numeric_bytes=100,
            step=128,
            lr=1e-5,
        )
        for v in PARAMETERS
        for m in MODES
    ]
    first = dict(variant="np", clean=0.5, p90=1.0, numeric_bytes=10, step=0, lr=None)
    second = dict(first, variant="we")
    selected = select_policies(rows + [first, second])
    assert selected["overall"]["variant"] == "np"
    assert len(selected["per_pair"]) == 21 and len(selected["per_architecture"]) == 7
    assert all(r["head_mode"] == "unit" for r in selected["per_architecture"].values())
    rows[0]["clean"] = float("nan")
    with pytest.raises(ValueError):
        select_policies(rows + [first, second])
```

</details>

### `test_pass_comparison_detects_wrong_intermediate_answer` · L54

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pass_comparison_detects_wrong_intermediate_answer():
    from chromaseed_head_range_verification import compare

    expected = np.zeros((4, 3))
    actual = expected.copy()
    actual[0, 1] = 0.01
    with pytest.raises(AssertionError):
        compare(actual, expected)
    assert compare(expected, expected) == 0
```

</details>

### `test_result_index_rejects_duplicates_even_if_record_count_is_preserved` · L65

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_result_index_rejects_duplicates_even_if_record_count_is_preserved():
    from chromaseed_head_range_verification import index_records

    row = dict(role="mixed", variant="patch_small__unit", seed=17)
    with pytest.raises(ValueError):
        index_records([row, row])
```

</details>

### `test_costs_include_failed_partial_without_double_counting_previous_banks` · L73

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_costs_include_failed_partial_without_double_counting_previous_banks():
    from chromaseed_head_range_verification import operational_costs

    banks = [
        dict(
            steps=128,
            full_bank_seconds=10.0,
            setup_seconds=1.0,
            write_and_prediction_inclusive_seconds=11.0,
            cuda_peak_allocated_bytes=100,
        ),
        dict(
            steps=512,
            full_bank_seconds=20.0,
            setup_seconds=2.0,
            write_and_prediction_inclusive_seconds=22.0,
            cuda_peak_allocated_bytes=200,
        ),
    ]
    result = operational_costs(banks, {"step": 1664, "seconds": 357.0525681999861}, 90.0)
    assert result["successful_bank_seconds"] == 30.0
    assert result["observed_fit_work_seconds"] == 387.0525681999861
    assert result["successful_presentations"] == (128 + 512) * 64 * 6
    assert result["discarded_presentations"] == 1664 * 64 * 6
    assert result["resumed_attempt_seconds"] == 90.0
    assert result["complete_original_wall_seconds"] is None
```

</details>

### `test_worker_inventory_excludes_own_wrapper_but_detects_another_fit` · L101

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_worker_inventory_excludes_own_wrapper_but_detects_another_fit():
    from chromaseed_head_range_verification import competing_workers

    rows = [
        dict(
            ProcessId=10, ParentProcessId=9, CommandLine="python chromaseed_head_range_runtime.py"
        ),
        dict(
            ProcessId=9,
            ParentProcessId=8,
            CommandLine="venv python chromaseed_head_range_runtime.py",
        ),
        dict(
            ProcessId=20,
            ParentProcessId=19,
            CommandLine="python chromaseed_head_range_recovery.py resume",
        ),
        dict(
            ProcessId=30, ParentProcessId=29, CommandLine="python apps/training-dashboard/server.py"
        ),
    ]
    assert competing_workers(rows, 10, 9) == [20]
    rows[-1]["CommandLine"] = None
    with pytest.raises(RuntimeError):
        competing_workers(rows, 10, 9)
```

</details>

### `test_os_process_probe_observes_actual_child_exit` · L128

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_os_process_probe_observes_actual_child_exit():
    import subprocess

    from chromaseed_head_range_verification import process_alive

    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        assert process_alive(child.pid) is True
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert process_alive(child.pid) is False
```

</details>

### `test_model_audit_rejects_mistagged_head_and_wrong_native_normalizer` · L145

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_model_audit_rejects_mistagged_head_and_wrong_native_normalizer():
    from chromaseed_head_range import Bank
    from chromaseed_head_range_audit import check_model
    from test_chromaseed_architecture_scale import fixture

    _, tokens, _, warm = fixture()
    model = Bank("soft_small", "wide", seeds=(17,)).export(0, [warm[0]], tokens)
    tm, ts = model["t_mean"].copy(), model["t_std"].copy()
    check_model(model, "soft_small", "wide", warm[0], tm, ts)
    with pytest.raises(AssertionError):
        check_model(model, "soft_small", "linear", warm[0], tm, ts)
    model["base_y_mean"] = model["base_y_mean"] + 1
    with pytest.raises(AssertionError):
        check_model(model, "soft_small", "wide", warm[0], tm, ts)
```

</details>

### `test_saved_receipts_cannot_be_silently_replaced` · L161

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_saved_receipts_cannot_be_silently_replaced(tmp_path):
    from chromaseed_head_range_verification import write_once

    path = tmp_path / "receipt.json"
    write_once(path, dict(passed=True, result=1))
    original = path.read_bytes()
    write_once(path, dict(passed=True, result=1))
    with pytest.raises(RuntimeError):
        write_once(path, dict(passed=True, result=2))
    assert path.read_bytes() == original
```

</details>

### `test_final_audit_checks_all_passes_of_actual_exported_models` · L173

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_final_audit_checks_all_passes_of_actual_exported_models(tmp_path, monkeypatch):
    import chromaseed_head_range_audit as audit
    import torch
    from chromaseed_gated import flatten
    from chromaseed_head_range import Bank, predict_torch
    from chromaseed_head_range_verification import digest
    from test_chromaseed_architecture_scale import fixture

    x, tokens, y, warm = fixture()
    net = Bank("soft_small", "wide")
    with torch.no_grad():
        net.layers["head"].weight.fill_(0.02)
        net.layers["head"].bias.fill_(0.01)
    models = [net.export(i, warm, tokens) for i in range(6)]
    bank = tmp_path / "bank"
    bank.mkdir()
    source = bank / "models_128.npz"
    np.savez(source, **flatten({str(i): m for i, m in enumerate(models)}))
    query = np.arange(3)
    data = dict(
        color=x,
        tokens=tokens,
        target=y,
        patient=np.array([f"p{i}" for i in range(len(x))]),
        site=np.zeros(len(x), int),
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "bank_path", lambda *args: bank)
    monkeypatch.setattr(
        audit,
        "inspect",
        lambda *args: (np.arange(3, len(x)), query, warm, models[0]["t_mean"], models[0]["t_std"]),
    )
    records, saved_outputs = {}, []
    pair = "soft_small__wide"
    for si, seed in enumerate((17, 29, 43)):
        slot = 2 * si
        model_path, output_path = tmp_path / f"model{seed}.npz", tmp_path / f"output{seed}.npz"
        np.savez(model_path, **models[slot])
        passes = predict_torch(models[slot], x[query], tokens[query], device="cpu", all_passes=True)
        output = dict(row_indices=query, predictions=passes[:, -1].copy(), passes=passes)
        saved_outputs.append(output)
        np.savez(output_path, **output)
        records["mixed", pair, seed] = dict(
            role="mixed",
            variant=pair,
            architecture="soft_small",
            head_mode="wide",
            seed=seed,
            slot=slot,
            step=128,
            lr=1e-5,
            source_path="bank/models_128.npz",
            source_sha256=digest(source),
            model=model_path.name,
            model_sha256=digest(model_path),
            output=output_path.name,
            output_sha256=digest(output_path),
            overall=True,
            parameters=15246,
            numeric_bytes=4 * 15246 + 458,
            metrics=audit.error_summary(
                output["predictions"], y[query], data["patient"][query], data["site"][query]
            )[0],
        )
    policies = dict(per_pair={pair: dict(step=128, lr=1e-5)}, overall=dict(variant=pair))
    count_keys = (
        "final_bank_payloads",
        "actual_single_calls",
        "final_models",
        "final_vectors",
        "final_banks",
    )
    counts = dict.fromkeys(count_keys, 0)
    audit.audit_final(data, "mixed", "soft_small", "wide", policies, records, {}, counts)
    assert counts["final_bank_payloads"] == 6 and counts["actual_single_calls"] == 9
    assert counts["final_models"] == 3
    # Keep final predictions/metrics unchanged; alter only the first intermediate answer.
    saved_outputs[0]["passes"][0, 0, 0] += 0.01
    output_path = tmp_path / "output17.npz"
    np.savez(output_path, **saved_outputs[0])
    records["mixed", pair, 17]["output_sha256"] = digest(output_path)
    with pytest.raises(AssertionError):
        audit.audit_final(
            data, "mixed", "soft_small", "wide", policies, records, {}, dict.fromkeys(count_keys, 0)
        )
    assert not torch.cuda.is_initialized()
```

</details>
