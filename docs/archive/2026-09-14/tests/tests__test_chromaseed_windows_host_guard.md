# `tests/test_chromaseed_windows_host_guard.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_windows_host_guard.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Measured WDDM activity must distinguish a desktop context from a competing fit.

SHA-256 исходника: `97c41a52fa13ffc5ac392bef273d301cf5f1d11fffdc250fdc5e5668031cfbf5`. Строк: **180**.

## Зависимости

```python
import copy
import sys
from pathlib import Path
import pytest
from chromaseed_windows_host_guard import assess, make_registry
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fixture` | FunctionDef | См. реализацию | [L14](../../../../tests/test_chromaseed_windows_host_guard.py#L14) |
| `test_desktop_gpu_context_is_permitted_only_with_low_measured_activity` | FunctionDef | См. реализацию | [L74](../../../../tests/test_chromaseed_windows_host_guard.py#L74) |
| `test_unsafe_or_unmeasurable_host_is_rejected` | FunctionDef | См. реализацию | [L102](../../../../tests/test_chromaseed_windows_host_guard.py#L102) |
| `test_own_cuda_is_excluded_but_another_compute_worker_is_not` | FunctionDef | См. реализацию | [L142](../../../../tests/test_chromaseed_windows_host_guard.py#L142) |
| `test_registry_cannot_whitelist_an_arbitrary_compute_application` | FunctionDef | См. реализацию | [L160](../../../../tests/test_chromaseed_windows_host_guard.py#L160) |
| `test_closed_desktop_process_is_allowed_without_allowing_pid_reuse` | FunctionDef | См. реализацию | [L167](../../../../tests/test_chromaseed_windows_host_guard.py#L167) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_desktop_gpu_context_is_permitted_only_with_low_measured_activity` · L74

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_desktop_gpu_context_is_permitted_only_with_low_measured_activity():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    value = assess(snap, registry, 10, 9)
    assert value["passed"] and value["maximum_background_engine_percent"] == 0.4
    assert value["maximum_background_compute_percent"] == 0
    assert set(registry["desktop_instances"]) == {"20", "4"}
```

</details>

### `test_unsafe_or_unmeasurable_host_is_rejected` · L102

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('change', ['worker', 'unknown_gpu', 'reused_pid', 'driver', 'uuid', 'compute', 'graphics', 'missing_samples', 'nan', 'status', 'unreadable_python', 'missing_counter', 'aggregate', 'unknown_active'])
def test_unsafe_or_unmeasurable_host_is_rejected(change):
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    if change == "worker":
        snap["processes"][-1]["CommandLine"] = "python skin_face_transfer_run.py run"
    elif change == "unknown_gpu":
        snap["nvidia_pids"].append(99)
    elif change == "reused_pid":
        snap["processes"][2]["CreationDate"] = "new desktop process"
    elif change in ("driver", "uuid"):
        snap["driver" if change == "driver" else "gpu_uuid"] = "changed"
    elif change == "compute":
        snap["samples"][1]["counters"][1]["value"] = 0.51
    elif change == "graphics":
        snap["samples"][1]["counters"][0]["value"] = 10.1
    elif change == "missing_samples":
        snap["samples"] = snap["samples"][:1]
    elif change == "nan":
        snap["samples"][0]["counters"][0]["value"] = float("nan")
    elif change == "status":
        snap["samples"][0]["counters"][0]["status"] = 0xC0000BC6
    elif change == "unreadable_python":
        snap["processes"][-1]["CommandLine"] = None
    elif change == "missing_counter":
        for sample in snap["samples"]:
            sample["counters"] = [r for r in sample["counters"] if "compute" not in r["instance"]]
    elif change == "aggregate":
        for sample in snap["samples"]:
            sample["counters"][0]["value"] = 6
            extra = copy.deepcopy(sample["counters"][0])
            extra["instance"] = extra["instance"].replace("pid_20_", "pid_4_")
            sample["counters"].append(extra)
    elif change == "unknown_active":
        snap["samples"][0]["counters"][0]["instance"] = snap["samples"][0]["counters"][0][
            "instance"
        ].replace("pid_20_", "pid_99_")
    with pytest.raises((ValueError, RuntimeError)):
        assess(snap, registry, 10, 9)
```

</details>

### `test_own_cuda_is_excluded_but_another_compute_worker_is_not` · L142

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_own_cuda_is_excluded_but_another_compute_worker_is_not():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    snap["nvidia_pids"].append(10)
    for sample in snap["samples"]:
        sample["counters"].append(
            dict(
                instance="pid_10_luid_0x00000000_0x00000001_phys_0_eng_1_engtype_compute 0",
                value=100.0,
                status=0,
            )
        )
    assert assess(snap, registry, 10, 9)["passed"]
    snap["nvidia_pids"].append(30)
    with pytest.raises(RuntimeError):
        assess(snap, registry, 10, 9)
```

</details>

### `test_registry_cannot_whitelist_an_arbitrary_compute_application` · L160

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_registry_cannot_whitelist_an_arbitrary_compute_application():
    snap = fixture()
    snap["processes"][2]["Name"] = "trainer.exe"
    with pytest.raises(RuntimeError):
        make_registry(snap, 10, 9)
```

</details>

### `test_closed_desktop_process_is_allowed_without_allowing_pid_reuse` · L167

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_closed_desktop_process_is_allowed_without_allowing_pid_reuse():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    snap["nvidia_pids"] = []
    snap["processes"] = [p for p in snap["processes"] if p["ProcessId"] != 20]
    for sample in snap["samples"]:
        sample["counters"] = [
            dict(
                instance="pid_4_luid_0x00000000_0x00000001_phys_0_eng_1_engtype_compute 0",
                value=0,
                status=0,
            )
        ]
    assert assess(snap, registry, 10, 9)["passed"]
```

</details>
