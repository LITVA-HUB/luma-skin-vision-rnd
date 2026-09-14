# `tests/test_chromaseed_head_range_host_recovery.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_head_range_host_recovery.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

The host-only adapter must preserve numerical calls and bind effective execution.

SHA-256 исходника: `288558cbcbe75c40d0a546a8aaf1f454e0ca9ea880edcbefdedbd913659e80ca`. Строк: **75**.

## Зависимости

```python
import sys
from pathlib import Path
from types import SimpleNamespace
from chromaseed_head_range_host_recovery import install_runtime, render_disclosure
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_adapter_keeps_numerical_functions_and_binds_every_runtime_output` | FunctionDef | См. реализацию | [L12](../../../../tests/test_chromaseed_head_range_host_recovery.py#L12) |
| `test_disclosure_preserves_original_report_and_states_measurement_scope` | FunctionDef | См. реализацию | [L70](../../../../tests/test_chromaseed_head_range_host_recovery.py#L70) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_adapter_keeps_numerical_functions_and_binds_every_runtime_output` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_adapter_keeps_numerical_functions_and_binds_every_runtime_output(tmp_path):
    calls, written = [], []

    def numerical(*args):
        calls.append(args)
        return {"unchanged": True}

    module = SimpleNamespace(
        require_quiet_host=lambda: None,
        verify_contract=lambda **kw: {"sources": {"original.py": "old"}, "counts": 42},
        binding=lambda: {"original_protocol": "old"},
        check_hashes=lambda mapping: None,
        write_once=lambda path, value: written.append((path, value)),
        OUT=tmp_path,
        response=numerical,
        replay=numerical,
        fit=numerical,
        upstream_fit=numerical,
        Predictor=numerical,
        predict_torch=numerical,
        compare=numerical,
        exact=numerical,
    )
    guard = SimpleNamespace(__call__=lambda: None)
    supplement = {"adapter.py": "new", "protocol.json": "contract"}

    def evidence():
        return {"snapshot.json": "evidence"}, {"passed": True, "checks": 232}

    install_runtime(module, guard, supplement, "contract", evidence)
    for name in (
        "response",
        "replay",
        "fit",
        "upstream_fit",
        "Predictor",
        "predict_torch",
        "compare",
        "exact",
    ):
        assert getattr(module, name) is numerical
    assert module.verify_contract()["sources"] == {"original.py": "old", **supplement}
    assert module.binding()["host_guard_protocol_sha256"] == "contract"
    module.write_once(tmp_path / "runtime.json", {"artifact_sha256": {"weights": "original"}})
    result = written[-1][1]
    assert result["artifact_sha256"] == {
        "weights": "original",
        **supplement,
        "snapshot.json": "evidence",
    }
    assert result["host_guard"]["checks"] == 232
    assert module.require_quiet_host is guard
    receipt = {"value": 1}
    module.write_once(Path("response.json"), receipt)
    assert written[-1][1] is receipt
    assert not calls
```

</details>

### `test_disclosure_preserves_original_report_and_states_measurement_scope` · L70

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_disclosure_preserves_original_report_and_states_measurement_scope():
    original = "# Original report\n\nAll numerical comparisons.\n"
    result = render_disclosure(original, "a" * 64)
    assert result.startswith(original)
    assert "WDDM" in result and "a" * 64 in result
    assert "0,5" in result and "10%" in result
```

</details>
