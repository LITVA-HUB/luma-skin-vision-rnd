# `tests/test_cc_v5_report.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v5_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scoring and artifact-integrity regression checks.

SHA-256 исходника: `adcfea8c1ab487542ff3d000b447d602e3d4bb0df62eedbcb4b5c629d86e9547`. Строк: **47**.

## Зависимости

```python
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `reporter` | FunctionDef | См. реализацию | [L13](../../../../tests/test_cc_v5_report.py#L13) |
| `test_independent_physical_error_and_integer_coverage` | FunctionDef | См. реализацию | [L22](../../../../tests/test_cc_v5_report.py#L22) |
| `test_manifest_rejects_modified_payload_and_parent_paths` | FunctionDef | См. реализацию | [L36](../../../../tests/test_cc_v5_report.py#L36) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_independent_physical_error_and_integer_coverage` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_independent_physical_error_and_integer_coverage():
    r = reporter()
    pred = np.array([[1., 1., 1.], [2., 1., 1.]])
    gt = np.array([[2., 1., 1.], [2., 1., 1.]])
    error = r.reproduction_degrees(pred, gt)
    assert error.tolist() == pytest.approx([19.47122063449, 0.], abs=1e-10)
    result = r.selective(np.array([1., 2., 9., 20., 30.]), np.array([3., 2., 1., 4., 5.]))
    assert result['fixed']['80']['accepted'] == 4
    assert result['fixed']['60']['mean'] == 4.
    assert len(result['curve']) == 5
    with pytest.raises(ValueError):
        r.reproduction_degrees(np.zeros((1, 3)), np.ones((1, 3)))
```

</details>

### `test_manifest_rejects_modified_payload_and_parent_paths` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_manifest_rejects_modified_payload_and_parent_paths(tmp_path):
    r = reporter()
    (tmp_path / 'payload').write_bytes(b'abc')
    (tmp_path / 'artifact_manifest.json').write_text(json.dumps({'sha256': {
        'payload': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}}))
    r.verify_manifest(tmp_path)
    (tmp_path / 'payload').write_bytes(b'abd')
    with pytest.raises(ValueError):
        r.verify_manifest(tmp_path)
    (tmp_path / 'artifact_manifest.json').write_text(json.dumps({'sha256': {'../outside': 'bad'}}))
    with pytest.raises(ValueError):
        r.verify_manifest(tmp_path)
```

</details>
