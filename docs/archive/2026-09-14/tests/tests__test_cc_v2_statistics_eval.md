# `tests/test_cc_v2_statistics_eval.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v2_statistics_eval.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Synthetic-only selector separation and frozen evaluation integrity.

SHA-256 исходника: `0a64d08a999a636360525a9e7cf977b26e1d1976b823eb13b03fd3cc17474e47`. Строк: **116**.

## Зависимости

```python
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `api` | FunctionDef | См. реализацию | [L12](../../../../tests/test_cc_v2_statistics_eval.py#L12) |
| `fixture` | FunctionDef | См. реализацию | [L20](../../../../tests/test_cc_v2_statistics_eval.py#L20) |
| `test_source_only_fit_and_matched_grid` | FunctionDef | См. реализацию | [L55](../../../../tests/test_cc_v2_statistics_eval.py#L55) |
| `test_selection_lock_and_changed_inputs_rejected` | FunctionDef | См. реализацию | [L80](../../../../tests/test_cc_v2_statistics_eval.py#L80) |
| `test_synthetic_external_invalid_refusal_and_bound_domains` | FunctionDef | См. реализацию | [L98](../../../../tests/test_cc_v2_statistics_eval.py#L98) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_source_only_fit_and_matched_grid` · L55

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_source_only_fit_and_matched_grid(api, tmp_path, monkeypatch):
    data, run, out, lock, rows = fixture(api, tmp_path)
    reader, seen = api.statistics.read_npz_rows, []

    def tracked(path, key, selected, **kwargs):
        if key == "gt":
            seen.extend(selected.tolist())
            assert all(rows[i]["subset"] in ("risk", "cal") for i in selected)
        return reader(path, key, selected, **kwargs)

    monkeypatch.setattr(api.statistics, "read_npz_rows", tracked)
    state = api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    assert len(seen) == 31
    assert len(state["fit_ids"]) == 25 and len(state["cal_ids"]) == 6
    assert set(state["heads"]) == {"context", "cheap", "combined"}
    for head in state["heads"].values():
        assert [c["name"] for c in head["candidates"]] == list(api.GRID)
        assert len(head["cal_scores"]) == 6
        assert head["scale"] > 0
    for fold in state["folds"]:
        fit_groups = {rows[10 + i]["group"] for i in fold["train"]}
        val_groups = {rows[10 + i]["group"] for i in fold["validation"]}
        assert fit_groups.isdisjoint(val_groups)
```

</details>

### `test_selection_lock_and_changed_inputs_rejected` · L80

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_lock_and_changed_inputs_rejected(api, tmp_path):
    data, run, out, lock, _ = fixture(api, tmp_path)
    with pytest.raises(ValueError, match="selection lock"):
        api.fit(run, "direct_ridge1", data, out, lock, expected_counts=(25, 6))
    api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    head_lock = tmp_path / "heads_lock.json"
    api.write_json(head_lock, {"statistics_selectors": {}})
    with pytest.raises(ValueError, match="head lock"):
        api.evaluate(out, head_lock, tmp_path / "result.json")
    api.write_json(
        head_lock, {"statistics_selectors": {"gw_ridge1": api.sha256(out / "selection.json")}}
    )
    with (data / "cube_manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="binding"):
        api.evaluate(out, head_lock, tmp_path / "result.json")
```

</details>

### `test_synthetic_external_invalid_refusal_and_bound_domains` · L98

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_synthetic_external_invalid_refusal_and_bound_domains(api, tmp_path):
    data, run, out, lock, _ = fixture(api, tmp_path)
    api.fit(run, "gw_ridge1", data, out, lock, expected_counts=(25, 6))
    head_lock = tmp_path / "heads_lock.json"
    api.write_json(
        head_lock, {"statistics_selectors": {"gw_ridge1": api.sha256(out / "selection.json")}}
    )
    external, manifest = tmp_path / "external.npz", tmp_path / "external.json"
    np.savez_compressed(external, images=np.zeros((2, 3, 8, 8)), gt=np.ones((2, 3)))
    api.write_json(
        manifest, [{"id": str(i), "group": "fresh", "camera": "Synthetic"} for i in range(2)]
    )
    output = tmp_path / "result.json"
    api.evaluate(out, head_lock, output, external=(external, manifest))
    result = json.loads(output.read_text())
    assert set(result["domains"]) == {"source_regression", "fresh_all", "fresh_Synthetic"}
    for record in result["domains"]["fresh_all"].values():
        assert record["invalid_n"] == 2
        assert all(item["n"] == 0 for item in record["frozen_source_thresholds"].values())
```

</details>
