# `scripts/cc_v2_statistics_eval.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_statistics_eval.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-only statistics selectors; target evaluation requires a separate head lock.

SHA-256 исходника: `5b2c2651d0184e5b0fd12e49c2eb61a378723dce1c89e2ead223bf0fbe115a5b`. Строк: **310**.

## Зависимости

```python
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.core import reproduction, selective_curve
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 27](../../../../scripts/cc_v2_statistics_eval.py#L27)

```python
GRID = ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7")
```

[Строка 28](../../../../scripts/cc_v2_statistics_eval.py#L28)

```python
BLOCKS = ("context", "cheap", "combined")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sibling` | FunctionDef | См. реализацию | [L19](../../../../scripts/cc_v2_statistics_eval.py#L19) |
| `load` | FunctionDef | См. реализацию | [L31](../../../../scripts/cc_v2_statistics_eval.py#L31) |
| `bindings` | FunctionDef | См. реализацию | [L35](../../../../scripts/cc_v2_statistics_eval.py#L35) |
| `check_bindings` | FunctionDef | См. реализацию | [L39](../../../../scripts/cc_v2_statistics_eval.py#L39) |
| `prediction` | FunctionDef | См. реализацию | [L47](../../../../scripts/cc_v2_statistics_eval.py#L47) |
| `fit` | FunctionDef | См. реализацию | [L73](../../../../scripts/cc_v2_statistics_eval.py#L73) |
| `evaluate` | FunctionDef | См. реализацию | [L183](../../../../scripts/cc_v2_statistics_eval.py#L183) |
| `main` | FunctionDef | См. реализацию | [L271](../../../../scripts/cc_v2_statistics_eval.py#L271) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L73–180</summary>

```python
def fit(run, candidate, data, out, estimator_lock, *, expected_counts=(259, 268)):
    run, data, out, estimator_lock = map(Path, (run, data, out, estimator_lock))
    lock, screen = load(estimator_lock), load(run / "screen.json")
    if candidate not in lock["statistics_controls"] or lock[
        "statistics_source_screen_sha256"
    ] != sha256(run / "screen.json"):
        raise ValueError("Estimator selection lock mismatch")
    if out.exists():
        raise FileExistsError(out)
    if str(data.resolve()) != screen["data_root"] or screen["data_hashes"] != {
        name: sha256(data / name) for name in screen["data_hashes"]
    }:
        raise ValueError("Source fitting dataset binding mismatch")
    model = next(r for r in screen["candidates"] if r["candidate"] == candidate)
    frozen = bindings(
        [
            Path(__file__),
            Path(statistics.__file__),
            Path(selector.__file__),
            run / "screen.json",
            run / model["model_file"],
            estimator_lock,
            data / "cube.npz",
            data / "cube_manifest.json",
        ]
    )
    rows, values = prediction(
        run,
        candidate,
        data / "cube.npz",
        data / "cube_manifest.json",
        out / "source_predictions.npz",
    )
    ix = split_indices(rows, "official")
    if (len(ix["risk"]), len(ix["cal"])) != tuple(expected_counts):
        raise ValueError("Unexpected immutable risk/cal split sizes")
    selected = np.r_[ix["risk"], ix["cal"]]
    ordered = np.sort(selected)
    gt = statistics.read_npz_rows(data / "cube.npz", "gt", ordered, expected_rows=len(rows))[
        np.searchsorted(ordered, selected)
    ]
    if not values["valid"][selected].all():
        raise ValueError("Unsupported source risk/cal inputs require protocol revision")
    errors = reproduction(values["pred"][selected], gt)
    n, ids = len(ix["risk"]), [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(5).split(np.arange(n), groups=[rows[i]["group"] for i in ix["risk"]]))
    frozen.update(
        bindings([out / "source_predictions.npz", out / "source_predictions.manifest.json"])
    )
    state = {
        "candidate": candidate,
        "run": str(run.resolve()),
        "data": str(data.resolve()),
        "source_hash": screen["source_identity"]["source_hash"],
        "bindings": frozen,
        "scope": "Source risk-fit/calibration only; no target labels or errors",
        "selection": "minimum pooled 5-date-group OOF risk80, AURC tie-break; identical CNN grid",
        "calibration": "positive scalar mean(error)/mean(predicted error) on separate source cal only; preserves ranking",
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "fit_errors": errors[:n].tolist(),
        "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds],
        "heads": {},
    }
    with threadpool_limits(limits=4):
        for block in BLOCKS:
            x, candidates = selector.features(values, block)[selected], []
            for name in GRID:
                oof = np.zeros(n)
                for tr, va in folds:
                    head = selector.estimator(name).fit(x[tr], np.log1p(errors[tr]))
                    oof[va] = selector.raw_predict(head, x[va])
                curve = selective_curve(errors[:n], oof, ids)
                candidates.append(
                    {
                        "name": name,
                        "risk80": curve["fixed"]["80"]["mean"],
                        "aurc": curve["aurc"],
                        "oof_scores": oof.tolist(),
                    }
                )
            best = min(candidates, key=lambda c: (c["risk80"], c["aurc"]))
            head = selector.estimator(best["name"]).fit(x[:n], np.log1p(errors[:n]))
            raw_cal = selector.raw_predict(head, x[n:])
            scale = max(1e-8, float(errors[n:].mean() / raw_cal.mean()))
            artifact = out / (block + ".joblib")
            joblib.dump({"model": head, "scale": scale, "block": block}, artifact)
            state["heads"][block] = {
                "selected": best["name"],
                "features": x.shape[1],
                "candidates": candidates,
                "scale": scale,
                "cal_scores": (raw_cal * scale).tolist(),
                "cal_errors": errors[n:].tolist(),
                "cal_mae": float(np.abs(raw_cal * scale - errors[n:]).mean()),
                "artifact_sha256": sha256(artifact),
            }
    check_bindings(frozen)
    if source_identity()["source_hash"] != state["source_hash"]:
        raise ValueError("Source implementation changed during fit")
    write_json(out / "selection.json", state)
    print(
        json.dumps(
            {"candidate": candidate, "heads": {b: h["selected"] for b, h in state["heads"].items()}}
        ),
        flush=True,
    )
    return state
```

</details>
