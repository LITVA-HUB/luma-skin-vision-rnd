# `scripts/cc_v2_select.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_select.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fit source-only risk selectors, then evaluate frozen predictions independently.

No target labels enter fit(): even source GT arrays are sliced to risk/cal first.
All candidates receive identical date-group CV; physical DeltaE is unavailable.

SHA-256 исходника: `cea9e3b55e7a887766fc9cee79346390cd5197fc666d3121134852946bc15686`. Строк: **369**.

## Зависимости

```python
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.benchmark import result
from luma_skin_vision.cc.core import reproduction, selective_curve, summarize
from luma_skin_vision.cc.v2_experiment import split_indices, verify_run
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `estimator` | FunctionDef | См. реализацию | [L28](../../../../scripts/cc_v2_select.py#L28) |
| `features` | FunctionDef | См. реализацию | [L42](../../../../scripts/cc_v2_select.py#L42) |
| `raw_predict` | FunctionDef | См. реализацию | [L50](../../../../scripts/cc_v2_select.py#L50) |
| `verify_predictions` | FunctionDef | См. реализацию | [L54](../../../../scripts/cc_v2_select.py#L54) |
| `selective_result` | FunctionDef | См. реализацию | [L79](../../../../scripts/cc_v2_select.py#L79) |
| `fit` | FunctionDef | См. реализацию | [L130](../../../../scripts/cc_v2_select.py#L130) |
| `evaluate` | FunctionDef | См. реализацию | [L230](../../../../scripts/cc_v2_select.py#L230) |
| `main` | FunctionDef | См. реализацию | [L350](../../../../scripts/cc_v2_select.py#L350) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L130–227</summary>

```python
def fit(run, data):
    config, checkpoint = verify_run(run, data)
    prediction_manifest = verify_predictions(run, data)
    source_input = prediction_manifest["outputs"]["predictions.npz"]
    if (
        Path(source_input["input_npz"]).resolve() != (data / "cube.npz").resolve()
        or Path(source_input["input_manifest"]).resolve() != (data / "cube_manifest.json").resolve()
    ):
        raise ValueError("Source prediction input differs from fitting dataset")
    out = run / "risk_v2"
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = split_indices(rows, config["protocol"])
    values = np.load(run / "predictions.npz", allow_pickle=False)
    if values["ids"].tolist() != [r["id"] for r in rows]:
        raise ValueError("Prediction/manifest ID mismatch")
    selected = np.r_[ix["risk"], ix["cal"]]
    with np.load(data / "cube.npz", allow_pickle=False) as cache:
        gt = cache["gt"][selected]
    pred = values["pred"][selected]
    valid = values["valid"][selected]
    if not valid.all():
        raise ValueError("Unsupported source risk/cal rows require explicit protocol revision")
    errors = reproduction(pred, gt)
    n = len(ix["risk"])
    groups = np.array([rows[i]["group"] for i in ix["risk"]])
    ids = [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=groups))
    out.mkdir()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "risk-fit and calibration source capture groups ONLY; no test or external errors",
        "selection": "lowest pooled 5-date-group-OOF risk at80%; AURC tie break",
        "target": "log1p reproduction angular error in degrees; no physical surface DeltaE",
        "calibration": "positive scalar mean-error/mean-predicted-error on separate cal; preserves ranking",
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "prediction_sha256": sha256(run / "predictions.npz"),
        "predictions_manifest_sha256": sha256(run / "predictions_manifest.json"),
        "script_sha256": sha256(Path(__file__)),
        "heads": {},
    }
    for block in ("context", "cheap", "combined"):
        x = features(values, block)[selected]
        candidates = []
        for name in ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7"):
            oof = np.zeros(n)
            for tr, va in folds:
                model = estimator(name)
                model.fit(x[tr], np.log1p(errors[tr]))
                oof[va] = raw_predict(model, x[va])
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
        model = estimator(best["name"])
        model.fit(x[:n], np.log1p(errors[:n]))
        raw_cal = raw_predict(model, x[n:])
        scale = max(1e-8, float(errors[n:].mean() / raw_cal.mean()))
        payload = {"model": model, "scale": scale, "block": block}
        joblib.dump(payload, out / (block + ".joblib"))
        report["heads"][block] = {
            "selected": best["name"],
            "features": x.shape[1],
            "candidates": candidates,
            "scale": scale,
            "cal_scores": (raw_cal * scale).tolist(),
            "cal_errors": errors[n:].tolist(),
            "cal_mae": float(np.abs(raw_cal * scale - errors[n:]).mean()),
            "artifact_sha256": sha256(out / (block + ".joblib")),
        }
    write_json(out / "selection.json", report)
    print(
        json.dumps(
            {
                "run": run.name,
                "heads": {
                    k: {
                        "selected": v["selected"],
                        "risk80": next(
                            c["risk80"] for c in v["candidates"] if c["name"] == v["selected"]
                        ),
                    }
                    for k, v in report["heads"].items()
                },
            }
        ),
        flush=True,
    )
```

</details>
