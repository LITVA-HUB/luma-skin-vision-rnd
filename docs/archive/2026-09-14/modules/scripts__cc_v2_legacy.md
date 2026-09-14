# `scripts/cc_v2_legacy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_legacy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen v1 comparators and source-only confidence-head upgrade, no retraining.

The original checkpoints/results remain untouched. Only source risk/cal labels
fit new heads; fresh camera labels are opened solely by the evaluate action.

SHA-256 исходника: `3a1d6b89a689097179dfd650d71dc12126bcf604171c6e72dd787ae2fd4c919a`. Строк: **283**.

## Зависимости

```python
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import torch
from cc_v2_select import estimator, raw_predict, selective_result
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.benchmark import apply_risk, data_hashes, indices, predict, risk_features
from luma_skin_vision.cc.core import EXPERT_NAMES, reproduction, selective_curve
from luma_skin_vision.cc.model import CompactCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `verify_legacy` | FunctionDef | См. реализацию | [L27](../../../../scripts/cc_v2_legacy.py#L27) |
| `model_load` | FunctionDef | См. реализацию | [L54](../../../../scripts/cc_v2_legacy.py#L54) |
| `predictions` | FunctionDef | См. реализацию | [L61](../../../../scripts/cc_v2_legacy.py#L61) |
| `fit` | FunctionDef | См. реализацию | [L69](../../../../scripts/cc_v2_legacy.py#L69) |
| `evaluate` | FunctionDef | См. реализацию | [L144](../../../../scripts/cc_v2_legacy.py#L144) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L69–141</summary>

```python
def fit(run, data, out):
    config = verify_legacy(run, data)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    ix = indices(rows, config["protocol"])
    selected = np.r_[ix["risk"], ix["cal"]]
    cache = np.load(data / "cube.npz", allow_pickle=False)
    # Only selected rows enter neural prediction or any target computation.
    x = cache["images"][selected].astype(np.float32)
    ex = cache["experts"][selected]
    gt = cache["gt"][selected]
    model = model_load(run, device="cpu")
    pred, context = predict(model, torch.tensor(x), torch.tensor(ex, dtype=torch.float32))
    errors, n = reproduction(pred, gt), len(ix["risk"])
    groups = np.array([rows[i]["group"] for i in ix["risk"]])
    ids = [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(5).split(np.arange(n), groups=groups))
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_run": str(run.resolve()),
        "checkpoint_sha256": sha256(run / "model.pt"),
        "legacy_config_sha256": sha256(run / "config.json"),
        "legacy_risk_heads_sha256": sha256(run / "risk_heads.json"),
        "source_hashes": config["data_hashes"],
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "selection": "five risk-date-group OOF folds; minimumrisk80 thenAURC; standard positive multiplicative calibration on separatecal",
        "scope": "source risk/cal labels ONLY; no external or regression target errors; no estimator retraining",
        "source_feature_device": "CPU FP32; avoid concurrent GPU load during estimator training",
        "heads": {},
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("cc_v2_select.py")),
    }
    payloads = {}
    for block in ("context", "combined"):
        x = risk_features(context, pred, ex, block)
        candidates = []
        for name in ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7"):
            oof = np.empty(n)
            for tr, va in folds:
                head = estimator(name).fit(x[tr], np.log1p(errors[tr]))
                oof[va] = raw_predict(head, x[va])
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
        head = estimator(best["name"]).fit(x[:n], np.log1p(errors[:n]))
        cal_raw = raw_predict(head, x[n:])
        scale = max(1e-8, float(errors[n:].mean() / cal_raw.mean()))
        payloads[block] = {"model": head, "scale": scale}
        metadata["heads"][block] = {
            "selected": best["name"],
            "candidates": candidates,
            "cal_scores": (cal_raw * scale).tolist(),
            "scale": scale,
        }
    joblib.dump(payloads, out / "heads.joblib")
    metadata["artifact_sha256"] = sha256(out / "heads.joblib")
    write_json(out / "selection.json", metadata)
    print(
        json.dumps(
            {"legacy": run.name, "heads": {b: h["selected"] for b, h in metadata["heads"].items()}}
        ),
        flush=True,
    )
```

</details>
