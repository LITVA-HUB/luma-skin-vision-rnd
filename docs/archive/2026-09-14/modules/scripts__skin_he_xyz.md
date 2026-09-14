# `scripts/skin_he_xyz.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_he_xyz.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Locked source-only controls for real paired facial RGB/XYZ, not DeltaE.

SHA-256 исходника: `e901ed924e533c6f3a0735f800d2c3ee255ce55134fb0ad57ebe31d757eb83db`. Строк: **167**.

## Зависимости

```python
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_he_xyz.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/skin_he_xyz.py#L20)

```python
DATA = ROOT / "data/processed/skin_he_xyz_v1"
```

[Строка 21](../../../../scripts/skin_he_xyz.py#L21)

```python
BENCH = ROOT / "docs/benchmarks/skin_he_xyz_v1"
```

[Строка 22](../../../../scripts/skin_he_xyz.py#L22)

```python
MODELS = ROOT / "experiments/runs/skin_he_xyz_v1"
```

[Строка 23](../../../../scripts/skin_he_xyz.py#L23)

```python
METHODS = ("constant", "linear3", "affine4", "poly2", "poly3", "root2", "mlp_5_25_5")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_he_xyz.py#L26) |
| `read` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_he_xyz.py#L30) |
| `write` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_he_xyz.py#L34) |
| `inputs` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_he_xyz.py#L38) |
| `model` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_he_xyz.py#L44) |
| `metrics` | FunctionDef | См. реализацию | [L54](../../../../scripts/skin_he_xyz.py#L54) |
| `fit` | FunctionDef | См. реализацию | [L65](../../../../scripts/skin_he_xyz.py#L65) |
| `evaluate` | FunctionDef | См. реализацию | [L115](../../../../scripts/skin_he_xyz.py#L115) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L65–112</summary>

```python
def fit():
    if MODELS.exists() or (BENCH/"model_lock.json").exists():
        raise ValueError("Immutable source fit exists")
    source = read(DATA/"train.json")
    assert source["role"] == "train"
    rows = source["rows"]
    assert len(rows) == 200 and len({r["subject"] for r in rows}) == 40
    gt = np.array([r["xyz"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    folds = list(GroupKFold(5).split(gt, groups=groups))
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
    MODELS.mkdir(parents=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    records, files = [], {}
    with threadpool_limits(limits=2):
        for format_name in ("raw", "jpg"):
            x = np.array([r[format_name] for r in rows])
            for name in METHODS:
                xx = inputs(x, name)
                predictions = np.zeros_like(gt)
                messages = []
                for train, val in folds:
                    estimator = model(name)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(xx[train], gt[train])
                    messages.extend(str(w.message) for w in caught)
                    predictions[val] = estimator.predict(xx[val])
                fitted = model(name)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted.fit(xx, gt)
                messages.extend(str(w.message) for w in caught)
                key = format_name+"__"+name
                destination = MODELS/(key+".joblib")
                joblib.dump(fitted, destination)
                files[str(destination.relative_to(ROOT))] = sha(destination)
                np.savez_compressed(MODELS/(key+"_oof.npz"), pred=predictions, gt=gt, ids=np.array([r["id"] for r in rows]))
                record = {"format": format_name, "method": name, "oof": metrics(predictions, gt), "warnings": messages}
                records.append(record)
                print(json.dumps({"format": format_name, "method": name, "oof_xyz_rmse": record["oof"]["xyz_rmse"], "warnings": len(messages)}), flush=True)
    selected = {fmt: min((r for r in records if r["format"] == fmt), key=lambda r:r["oof"]["xyz_rmse"])["method"] for fmt in ("raw", "jpg")}
    write(BENCH/"source_fit.json", {"records": records, "selected_by_source_oof": selected, "folds": [{"train":t.tolist(), "validation":v.tolist()} for t,v in folds], "fit_people":40, "metric":"Original XYZ coordinate error, not perceptual DeltaE"})
    for relative in ("scripts/skin_he_xyz.py", "scripts/skin_he_data.py", "docs/research/skin_he_xyz_protocol_v1.md", "data/processed/skin_he_xyz_v1/train.json", "docs/benchmarks/skin_he_xyz_v1/source_fit.json"):
        files[relative] = sha(ROOT/relative)
    write(BENCH/"model_lock.json", {"status":"ALL14 CONTROLS FROZEN BEFORE SKIN TEST NUMERIC EXTRACTION", "files":files, "selected":selected, "methods":METHODS})
    print(json.dumps({"lock_sha256":sha(BENCH/"model_lock.json"), "selected":selected}), flush=True)
```

</details>
