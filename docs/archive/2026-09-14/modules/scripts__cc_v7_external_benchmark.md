# `scripts/cc_v7_external_benchmark.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_benchmark.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Predict and score the frozen29-method real-camera benchmark; no fitting.

SHA-256 исходника: `d6b3156b6dbe3e046fdf3386dea8111bd8c262325e5af76b0ab2c7c01db93522`. Строк: **157**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from cc_v2_select import features, raw_predict, selective_result
from cc_v2_statistics import predict_arrays
from cc_v7_external_controls import classical_disagreement, fourier_predictions
from cc_v7_external_data import CACHE
from cc_v7_external_lock import BENCH, LOCK, ROOT, checked_lock, load
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.v2 import CompactResidualCC, input_validity, risk_features_invariant
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_cache` | FunctionDef | См. реализацию | [L21](../../../../scripts/cc_v7_external_benchmark.py#L21) |
| `method_prediction` | FunctionDef | См. реализацию | [L33](../../../../scripts/cc_v7_external_benchmark.py#L33) |
| `predict` | FunctionDef | См. реализацию | [L74](../../../../scripts/cc_v7_external_benchmark.py#L74) |
| `populations` | FunctionDef | См. реализацию | [L100](../../../../scripts/cc_v7_external_benchmark.py#L100) |
| `evaluate` | FunctionDef | См. реализацию | [L112](../../../../scripts/cc_v7_external_benchmark.py#L112) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L74–97</summary>

```python
def predict(digest):
    lock=checked_lock(digest)
    check_cache(digest)
    out=BENCH/"predictions"
    if out.exists():
        raise ValueError("Immutable target predictions exist")
    out.mkdir(parents=True)
    raw=load(ROOT/lock["raw_calibration"])["methods"]
    torch.set_num_threads(2)
    outputs={}
    with threadpool_limits(limits=2):
        for dataset in ("external","source_test"):
            rows=load(CACHE/(dataset+"_manifest.json"))
            with np.load(CACHE/(dataset+".npz"),allow_pickle=False) as values:
                # Ground-truth arrays are deliberately not deserialized here.
                images=values["images"].astype(np.float32)
                experts=values["experts"].astype(np.float64)
            for method in lock["methods"]:
                prediction=method_prediction(method,images,experts,raw)
                name=dataset+"__"+method["id"]+".npz"
                np.savez_compressed(out/name,**prediction,ids=np.array([r["id"] for r in rows]))
                outputs[name]=sha256(out/name)
                print(json.dumps({"dataset":dataset,"method":method["id"],"rows":len(rows),"unsupported":int((~prediction["valid"]).sum())}),flush=True)
    write_json(out/"manifest.json",{"method_lock_sha256":sha256(LOCK),"preparation_sha256":sha256(CACHE/"preparation.json"),"script_sha256":sha256(Path(__file__)),"outputs_sha256":outputs,"scope":"Images and cached classical estimates only; no label fitting or evaluation in prediction"})
```

</details>
