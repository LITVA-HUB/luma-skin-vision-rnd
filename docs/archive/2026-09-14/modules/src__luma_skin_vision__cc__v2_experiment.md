# `src/luma_skin_vision/cc/v2_experiment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/v2_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Additive v2 source-only fitting and frozen prediction export.

Training never evaluates test/Sony errors. Prediction writes per-image model
outputs without fitting selectors, evaluating errors, or choosing a method.
NPZ is a compressed array format: numpy materializes an array when indexing;
only selected source rows are retained or transferred to the training device.

SHA-256 исходника: `a8e03ba8f97521d15401341d0b8df0cf43417cc0d6e89adbbfd73146f4ad44fc`. Строк: **470**.

## Зависимости

```python
import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json
from .benchmark import indices
from .core import reproduction
from .model import reproduction_loss
from .v2 import (
    CHEAP_FEATURE_COLUMNS,
    FEATURE_UNITS,
    CompactResidualCC,
    gain_augment,
    input_validity,
    risk_features_invariant,
    validate_image,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 36](../../../../src/luma_skin_vision/cc/v2_experiment.py#L36)

```python
SCHEMA = "cc-v2-1"
```

[Строка 37](../../../../src/luma_skin_vision/cc/v2_experiment.py#L37)

```python
STRESS_GAINS = [[2, 1, 0.5], [0.5, 1, 2], [1, 2, 0.5], [0.7, 0.8, 1.6]]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `data_fingerprints` | FunctionDef | См. реализацию | [L40](../../../../src/luma_skin_vision/cc/v2_experiment.py#L40) |
| `verify_data` | FunctionDef | См. реализацию | [L51](../../../../src/luma_skin_vision/cc/v2_experiment.py#L51) |
| `split_indices` | FunctionDef | Reuse v1 partitions; additionally enforce disjoint source fitting roles. | [L56](../../../../src/luma_skin_vision/cc/v2_experiment.py#L56) |
| `source_snapshot` | FunctionDef | Save exact bytes with the same ordered hash construction as source_identity. | [L81](../../../../src/luma_skin_vision/cc/v2_experiment.py#L81) |
| `_device` | FunctionDef | См. реализацию | [L109](../../../../src/luma_skin_vision/cc/v2_experiment.py#L109) |
| `predict_tensor` | FunctionDef | См. реализацию | [L118](../../../../src/luma_skin_vision/cc/v2_experiment.py#L118) |
| `_validate_args` | FunctionDef | См. реализацию | [L128](../../../../src/luma_skin_vision/cc/v2_experiment.py#L128) |
| `train` | FunctionDef | См. реализацию | [L137](../../../../src/luma_skin_vision/cc/v2_experiment.py#L137) |
| `verify_run` | FunctionDef | См. реализацию | [L314](../../../../src/luma_skin_vision/cc/v2_experiment.py#L314) |
| `_predict_cache` | FunctionDef | См. реализацию | [L350](../../../../src/luma_skin_vision/cc/v2_experiment.py#L350) |
| `predict` | FunctionDef | См. реализацию | [L376](../../../../src/luma_skin_vision/cc/v2_experiment.py#L376) |
| `main` | FunctionDef | См. реализацию | [L438](../../../../src/luma_skin_vision/cc/v2_experiment.py#L438) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L376–435</summary>

```python
def predict(args):
    out, data = Path(args.out).resolve(), Path(args.data).resolve()
    config, checkpoint = verify_run(out, data)
    if args.batch < 1:
        raise ValueError("Prediction batch must be positive")
    device = _device(args.device)
    sources = [("predictions.npz", data / "cube.npz", data / "cube_manifest.json")]
    if "sony.npz" in config["data_hashes"]:
        sources.append(("sony_predictions.npz", data / "sony.npz", data / "sony_manifest.json"))
    if args.external is not None:
        sources.append(
            (
                "external_predictions.npz",
                Path(args.external[0]).resolve(),
                Path(args.external[1]).resolve(),
            )
        )
    targets = [out / name for name, _, _ in sources] + [out / "predictions_manifest.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Prediction diagnostics exist; use --overwrite explicitly to replace")
    torch.set_num_threads(4)
    model = CompactResidualCC(config["mode"], config["backbone"]).to(device).eval()
    model.load_state_dict(
        torch.load(out / "model.pt", weights_only=True, map_location=device)["state"]
    )
    outputs = {}
    for name, npz_path, manifest_path in sources:
        source_hashes = {"npz": sha256(npz_path), "manifest": sha256(manifest_path)}
        values = _predict_cache(model, npz_path, manifest_path, args.batch, device)
        temp = (out / name).with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temp.replace(out / name)
        outputs[name] = {
            "sha256": sha256(out / name),
            "images": len(values["pred"]),
            "invalid_images": int((~values["valid"]).sum()),
            "input_npz": str(npz_path),
            "input_manifest": str(manifest_path),
            "input_hashes": source_hashes,
        }
    write_json(
        out / "predictions_manifest.json",
        {
            "schema_version": SCHEMA,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "config_sha256": checkpoint["config_sha256"],
            "training_source_hash": config["source_hash"],
            "prediction_source_identity": source_identity(),
            "outputs": outputs,
            "context_columns": [f"context_{i:02d}" for i in range(64)],
            "cheap_feature_columns": CHEAP_FEATURE_COLUMNS,
            "cheap_feature_units": FEATURE_UNITS,
            "invariance_scope": "anchored modes, nondegenerate inputs, diagonal channel gains; direct mode has no invariance promise",
            "invalid_policy": "valid=False requires rejection independently of any learned risk score",
            "fitting_or_error_evaluation": False,
        },
    )
    return out / "predictions_manifest.json"
```

</details>
