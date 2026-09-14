# `scripts/profile_cc_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/profile_cc_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen CCv2 latency and portable inference; no fitting or accuracy optimization.

SHA-256 исходника: `aeaae6674d453ca3f7a6877099d5d7518ebc3fe25baf3036e3bdf50ae294a4b8`. Строк: **414**.

## Зависимости

```python
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import time
import zipfile
from pathlib import Path
import cv2
import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.data import decode, sample
from luma_skin_vision.cc.v2 import EPS, CompactResidualCC, risk_features_invariant
from luma_skin_vision.cc.v2_experiment import verify_run
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 29](../../../../scripts/profile_cc_v2.py#L29)

```python
ROOT = Path(__file__).resolve().parents[1]
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PortableCC` | torch.nn.Module | [L39](../../../../scripts/profile_cc_v2.py#L39) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load` | FunctionDef | См. реализацию | [L35](../../../../scripts/profile_cc_v2.py#L35) |
| `PortableCC` | ClassDef | FP32 image/model/head by default; optional FP64 sklearn reference head.  Validity is computed on original pixels. Sanitizing only enables finite fallback inference: invalid inputs always have accept80=False. | [L39](../../../../scripts/profile_cc_v2.py#L39) |
| `locked_model` | FunctionDef | См. реализацию | [L81](../../../../scripts/profile_cc_v2.py#L81) |
| `check_bindings` | FunctionDef | См. реализацию | [L115](../../../../scripts/profile_cc_v2.py#L115) |
| `preprocess` | FunctionDef | См. реализацию | [L120](../../../../scripts/profile_cc_v2.py#L120) |
| `source_inputs` | FunctionDef | См. реализацию | [L128](../../../../scripts/profile_cc_v2.py#L128) |
| `timing` | FunctionDef | См. реализацию | [L143](../../../../scripts/profile_cc_v2.py#L143) |
| `profile` | FunctionDef | См. реализацию | [L162](../../../../scripts/profile_cc_v2.py#L162) |
| `export_and_check` | FunctionDef | См. реализацию | [L218](../../../../scripts/profile_cc_v2.py#L218) |
| `main` | FunctionDef | См. реализацию | [L310](../../../../scripts/profile_cc_v2.py#L310) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L46–63</summary>

```python
def __init__(self, model, payload, threshold, head_dtype=torch.float32):
        super().__init__()
        self.model = model
        if payload["block"] != "combined":
            raise ValueError("Portable component requires frozen combined head")
        scaler = payload["model"].named_steps["standardscaler"]
        ridge = payload["model"].named_steps["ridge"]
        if np.asarray(ridge.coef_).shape != (85,):
            raise ValueError("Expected 85-feature scalar Ridge")
        for name, value in {
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "coef": ridge.coef_,
            "intercept": ridge.intercept_,
            "cal_scale": payload["scale"],
            "threshold": threshold,
        }.items():
            self.register_buffer(name, torch.as_tensor(np.asarray(value), dtype=head_dtype))
```

</details>

<details><summary>forward · L65–78</summary>

```python
def forward(self, image):
        finite = torch.isfinite(image)
        valid = finite.flatten(1).all(1) & (image >= 0).flatten(1).all(1)
        valid = valid & (image.mean((-2, -1)) > EPS).all(1)
        safe = torch.where(finite, image, torch.zeros_like(image)).clamp_min(0)
        pred, context = self.model(safe)
        feature = risk_features_invariant(safe, pred, context)["combined"].to(self.mean.dtype)
        raw = ((feature - self.mean) / self.scale * self.coef).sum(1) + self.intercept
        clipped = raw.clamp(0, math.log(181))
        # ONNX has Exp but no Expm1. Exp-1 is algebraically equivalent;
        # explicit runtime parity checks bound the cancellation error.
        transformed = clipped.exp() - 1 if torch.onnx.is_in_onnx_export() else torch.expm1(clipped)
        score = (transformed + 1e-8) * self.cal_scale
        return pred, score, valid, valid & (score <= self.threshold)
```

</details>
