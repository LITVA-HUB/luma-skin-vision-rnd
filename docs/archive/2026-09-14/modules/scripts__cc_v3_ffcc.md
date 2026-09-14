# `scripts/cc_v3_ffcc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v3_ffcc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

FFCC-inspired, source-grounded compact control; not benchmark reproduction.

Only floating linear RGB with explicit [0,1] full scale is accepted. No resize,
black-level subtraction, quantization, chart/saturation detection, data loading,
optimizer or pretrained weights is included. Real spatial parameters preserve
the full-FFT quadratic objective but differ from packed/preconditioned minFunc
optimization. Zero circular resultants are explicitly undefined, with finite
diagnostic decoding and confidence0; train cross entropy before Gaussian NLL.

Batch-first API: RGB Nx3xHxW; mask NxHxW; hist Nx2xnxn; PMF Nxnxn.
All histograms and toroidal moments are computed in FP64. The model casts
histograms to parameter dtype for FFT filtering; model.double() gives FP64
filtering too. Nearest histogram inputs are nondifferentiable; filters/bias,
moment decode and objectives are differentiable away from chart switches.

SHA-256 исходника: `f451744a67ff4c2d58cb05c34d4af0689332c6acd830904b696c7dac27e16391`. Строк: **328**.

## Зависимости

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 45](../../../../scripts/cc_v3_ffcc.py#L45)

```python
SOURCE_COMMIT = "2fa9e1316954dbd3913630b7d597927941b4dd32"
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `FFCCInspiredNet` | nn.Module | [L303](../../../../scripts/cc_v3_ffcc.py#L303) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `round_matlab` | FunctionDef | Round nearest with half ties away from zero, as MATLAB round. | [L48](../../../../scripts/cc_v3_ffcc.py#L48) |
| `_validate_grid` | FunctionDef | См. реализацию | [L53](../../../../scripts/cc_v3_ffcc.py#L53) |
| `rgb_to_uv` | FunctionDef | Positive RGB Nx3 -> (log G-log R, log G-log B); exposure cancels. | [L58](../../../../scripts/cc_v3_ffcc.py#L58) |
| `uv_to_rgb` | FunctionDef | Projective illuminant decode; remove common log scale before exponent. | [L66](../../../../scripts/cc_v3_ffcc.py#L66) |
| `periodic_histogram` | FunctionDef | Nearest periodic unit-count histogram: uv NxPx2; u rows, v columns.  Each histogram normalizes separately. Empty histograms stay exactly zero; returned count is the number of included pixels, before normalization. | [L73](../../../../scripts/cc_v3_ffcc.py#L73) |
| `masked_local_absolute_deviation` | FunctionDef | Source double path: eight masked neighbors, replicated image/mask edges.  No valid neighbors yields NaN, which featurize explicitly excludes. No integer casting/bitshift path is implemented. | [L93](../../../../scripts/cc_v3_ffcc.py#L93) |
| `featurize` | FunctionDef | Two independently normalized histograms and source all-pixel linear mean.  Float full scale1 resolves the source's ambiguous isa(im,'float') branch. Default mask follows GehlerShi MASK_ZERO_PIXELS. An explicit mask affects histograms only: average_rgb uses every original spatial pixel, including existing zero-filled pixels, exactly as PrecomputeTrainingData.m specifies. | [L120](../../../../scripts/cc_v3_ffcc.py#L120) |
| `circular_score` | FunctionDef | MATLAB fft2/ifft2 convention: no conjugation, shift or gain map. | [L158](../../../../scripts/cc_v3_ffcc.py#L158) |
| `decode` | FunctionDef | Circular mean and source rounded-chart covariance, with pad only.  Mean validity requires both circular resultant lengths>1e-10. Undefined axes get a finite zero-index diagnostic; confidence is then0. This explicit refusal is an adaptation, not a confident source estimate for uniform PMFs. Output covariance and means use FP64; no extra covariance jitter is added. | [L168](../../../../scripts/cc_v3_ffcc.py#L168) |
| `target_distribution` | FunctionDef | Source SMOOTH_CROSS_ENTROPY=true means nearest; false means bilinear. | [L224](../../../../scripts/cc_v3_ffcc.py#L224) |
| `_reduce` | FunctionDef | См. реализацию | [L245](../../../../scripts/cc_v3_ffcc.py#L245) |
| `cross_entropy` | FunctionDef | См. реализацию | [L255](../../../../scripts/cc_v3_ffcc.py#L255) |
| `gaussian_uv_nll` | FunctionDef | Shifted Gaussian UV likelihood from source; residual is NOT wrapped.  Requires defined decoded means for every row. Undefined circular means require CE warmup or an explicit caller policy; no zero-loss masking occurs. This is not the exact BVM density and is not an RGB angular training loss. | [L261](../../../../scripts/cc_v3_ffcc.py#L261) |
| `fft_regularizer` | FunctionDef | Full-FFT quadratic source objective, including MATLAB/Parseval scaling.  For averaged data loss use data_mass1; for a sum use the sum of sample weights. This is not L1 TV and is not interchangeable with AdamW decay. | [L279](../../../../scripts/cc_v3_ffcc.py#L279) |
| `FFCCInspiredNet` | ClassDef | Two spatial-domain filters plus bias, default12,288 real parameters. | [L303](../../../../scripts/cc_v3_ffcc.py#L303) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L306–312</summary>

```python
def __init__(self, n=64, h=1/32, lo=-.4375, eps_bins=1., unwrap_mode="gray_light"):
        super().__init__()
        _validate_grid(n, h, lo)
        self.n, self.h, self.lo = n, h, lo
        self.eps_bins, self.unwrap_mode = eps_bins, unwrap_mode
        self.filters = nn.Parameter(torch.zeros(2, n, n))
        self.bias = nn.Parameter(torch.zeros(n, n))
```

</details>

<details><summary>forward · L322–325</summary>

```python
def forward(self, rgb, mask=None):
        features = featurize(rgb, mask, self.n, self.h, self.lo)
        output = self.forward_histograms(features["hist"], features["average_rgb"])
        return {**features, **output}
```

</details>
