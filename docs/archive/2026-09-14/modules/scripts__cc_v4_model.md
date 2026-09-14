# `scripts/cc_v4_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v4_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Experimental correction-conditioned evidence model in linear camera RGB.

SYNTHETIC correctness checks are not benchmark accuracy or novelty evidence.
Actions are log(R/G), log(B/G) of positive candidate illuminants; their costs
are neutral reproduction costs, never physical surface DeltaE. The 16 initial
local offsets are an explicit proposal prior, with maximum Euclidean norm .06.

Transport uses corrected simplex RGB, a nonlinear transformation of observed
log colors. The action null holds that color at the point while retaining the
relative action input; it controls for a generic action-conditioned router.
All modes contain the same modules. The posterior/direct router fixes its last
two inputs to zero: 128 first-layer scalar weights therefore have no effect in
those modes. Equal declared parameter counts do not imply equal active graphs.
The direct mode uses the point for inference; posterior weights only diagnose it.

SHA-256 исходника: `1c7c4b54c3b07bfbaf9e26a689e08ca0e511532b62403421664ed7c18054ae68`. Строк: **280**.

## Зависимости

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 24](../../../../scripts/cc_v4_model.py#L24)

```python
ANGLE_NORM_EPS = 1e-8
```

[Строка 25](../../../../scripts/cc_v4_model.py#L25)

```python
COLOR_FLOOR = 1e-12
```

[Строка 26](../../../../scripts/cc_v4_model.py#L26)

```python
MAX_ACTION = 4.0
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `CorrectionEvidenceNet` | nn.Module | [L106](../../../../scripts/cc_v4_model.py#L106) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_action_structure` | FunctionDef | См. реализацию | [L29](../../../../scripts/cc_v4_model.py#L29) |
| `_assert_action_domain` | FunctionDef | См. реализацию | [L37](../../../../scripts/cc_v4_model.py#L37) |
| `_costs` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_v4_model.py#L46) |
| `analytic_costs` | FunctionDef | Return angular degrees and exact sin² costs, each NxKxH.  hypotheses: finite floating NxHx2; actions: finite floating NxKx2 with absolute coordinates <= 4. Compute in FP64 if either input is double, otherwise FP32, including under autocast. Positive illuminants are implied by exp([aR, 0, aB]); no ground-truth-specific API or parameter is needed. | [L63](../../../../scripts/cc_v4_model.py#L63) |
| `_illuminant` | FunctionDef | См. реализацию | [L86](../../../../scripts/cc_v4_model.py#L86) |
| `corrected_simplex` | FunctionDef | Physical diagonal correction followed by centered simplex RGB.  Inputs have broadcast-compatible leading dimensions and final dimension2. For mean/RMS log chroma z and correction a, p=softmax([zR-aR,0,zB-aB]); return (3*pR-1,3*pB-1). Bounded output (-1,2) retains actual image color. Its nonlinear dependence on a is not an affine log-input reparameterization; this fact alone does not establish novelty or superior empirical accuracy. | [L92](../../../../scripts/cc_v4_model.py#L92) |
| `CorrectionEvidenceNet` | ClassDef | One cached encoder and an action-conditioned 16-cell evidence router.  Inputs must be nonempty floating NCHW RGB with both spatial dimensions >=32. Negative/nonfinite pixels, all-black rows, or globally absent channels mark the corresponding row invalid. Invalid rows use finite neutral diagnostics and are never accepted. Zero local cells use a declared finite color floor.  Standard MobileNet BatchNorm is preserved. Training a single32x32 image is unsupported (one value/channel); use eval for deployment or a larger batch. Network arithmetic follows parameter dtype: FP32 by default, FP64 after net.double(). Input common scaling precedes FP32 conversion, preserving finite extreme FP64 exposures. Autocast is explicitly disabled here. | [L106](../../../../scripts/cc_v4_model.py#L106) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L121–141</summary>

```python
def __init__(self, mode="transport"):
        super().__init__()
        if mode not in {"transport", "action", "posterior", "direct"}:
            raise ValueError("mode must be transport, action, posterior or direct")
        self.mode = mode
        self.backbone = mobilenet_v3_large(weights=None).features
        self.context_head = nn.Sequential(nn.Linear(960, 64), nn.SiLU())
        self.point_head = nn.Linear(64, 2)
        self.local_project = nn.Linear(960, 48)
        self.vote_head = nn.Sequential(nn.Linear(118, 64), nn.SiLU(), nn.Linear(64, 2))
        self.router = nn.Sequential(nn.Linear(118, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing="ij")
        xy = torch.stack((xx.flatten(), yy.flatten()), -1)
        self.register_buffer("coordinates", xy)
        self.register_buffer("proposal_prior", (xy * (0.06 / math.sqrt(2)) / 0.5).atanh())
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 5), torch.linspace(-1, 1, 5), indexing="ij")
        self.register_buffer("search_offsets", torch.stack((xx.flatten(), yy.flatten()), -1))
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        nn.init.zeros_(self.vote_head[-1].weight)
        nn.init.zeros_(self.vote_head[-1].bias)
```

</details>

<details><summary>forward · L278–280</summary>

```python
def forward(self, image):
        cache = self.encode(image)
        return {**self.select(cache), "context": cache["context"]}
```

</details>
