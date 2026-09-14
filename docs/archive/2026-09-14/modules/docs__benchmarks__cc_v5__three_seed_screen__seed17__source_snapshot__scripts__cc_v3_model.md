# `docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Experimental color-frame graph posterior; constructed correctness is not accuracy evidence.

Frames are expressed after a positive per-image common scale. Projective GL(3)
equivariance only holds where the selected triple and full-rank validity stay
unchanged. Floating-point ties, rank refusal, nonpositive outputs and diagnostic
fallbacks are outside that claim. No pretrained components or camera metadata.

SHA-256 исходника: `e4fdabf2ef60fe818133184b7a76a2fee1e4b99c1900f830661a14212f6dad88`. Строк: **298**.

## Зависимости

```python
import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `EdgeDiffusionBlock` | nn.Module | [L175](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L175) |
| `ColorFramePosteriorNet` | nn.Module | [L195](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L195) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `vmf_log_normalizer` | FunctionDef | log(k / (4*pi*sinh(k))) on S², stable at zero and large k. | [L17](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L17) |
| `canonical_target` | FunctionDef | Normalize B^-1 GT with an FP64 solve. Singular frames are caller errors.  Use output['frame_valid'] before calling for external targets. The likelihood helper does this masking itself, and rejects nonfinite/nonpositive GT. | [L27](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L27) |
| `posterior_nll` | FunctionDef | Canonical directional NLL; mean over valid frames/GT, never silent zero.  Point positivity does not mask a valid-frame likelihood. reduction='none' returns N rows, NaN for excluded rows; 'mean' raises if all are excluded. This is a density in canonical solid angle, not camera-space solid angle. | [L54](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L54) |
| `camera_posterior_nll` | FunctionDef | Pushforward density NLL on camera S², using J=abs(det B)/&#124;&#124;Bq&#124;&#124;³.  Canonical NLLs from different frames are not comparable densities. This correction has no learned-parameter gradient because B and GT are inputs. Real reproduction errors remain the architecture-selection metric. | [L91](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L91) |
| `_vmf_quadrature` | FunctionDef | 32 equal-weight nodes: four midpoint CDF quantiles x eight azimuths.  For vMF about mu, z=cos(theta) has CDF inverse 1 + log(u+(1-u)*exp(-2*k))/k. A fixed least-aligned coordinate axis constructs the tangent basis. This low-order quadrature is approximate and can have orientation/validity-boundary artifacts, especially at low k. | [L119](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L119) |
| `transported_risk` | FunctionDef | Camera reproduction-angle spread proxy in degrees, preserving invalid mass.  Each of K*32 hypotheses is transported by B. Nonpositive or nonfinite mapped hypotheses receive cost90 degrees (a diagnostic convention, not a physical expectation). No renormalization of the remaining posterior mass is applied. | [L144](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L144) |
| `EdgeDiffusionBlock` | ClassDef | Directed eight-neighbor messages gated by embeddings and token differences. | [L175](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L175) |
| `ColorFramePosteriorNet` | ClassDef | См. реализацию | [L195](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed17/source_snapshot/scripts/cc_v3_model.py#L195) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L178–183</summary>

```python
def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Sequential(nn.Linear(2 * width + 11, width // 2), nn.SiLU(), nn.Linear(width // 2, 1), nn.Sigmoid())
        self.value = nn.Linear(width + 11, width)
        self.update = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
```

</details>

<details><summary>forward · L185–192</summary>

```python
def forward(self, h, features, source, target, degree):
        z = self.norm(h)
        delta = features[:, source] - features[:, target]
        gate = self.gate(torch.cat([z[:, source], z[:, target], delta], dim=-1))
        value = self.value(torch.cat([z[:, source] - z[:, target], delta], dim=-1))
        aggregate = torch.zeros_like(h).index_add(1, target, gate * value)
        aggregate = aggregate / degree[None, :, None]
        return h + self.update(torch.cat([z, aggregate], dim=-1))
```

</details>

<details><summary>__init__ · L196–220</summary>

```python
def __init__(self, mode="frame", width=192, layers=4, hypotheses=8):
        super().__init__()
        if mode not in {"direct", "diagonal", "frame"}:
            raise ValueError("mode must be direct, diagonal or frame")
        if width < 2 or layers < 1 or hypotheses < 1:
            raise ValueError("width>=2, layers>=1 and hypotheses>=1 required")
        self.mode = mode
        self.hypotheses = hypotheses
        self.condition_limit = 1e6
        self.register_buffer("triples", torch.tensor(list(itertools.combinations(range(16), 3))))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8), torch.linspace(-1, 1, 8), indexing="ij")
        self.register_buffer("coordinates", torch.stack([xx.flatten(), yy.flatten()], dim=-1))
        edges = [(a, b) for a in range(64) for b in range(64) if a != b and max(abs(a // 8 - b // 8), abs(a % 8 - b % 8)) == 1]
        self.register_buffer("source", torch.tensor([a for a, _ in edges]))
        self.register_buffer("target", torch.tensor([b for _, b in edges]))
        self.register_buffer("degree", torch.bincount(self.target, minlength=64).float())
        self.embedding = nn.Linear(11, width)
        self.blocks = nn.ModuleList([EdgeDiffusionBlock(width) for _ in range(layers)])
        self.context_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.SiLU())
        self.point_head = nn.Linear(64, 3)
        self.direction_head = nn.Linear(64, 3 * hypotheses)
        self.logit_head = nn.Linear(64, hypotheses)
        self.concentration_head = nn.Linear(64, hypotheses)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
```

</details>

<details><summary>forward · L259–298</summary>

```python
def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point() or min(x.shape[-2:]) < 8 or x.shape[0] == 0:
            raise ValueError("Expected nonempty floating NCHW RGB, H/W >= 8")
        canonical, fallback, frame, diagnostics = self._frame_and_pixels(x)
        dtype = self.embedding.weight.dtype
        # Population standard deviation; floor avoids unstable sqrt derivative at zero.
        means = F.adaptive_avg_pool2d(canonical, (8, 8))
        variance = F.adaptive_avg_pool2d(canonical.square(), (8, 8)) - means.square()
        std = variance.clamp_min(1e-24).sqrt()
        asinh = F.adaptive_avg_pool2d(canonical.asinh(), (8, 8))
        stats = torch.cat([means, std, asinh], dim=1).flatten(2).transpose(1, 2).to(dtype)
        coordinates = self.coordinates.to(dtype).expand(x.shape[0], -1, -1)
        features = torch.cat([stats, coordinates], dim=-1)
        h = self.embedding(features)
        for block in self.blocks:
            h = block(h, features, self.source, self.target, self.degree.to(dtype))
        context = self.context_head(h.mean(1))
        canonical_mean = canonical.mean((-2, -1)).to(dtype)
        anchor = F.normalize(canonical_mean, dim=-1)
        point = F.normalize(anchor + self.point_head(context), dim=-1)
        raw_pred = torch.einsum("nij,nj->ni", frame.to(dtype), point)
        point_valid = torch.isfinite(raw_pred).all(-1) & (raw_pred > 0).all(-1)
        valid = diagnostics["frame_valid"] & point_valid
        pred = torch.where(valid[:, None], F.normalize(raw_pred, dim=-1), fallback.to(dtype))
        direction_raw = anchor[:, None] + self.direction_head(context).reshape(-1, self.hypotheses, 3)
        directions = F.normalize(direction_raw, dim=-1)
        # Exact zero direction has a deterministic canonical axis; no GL claim is
        # attached to positivity fallback. Avoid an undefined sphere density.
        axis = torch.zeros_like(directions)
        axis[..., 0] = 1
        directions = torch.where(direction_raw.norm(dim=-1, keepdim=True) > 1e-12, directions, axis)
        logits = self.logit_head(context)
        concentration = F.softplus(self.concentration_head(context)) + 1e-4
        risk, mass = transported_risk(frame, pred, directions, logits, concentration, diagnostics["frame_valid"])
        return {
            "pred": pred, "raw_pred": raw_pred, "context": context, "valid": valid,
            "point_valid": point_valid, "frame": frame, "canonical_mean": canonical_mean,
            "directions": directions, "logits": logits, "concentration": concentration,
            "transport_risk": risk, "invalid_posterior_mass": mass, **diagnostics,
        }
```

</details>
