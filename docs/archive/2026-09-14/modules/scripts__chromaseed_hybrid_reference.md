# `scripts/chromaseed_hybrid_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent dense distances, weighted SVD geometry and QR/SVD readouts for H.

SHA-256 исходника: `93da95af8c3103b7a151cece44f5249cf89e992bdcbaf2eaac69ee0f5dc83e52`. Строк: **163**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_affine_reference import qr_ridge
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor
from chromaseed_projection_reference import direct_width, projector, spectrum
from chromaseed_projection_reference import predict as original_predict
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Geometry` | — | [L59](../../../../scripts/chromaseed_hybrid_reference.py#L59) |
| `Basis` | — | [L102](../../../../scripts/chromaseed_hybrid_reference.py#L102) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `signal` | FunctionDef | См. реализацию | [L16](../../../../scripts/chromaseed_hybrid_reference.py#L16) |
| `predict` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_hybrid_reference.py#L27) |
| `whitening` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_hybrid_reference.py#L53) |
| `Geometry` | ClassDef | См. реализацию | [L59](../../../../scripts/chromaseed_hybrid_reference.py#L59) |
| `Basis` | ClassDef | См. реализацию | [L102](../../../../scripts/chromaseed_hybrid_reference.py#L102) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L27–50</summary>

```python
def predict(model, x):
    if "hybrid_mode" not in model:
        return original_predict(model, x)
    raw = norm(model, x)
    p, m = model["latent_projection"].astype(np.float64), model["latent_mean"].astype(np.float64)
    q = ((raw - m) @ p).astype(np.float32).astype(np.float64)
    centers = ((model["centers"].astype(np.float64) - m) @ p).astype(np.float32)
    kr = direct_kernel(raw, model["centers"], float(model["width"]))
    kp = direct_kernel(q, centers, float(model["latent_width"]))
    base, branch = (
        kr @ model["coefficient"].astype(np.float64),
        kp @ model["latent_coefficient"].astype(np.float64),
    )
    s = signal(model, x)
    if s is not None:
        base += s[:, None] * (kr @ model["correction"].astype(np.float64))
        branch += s[:, None] * (kp @ model["latent_correction"].astype(np.float64))
    mix = float(model["mix"])
    if int(model["hybrid_mode"]) == 1:
        value = base * (1 - mix) + mix * branch
    else:
        cover = (np.sum(kr, axis=1) / kr.shape[1]) ** int(model["support_power"])
        value = base + mix * cover[:, None] * branch
    return value * model["y_std"] + model["y_mean"]
```

</details>

<details><summary>__init__ · L60–96</summary>

```python
def __init__(self, x, y, person, site, camera):
        self.x, self.y, self.w = x, y, balanced(person, site)
        self.prep = {}
        for prefix, a in (("x", x), ("y", y)):
            a = a.astype(np.float64)
            self.prep[prefix + "_mean"] = a.mean(0).astype(np.float32)
            self.prep[prefix + "_std"] = np.maximum(a.std(0), 1e-6).astype(np.float32)
        self.raw = norm(self.prep, x)
        self.spectrum = spectrum(self.raw, self.w)
        self.projection = projector(self.spectrum, 16, 0.5)
        p, m = (
            self.projection["projection"].astype(np.float64),
            self.projection["projection_mean"].astype(np.float64),
        )
        self.q = ((self.raw - m) @ p).astype(np.float32).astype(np.float64)
        self.rw, self.pw = (float(np.float32(direct_width(v))) for v in (self.raw, self.q))
        self.kr = direct_kernel(self.raw, self.raw, self.rw)
        self.kp = direct_kernel(self.q, self.q, self.pw)
        self.beta = (
            reference_gate(self.raw, person, site, camera) if len(np.unique(camera)) > 1 else None
        )
        self.signal = (
            None
            if self.beta is None
            else np.clip(
                np.sum(self.raw * self.beta[None, 1:].astype(np.float64), axis=1)
                + float(self.beta[0]),
                -1,
                1,
            )
        )
        self.target = (y.astype(np.float64) - self.prep["y_mean"]) / self.prep["y_std"]
        tensor = analytic_tensor(y)
        std = self.prep["y_std"].astype(np.float64)
        tensor = tensor * std[None, :, None] * std[None, None, :]
        tensor /= np.average(np.trace(tensor, axis1=1, axis2=2), weights=self.w) / 3
        self.metrics = dict(norm=np.eye(3), perceptual=np.average(tensor, axis=0, weights=self.w))
```

</details>

<details><summary>__init__ · L103–114</summary>

```python
def __init__(self, geometry, seed, rank=128):
        self.g = g = geometry
        self.ids, _, _ = select_landmarks(g.kr, g.w, "rpchol", rank, seed)
        self.rwhite, self.pwhite = (whitening(k[np.ix_(self.ids, self.ids)]) for k in (g.kr, g.kp))
        self.rd, self.pd = (
            g.design(g.kr[:, self.ids] @ self.rwhite),
            g.design(g.kp[:, self.ids] @ self.pwhite),
        )
        self.raw = {}
        for loss, metric in g.metrics.items():
            theta = qr_ridge(self.rd, g.target, g.w, metric, 0.1)
            self.raw[loss] = self.export(theta, False)
```

</details>

<details><summary>export · L116–135</summary>

```python
def export(self, theta, projected):
        g = self.g
        white, coords, width = (self.pwhite, g.q, g.pw) if projected else (self.rwhite, g.raw, g.rw)
        d = white.shape[1]
        model = dict(
            **g.prep,
            centers=coords[self.ids].astype(np.float32),
            width=np.array(width, np.float32),
            coefficient=(white @ theta[:d]).astype(np.float32),
        )
        if projected:
            model.update(g.projection)
        if g.beta is not None:
            model.update(
                correction=(white @ theta[d:]).astype(np.float32),
                gate_beta=g.beta,
                gate_mode=np.array(1, np.uint8),
                rho=np.array(1.0, np.float32),
            )
        return model
```

</details>
