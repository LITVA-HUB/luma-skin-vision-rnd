# `scripts/chromaseed_projection.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fit-only covariance projections with compact analytic joint color readouts.

SHA-256 исходника: `fa878f809f194c992f3e9253ca0cd6f32c9e4b0ca1d81df733fa6f735160f6c0`. Строк: **373**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_fast_kernel import streaming_landmarks
from chromaseed_gated import fit_gate, package_model
from chromaseed_gated import predict as g_predict
from chromaseed_kernel import coordinates, fit_normalizer, gaussian_kernel
from chromaseed_perceptual import make_basis, prepare
from scipy.spatial.distance import pdist
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_projection.py#L17)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 19](../../../../scripts/chromaseed_projection.py#L19)

```python
REPRESENTATIONS = (dict(name="raw", dimension=36, shrinkage=None),) + tuple(
    dict(name=f"d{d}_t{label}", dimension=d, shrinkage=t)
    for d in (8, 16, 36)
    for label, t in (("0", 0.0), ("01", 0.1), ("05", 0.5), ("1", 1.0))
)
```

[Строка 24](../../../../scripts/chromaseed_projection.py#L24)

```python
POLICIES = ("quality", "compact")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `VariableColumns` | — | [L122](../../../../scripts/chromaseed_projection.py#L122) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `covariance` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_projection.py#L27) |
| `projection` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_projection.py#L58) |
| `project` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_projection.py#L87) |
| `exact_width` | FunctionDef | См. реализацию | [L98](../../../../scripts/chromaseed_projection.py#L98) |
| `VariableColumns` | ClassDef | См. реализацию | [L122](../../../../scripts/chromaseed_projection.py#L122) |
| `projected_basis` | FunctionDef | См. реализацию | [L148](../../../../scripts/chromaseed_projection.py#L148) |
| `model_id` | FunctionDef | См. реализацию | [L164](../../../../scripts/chromaseed_projection.py#L164) |
| `choose` | FunctionDef | См. реализацию | [L168](../../../../scripts/chromaseed_projection.py#L168) |
| `export` | FunctionDef | См. реализацию | [L189](../../../../scripts/chromaseed_projection.py#L189) |
| `predict` | FunctionDef | См. реализацию | [L205](../../../../scripts/chromaseed_projection.py#L205) |
| `representation_state` | FunctionDef | См. реализацию | [L227](../../../../scripts/chromaseed_projection.py#L227) |
| `basis_for` | FunctionDef | См. реализацию | [L243](../../../../scripts/chromaseed_projection.py#L243) |
| `fit_bank` | FunctionDef | См. реализацию | [L254](../../../../scripts/chromaseed_projection.py#L254) |
| `fit_single` | FunctionDef | См. реализацию | [L327](../../../../scripts/chromaseed_projection.py#L327) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L123–136</summary>

```python
def __init__(self, x, width):
        self.x = np.asarray(x, np.float64)
        if (
            self.x.ndim != 2
            or not len(self.x)
            or not 1 <= self.x.shape[1] <= 36
            or not np.isfinite(self.x).all()
            or not np.isfinite(width)
            or width <= 0
        ):
            raise ValueError("finite feature matrix and positive width required")
        self.norms = np.einsum("ij,ij->i", self.x, self.x)
        self.denominator = 2 * self.x.shape[1] * float(width) ** 2
        self.entries_evaluated = 0
```

</details>

<details><summary>export · L189–202</summary>

```python
def export(state, projection_payload, basis, theta, beta, joint):
    if not projection_payload:
        return export_model(state, basis, theta, beta, joint)
    dim = basis["whitening"].shape[1]
    base = dict(
        **state["prep"],
        **projection_payload,
        centers=basis["centers"],
        coefficient=(basis["whitening"] @ theta[:dim]).astype(np.float32),
        width=np.array(basis["width"], np.float32),
    )
    if joint and beta is not None:
        return package_model(base, basis["whitening"] @ theta[dim:], beta, "soft", 1.0)
    return base
```

</details>

<details><summary>predict · L205–224</summary>

```python
def predict(model, x):
    if "constant_lab" in model:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("finite color36 matrix required")
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    if "projection" not in model:
        return g_predict(model, x)
    q = project(model, x)
    k = gaussian_kernel(q, model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        raw = coordinates(model, x)
        s = raw @ model["gate_beta"][1:].astype(np.float64) + float(model["gate_beta"][0])
        value += (
            float(model["rho"])
            * np.clip(s, -1, 1)[:, None]
            * (k @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]
```

</details>
