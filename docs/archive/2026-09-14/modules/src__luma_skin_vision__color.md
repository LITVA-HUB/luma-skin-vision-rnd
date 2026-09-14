# `src/luma_skin_vision/color.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/color.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

sRGB (IEC transfer) <-> CIELAB, D65 white / CIE 1931 2 degree observer.

CIEDE2000 follows Sharma et al., DOI 10.1002/col.20070, with kL=kC=kH=1.
These conversions describe encoded colors, not inferred physical reflectance.

SHA-256 исходника: `679005e5d5dd33bc891da8bc94f7954bf313cc092c007204c0c8f10f7a7fad46`. Строк: **101**.

## Зависимости

```python
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../src/luma_skin_vision/color.py#L9)

```python
WHITE = np.array([0.95047, 1.0, 1.08883])
```

[Строка 10](../../../../src/luma_skin_vision/color.py#L10)

```python
RGB_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_triples` | FunctionDef | См. реализацию | [L19](../../../../src/luma_skin_vision/color.py#L19) |
| `srgb_to_linear` | FunctionDef | См. реализацию | [L26](../../../../src/luma_skin_vision/color.py#L26) |
| `linear_to_srgb` | FunctionDef | См. реализацию | [L33](../../../../src/luma_skin_vision/color.py#L33) |
| `srgb_to_lab` | FunctionDef | См. реализацию | [L38](../../../../src/luma_skin_vision/color.py#L38) |
| `lab_to_srgb` | FunctionDef | См. реализацию | [L48](../../../../src/luma_skin_vision/color.py#L48) |
| `delta_e00` | FunctionDef | См. реализацию | [L58](../../../../src/luma_skin_vision/color.py#L58) |
