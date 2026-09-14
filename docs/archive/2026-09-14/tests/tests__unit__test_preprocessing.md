# `tests/unit/test_preprocessing.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_preprocessing.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `0a1a32740286ccf2c859e203f72eb6d842a24ed22e253ad7e40faa6186feeadc`. Строк: **25**.

## Зависимости

```python
import numpy as np
import pytest
from PIL import Image
from luma_skin_vision.preprocessing import decode_image, quality_gate
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_decode_orientation_and_quality` | FunctionDef | См. реализацию | [L8](../../../../tests/unit/test_preprocessing.py#L8) |
| `test_bad_icc_rejected` | FunctionDef | См. реализацию | [L21](../../../../tests/unit/test_preprocessing.py#L21) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_decode_orientation_and_quality` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_decode_orientation_and_quality(tmp_path):
    arr = np.random.default_rng(1).integers(20, 230, (40, 80, 3), dtype=np.uint8)
    image = Image.fromarray(arr)
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    result = decode_image(path)
    assert result.shape == (80, 40, 3)
    assert quality_gate(np.zeros((64, 64, 3)), [0, 0, 64, 64])
    assert "face_too_small" in quality_gate(result, [0, 0, 20, 20])
```

</details>

### `test_bad_icc_rejected` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bad_icc_rejected(tmp_path):
    path = tmp_path / "bad.png"
    Image.new("RGB", (40, 40)).save(path, icc_profile=b"not-an-icc")
    with pytest.raises(ValueError, match="ICC"):
        decode_image(path)
```

</details>
