# `tests/test_skin_data_growth.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_data_growth.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Meaningful data-boundary tests; no private photographs or network needed.

SHA-256 исходника: `0babd086fdda6c9bf89f8e5f45adaf96dcc02e7f9c793e894392ace155f85bf7`. Строк: **83**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `module` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_data_growth.py#L12) |
| `test_validate_bgr_labels_and_shape` | FunctionDef | См. реализацию | [L17](../../../../tests/test_skin_data_growth.py#L17) |
| `test_color_groups_do_not_cross_splits_or_depend_on_order` | FunctionDef | См. реализацию | [L29](../../../../tests/test_skin_data_growth.py#L29) |
| `test_profile_counts_conflicting_colors` | FunctionDef | См. реализацию | [L42](../../../../tests/test_skin_data_growth.py#L42) |
| `test_photo_ingestion_preserves_bytes_and_missing_ground_truth` | FunctionDef | См. реализацию | [L51](../../../../tests/test_skin_data_growth.py#L51) |
| `test_corrupt_photo_is_rejected_before_copy` | FunctionDef | См. реализацию | [L68](../../../../tests/test_skin_data_growth.py#L68) |
| `test_ingestion_detects_changed_existing_destination` | FunctionDef | См. реализацию | [L76](../../../../tests/test_skin_data_growth.py#L76) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_validate_bgr_labels_and_shape` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_validate_bgr_labels_and_shape():
    m = module()
    a = np.array([[0, 128, 255, 1], [255, 20, 0, 2]], dtype=float)
    bgr, labels = m.validate_uci(a)
    np.testing.assert_array_equal(bgr, a[:, :3])
    np.testing.assert_array_equal(labels, [1, 0])
    for invalid in [a[:, :3], [[-1, 0, 0, 1]], [[0, 0, 256, 2]],
                    [[0.1, 0, 0, 1]], [[0, 0, 0, 3]], [[np.nan, 0, 0, 1]]]:
        with pytest.raises(ValueError):
            m.validate_uci(np.asarray(invalid))
```

</details>

### `test_color_groups_do_not_cross_splits_or_depend_on_order` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_color_groups_do_not_cross_splits_or_depend_on_order():
    m = module()
    colors = np.random.default_rng(14).integers(0, 256, (500, 3), dtype=np.uint8)
    colors = np.concatenate([colors, colors[:50], colors[:50]])
    assignment = m.color_partitions(colors)
    np.testing.assert_array_equal(assignment[:50], assignment[-50:])
    order = np.random.default_rng(21).permutation(len(colors))
    np.testing.assert_array_equal(m.color_partitions(colors[order]), assignment[order])
    assert set(assignment) == {0, 1, 2}
    for color in np.unique(colors, axis=0):
        assert len(set(assignment[np.all(colors == color, axis=1)])) == 1
```

</details>

### `test_profile_counts_conflicting_colors` · L42

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_profile_counts_conflicting_colors():
    m = module()
    bgr = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4], [8, 9, 10]], np.uint8)
    p = m.profile_uci(bgr, np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]))
    assert p['rows'] == 4 and p['unique_colors'] == 2
    assert p['duplicate_color_rows'] == 2 and p['conflicting_colors'] == 1
    assert p['skin_rows'] == 2 and p['non_skin_rows'] == 2
```

</details>

### `test_photo_ingestion_preserves_bytes_and_missing_ground_truth` · L51

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_photo_ingestion_preserves_bytes_and_missing_ground_truth(tmp_path):
    m = module()
    original = tmp_path / 'provided.jpg'
    Image.new('RGB', (17, 19), (180, 130, 100)).save(original)
    before = original.read_bytes()
    output = tmp_path / 'private'
    manifest = m.ingest_photos([original, original], output)
    assert manifest['supplied_files'] == 2 and len(manifest['records']) == 1
    r = manifest['records'][0]
    assert Path(r['path']).read_bytes() == before == original.read_bytes()
    assert r['width'] == 17 and r['height'] == 19
    assert r['ground_truth_lab'] is None and not r['eligible_supervised_color']
    assert r['role'] == 'unlabeled_diagnostic'
    assert r['leakage_group'] == manifest['batch_group']
    assert m.ingest_photos([original, original], output)['records'] == manifest['records']
```

</details>

### `test_corrupt_photo_is_rejected_before_copy` · L68

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_corrupt_photo_is_rejected_before_copy(tmp_path):
    m = module()
    bad = tmp_path / 'bad.jpg'
    bad.write_bytes(b'not an image')
    with pytest.raises(ValueError, match='decode'):
        m.ingest_photos([bad], tmp_path / 'private')
```

</details>

### `test_ingestion_detects_changed_existing_destination` · L76

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_ingestion_detects_changed_existing_destination(tmp_path):
    m = module()
    original = tmp_path / 'ok.jpg'
    Image.new('RGB', (12, 12)).save(original)
    manifest = m.ingest_photos([original], tmp_path / 'private')
    Path(manifest['records'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        m.ingest_photos([original], tmp_path / 'private')
```

</details>
