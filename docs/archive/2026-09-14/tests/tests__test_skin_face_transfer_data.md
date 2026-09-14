# `tests/test_skin_face_transfer_data.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_transfer_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `322f84ee28fae563852b33c49b3dbc6c672ee1412b9663d7c17d3f03b7f4a70d`. Строк: **96**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_gapped_global_ids_cannot_read_held_rows` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_face_transfer_data.py#L10) |
| `test_binary_source_and_label_domain_are_checked_at_read` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_face_transfer_data.py#L24) |
| `test_test_access_is_refused_before_opening_any_files` | FunctionDef | См. реализацию | [L40](../../../../tests/test_skin_face_transfer_data.py#L40) |
| `test_arms_share_anchor_and_sampling_stays_in_allowed_training_ids` | FunctionDef | См. реализацию | [L47](../../../../tests/test_skin_face_transfer_data.py#L47) |
| `test_each_sampler_stream_finishes_a_permutation_before_repeating` | FunctionDef | См. реализацию | [L69](../../../../tests/test_skin_face_transfer_data.py#L69) |
| `test_mixed_batch_keeps_source_row_correspondence` | FunctionDef | См. реализацию | [L81](../../../../tests/test_skin_face_transfer_data.py#L81) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_gapped_global_ids_cannot_read_held_rows` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gapped_global_ids_cannot_read_held_rows():
    from skin_face_transfer_data import ArraySplit
    rgb = np.broadcast_to(np.arange(9, dtype=np.uint8)[:, None, None, None], (9, 16, 16, 3)).copy()
    labels = np.zeros((9, 16, 16), dtype=np.uint8)
    labels[2, :, :4], labels[5, :, 4:8] = 1, 6
    split = ArraySplit('lapa', 'train', rgb, labels, np.array([2, 5, 8]), 'lapa11')
    x, y = split.get(np.array([5, 2, 5]))
    assert x[:, 0, 0, 0].tolist() == [5, 2, 5]
    assert y.dtype == np.uint8 and y.sum(axis=(1, 2)).tolist() == [64, 64, 64]
    for bad in [np.array([0]), np.array([-1]), np.array([9]), np.array([2.1])]:
        with pytest.raises(ValueError):
            split.get(bad)
```

</details>

### `test_binary_source_and_label_domain_are_checked_at_read` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_binary_source_and_label_domain_are_checked_at_read():
    from skin_face_transfer_data import ArraySplit
    rgb = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    labels = np.zeros((3, 16, 16), dtype=np.uint8)
    labels[1, 4:6, 7:10] = 1
    split = ArraySplit('celeba', 'validation', rgb, labels, np.array([1]), 'binary')
    assert int(split.get(np.array([1]))[1].sum()) == 6
    labels[1, 0, 0] = 6
    with pytest.raises(ValueError, match='label'):
        split.get(np.array([1]))
    with pytest.raises(ValueError):
        ArraySplit('celeba', 'train', rgb, labels, np.array([0, 0]), 'binary')
    with pytest.raises(ValueError):
        ArraySplit('celeba', 'train', rgb.astype(float), labels, np.array([0]), 'binary')
```

</details>

### `test_test_access_is_refused_before_opening_any_files` · L40

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_test_access_is_refused_before_opening_any_files(tmp_path):
    from skin_face_transfer_data import load_split
    with pytest.raises(ValueError, match='test'):
        load_split('celeba', 'test', data_root=tmp_path)
    assert list(tmp_path.iterdir()) == []
```

</details>

### `test_arms_share_anchor_and_sampling_stays_in_allowed_training_ids` · L47

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_arms_share_anchor_and_sampling_stays_in_allowed_training_ids():
    from skin_face_transfer_data import PairedSampler
    lapa, celeba = np.array([2, 5, 8, 11, 14]), np.array([101, 104, 109])
    left = PairedSampler(lapa, celeba, seed=17, batch_size=4)
    right = PairedSampler(lapa, celeba, seed=17, batch_size=4)
    first = []
    for _ in range(20):
        ls, li = left.next('lapa_only')
        rs, ri = right.next('lapa_celeba')
        assert ls.tolist() == [0, 0, 0, 0] and rs.tolist() == [0, 0, 1, 1]
        np.testing.assert_array_equal(li[:2], ri[:2])
        assert set(li) <= set(lapa) and set(ri[:2]) <= set(lapa) and set(ri[2:]) <= set(celeba)
        first.append((rs.copy(), ri.copy()))
    replay = PairedSampler(lapa, celeba, seed=17, batch_size=4)
    for expected in first[:7]:
        actual = replay.next('lapa_celeba')
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        replay.next('unregistered')
```

</details>

### `test_each_sampler_stream_finishes_a_permutation_before_repeating` · L69

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_each_sampler_stream_finishes_a_permutation_before_repeating():
    from skin_face_transfer_data import PairedSampler
    sampler = PairedSampler(np.arange(17), np.arange(100, 111), seed=29, batch_size=2)
    anchor, extra = [], []
    for _ in range(22):
        _, ids = sampler.next('lapa_celeba')
        anchor.append(int(ids[0]))
        extra.append(int(ids[1]))
    assert sorted(anchor[:17]) == list(range(17))
    assert sorted(extra[:11]) == sorted(extra[11:22]) == list(range(100, 111))
```

</details>

### `test_mixed_batch_keeps_source_row_correspondence` · L81

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_mixed_batch_keeps_source_row_correspondence():
    from skin_face_transfer_data import ArraySplit, assemble_batch
    rgb = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    for i in range(4):
        rgb[i, :, :, 0] = i + 10
    raw = np.zeros((4, 16, 16), dtype=np.uint8)
    raw[2] = 6
    a = ArraySplit('lapa', 'train', rgb, raw, np.array([0, 2]), 'lapa11')
    crgb = rgb + 40
    binary = (raw > 0).astype(np.uint8)
    b = ArraySplit('celeba', 'train', crgb, binary, np.array([1, 2]), 'binary')
    x, y = assemble_batch({0:a, 1:b}, np.array([1, 0, 1, 0]), np.array([1, 2, 2, 0]))
    assert x[:, 0, 0, 0].tolist() == [51, 12, 52, 10]
    assert y[:, 0, 0].tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        assemble_batch({0:a, 1:b}, np.array([1]), np.array([0]))
```

</details>
