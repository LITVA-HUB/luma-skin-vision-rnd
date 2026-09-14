# `tests/test_skin_dast_qualify.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_dast_qualify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `b589e5cb3c5d4fbc2c0b5e965db1a4ab94c8b53aa610d03af71c54d56819d9bb`. Строк: **118**.

## Зависимости

```python
import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest
from skin_dast_qualify import (  # noqa: E402
    audit_originals,
    infer_white,
    qualify_profile,
    white_diagnostics,
    xyz_to_lab,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../tests/test_skin_dast_qualify.py#L19)

```python
WHITES = {'10deg': [94.811, 100., 107.304], '2deg': [95.047, 100., 108.883]}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sample_profile` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_dast_qualify.py#L22) |
| `test_lab_black_white_and_inverse_use_the_low_light_branch` | FunctionDef | См. реализацию | [L36](../../../../tests/test_skin_dast_qualify.py#L36) |
| `test_white_comparison_does_not_always_prefer_ten_degrees` | FunctionDef | См. реализацию | [L46](../../../../tests/test_skin_dast_qualify.py#L46) |
| `test_convention_qualification_does_not_invent_image_or_site_targets` | FunctionDef | См. реализацию | [L58](../../../../tests/test_skin_dast_qualify.py#L58) |
| `test_documented_convention_is_rejected_when_numbers_disagree` | FunctionDef | См. реализацию | [L76](../../../../tests/test_skin_dast_qualify.py#L76) |
| `test_qualification_rejects_bad_grain_targets_and_nonfinite_values` | FunctionDef | См. реализацию | [L92](../../../../tests/test_skin_dast_qualify.py#L92) |
| `test_original_archive_replay_is_read_only_and_detects_modified_metadata` | FunctionDef | См. реализацию | [L99](../../../../tests/test_skin_dast_qualify.py#L99) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_lab_black_white_and_inverse_use_the_low_light_branch` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lab_black_white_and_inverse_use_the_low_light_branch():
    assert xyz_to_lab([0., 0., 0.], WHITES['10deg']) == pytest.approx([0., 0., 0.])
    assert xyz_to_lab(WHITES['10deg'], WHITES['10deg']) == pytest.approx([100., 0., 0.])
    for xyz in ([.2, .3, .4], [23.1, 23.13, 17.24]):
        lab = xyz_to_lab(xyz, WHITES['10deg'])
        assert infer_white(lab, xyz) == pytest.approx(WHITES['10deg'], abs=1e-10)
    with pytest.raises(ValueError, match='positive'):
        xyz_to_lab([1., 2., 3.], [0., 100., 100.])
```

</details>

### `test_white_comparison_does_not_always_prefer_ten_degrees` · L46

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_white_comparison_does_not_always_prefer_ten_degrees():
    profile = sample_profile()
    measurements = profile['measurements']['001']
    result = white_diagnostics(measurements, WHITES)
    assert result['candidates']['10deg']['mean_lab_residual'] < 1e-10
    assert result['candidates']['2deg']['mean_lab_residual'] > .4
    measurements[0]['lab_native'] = xyz_to_lab(measurements[0]['xyz_native'], WHITES['2deg'])
    result = white_diagnostics(measurements, WHITES)
    assert result['candidates']['2deg']['mean_lab_residual'] < 1e-10
    assert result['candidates']['10deg']['mean_lab_residual'] > .4
```

</details>

### `test_convention_qualification_does_not_invent_image_or_site_targets` · L58

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_convention_qualification_does_not_invent_image_or_site_targets():
    profile = sample_profile()
    profile['measurements']['001'][0]['site_code'] = 'Measure 3'
    before = copy.deepcopy(profile)
    result = qualify_profile(profile, WHITES)
    assert profile == before
    assert result['counts'] == dict(photographs=2, author_subject_groups=1,
                                   native_measurements=1, qualified_conventions=1,
                                   confirmed_site_mappings=0, eligible_camera_lab_pairs=0)
    assert result['measurements'][0]['observer_degrees'] == 10
    assert result['measurements'][0]['observer'] == 'CIE 1964 10-degree standard observer'
    assert result['measurements'][0]['anatomical_site'] is None
    assert result['measurements'][0]['paper_site_code'] is None
    assert result['measurements'][0]['site_code'] == 'Measure 3'
    assert all(row['ground_truth_lab'] is None for row in result['images'])
    assert all(row['role'] == 'external_diagnostic' for row in result['images'])
```

</details>

### `test_documented_convention_is_rejected_when_numbers_disagree` · L76

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_documented_convention_is_rejected_when_numbers_disagree():
    profile = sample_profile()
    row = profile['measurements']['001'][0]
    row['lab_native'] = xyz_to_lab(row['xyz_native'], WHITES['2deg'])
    with pytest.raises(ValueError, match='do not support'):
        qualify_profile(profile, WHITES)
```

</details>

### `test_qualification_rejects_bad_grain_targets_and_nonfinite_values` · L92

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('edit', [lambda p: p['images'][0].update(ground_truth_lab=[50.0, 5.0, 10.0]), lambda p: p['images'][0].update(role='train'), lambda p: p['images'][0].update(subject_id='missing'), lambda p: p['images'][1].update(image_id='I01'), lambda p: p.update(photographs=3), lambda p: p['measurements']['001'][0].update(lab_native=[float('nan'), 0.0, 0.0])])
def test_qualification_rejects_bad_grain_targets_and_nonfinite_values(edit):
    profile = sample_profile()
    edit(profile)
    with pytest.raises(ValueError):
        qualify_profile(profile, WHITES)
```

</details>

### `test_original_archive_replay_is_read_only_and_detects_modified_metadata` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_original_archive_replay_is_read_only_and_detects_modified_metadata(tmp_path):
    import zipfile

    root = tmp_path / 'dast'
    root.mkdir()
    relative = Path('001/Colorimeter/001.cmf')
    original = root / 'originals' / relative
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original instrument bytes')
    with zipfile.ZipFile(root / 'example-data.zip', 'w') as archive:
        archive.writestr('example-data/' + relative.as_posix(), original.read_bytes())
    profile = dict(archive_sha256=hashlib.sha256((root / 'example-data.zip').read_bytes()).hexdigest())
    (root / 'profile.json').write_text(json.dumps(profile), encoding='utf-8')
    before = original.read_bytes()
    manifest = audit_originals(root, profile)
    assert original.read_bytes() == before
    assert any(Path(row['path']) == original for row in manifest)
    original.write_bytes(b'changed instrument bytes')
    with pytest.raises(ValueError, match='original differs'):
        audit_originals(root, profile)
```

</details>
