"""Sparse public-phone acquisition mechanics; no benchmark claims."""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/cc_phone_data.py'
    assert path.exists(), 'Phone sparse acquisition not implemented'
    spec = importlib.util.spec_from_file_location('cc_phone_data', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_split_archive_ranges_cover_cross_part_record_without_extra_bytes():
    m = module()
    assert m.split_ranges([10, 20, 5], 8, 25) == [(0, 8, 9), (1, 0, 19), (2, 0, 2)]
    assert m.split_ranges([10, 20], 10, 20) == [(1, 0, 19)]
    with pytest.raises(ValueError):
        m.split_ranges([10, 20], 29, 2)
    with pytest.raises(ValueError):
        m.split_ranges([10, 20], -1, 1)


def test_scene_completeness_requires_two_phone_targets_and_keeps_privacy_masks():
    m = module()
    scene = 'outdoor/1'
    names = [f'beyond-unzip/beyondRGB/{scene}/{name}' for name in m.REQUIRED]
    rows = {name: {'path': name, 'compressed_bytes': 1} for name in names}
    mask = f'beyond-unzip/beyondRGB/{scene}/NT/samsung_blurred_areas_detection.json'
    rows[mask] = {'path': mask, 'compressed_bytes': 1}
    assert any(r['path'] == mask for r in m.scene_members(rows, scene))
    del rows[f'beyond-unzip/beyondRGB/{scene}/WT/oppo.h5']
    assert m.scene_members(rows, scene) is None
