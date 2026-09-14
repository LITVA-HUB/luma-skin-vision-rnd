import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from skin_dast_qualify import (  # noqa: E402
    audit_originals,
    infer_white,
    qualify_profile,
    white_diagnostics,
    xyz_to_lab,
)

WHITES = {'10deg': [94.811, 100., 107.304], '2deg': [95.047, 100., 108.883]}


def sample_profile():
    xyz = [23.1, 23.13, 17.24]
    return {
        'photographs': 2, 'subjects': 1, 'measurement_count': 1,
        'images': [dict(subject_id='001', image_id=f'I{i:02d}',
                        role='external_diagnostic', ground_truth_lab=None,
                        measurement_table_subject='001', camera=None, exposure=None)
                   for i in (1, 2)],
        'measurements': {'001': [dict(site_code='S', anatomical_site=None,
                                    lab_native=xyz_to_lab(xyz, WHITES['10deg']),
                                    xyz_native=xyz)]},
    }


def test_lab_black_white_and_inverse_use_the_low_light_branch():
    assert xyz_to_lab([0., 0., 0.], WHITES['10deg']) == pytest.approx([0., 0., 0.])
    assert xyz_to_lab(WHITES['10deg'], WHITES['10deg']) == pytest.approx([100., 0., 0.])
    for xyz in ([.2, .3, .4], [23.1, 23.13, 17.24]):
        lab = xyz_to_lab(xyz, WHITES['10deg'])
        assert infer_white(lab, xyz) == pytest.approx(WHITES['10deg'], abs=1e-10)
    with pytest.raises(ValueError, match='positive'):
        xyz_to_lab([1., 2., 3.], [0., 100., 100.])


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


def test_documented_convention_is_rejected_when_numbers_disagree():
    profile = sample_profile()
    row = profile['measurements']['001'][0]
    row['lab_native'] = xyz_to_lab(row['xyz_native'], WHITES['2deg'])
    with pytest.raises(ValueError, match='do not support'):
        qualify_profile(profile, WHITES)


@pytest.mark.parametrize('edit', [
    lambda p: p['images'][0].update(ground_truth_lab=[50., 5., 10.]),
    lambda p: p['images'][0].update(role='train'),
    lambda p: p['images'][0].update(subject_id='missing'),
    lambda p: p['images'][1].update(image_id='I01'),
    lambda p: p.update(photographs=3),
    lambda p: p['measurements']['001'][0].update(lab_native=[float('nan'), 0., 0.]),
])
def test_qualification_rejects_bad_grain_targets_and_nonfinite_values(edit):
    profile = sample_profile()
    edit(profile)
    with pytest.raises(ValueError):
        qualify_profile(profile, WHITES)


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
