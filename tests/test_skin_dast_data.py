import io
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def cmf(label='S'):
    return 'Untrusted notes\n\nLabel   \tDate\tCIE-L*\tCIE-a*\tCIE-b*\tX\tY\tZ\n' + label + '\tdate\t55\t10\t14\t24\t23\t17\n'


def archive(path, extra=None):
    photo = io.BytesIO()
    Image.new('RGB', (10, 12), (200, 150, 100)).save(photo, format='JPEG')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('example-data/002/Pictures/cropped/002-I13-0.jpg', photo.getvalue())
        z.writestr('example-data/002/Colorimeter/002.cmf', cmf('Measure 3'))
        z.writestr('example-data/example_overview.csv', ';'.join([
            '2', 'I13', 'VIS/RGB', 'subjects/002/Pictures/cropped/002-I13.jpg',
            'original', 'ofiq', 'ofiq', '4', '', '', '', 'true', 'true', 'true', 'true', 'true', 'meta']))
        if extra:
            z.writestr(extra, b'unsafe')
    return photo.getvalue()


def test_dast_native_measurements_preserve_unknown_sites():
    from skin_dast_data import parse_cmf
    known = parse_cmf(cmf())[0]
    assert known['lab_native'] == [55., 10., 14.]
    assert known['site_code'] == 'S'
    assert known['illuminant'] is None
    assert parse_cmf(cmf('Measure 3'))[0]['anatomical_site'] is None
    with pytest.raises(ValueError, match='finite'):
        parse_cmf(cmf().replace('\t55\t', '\tnan\t'))
    with pytest.raises(ValueError, match='header'):
        parse_cmf('not a measurement table')


def test_dast_filename_alias_is_unique_and_photos_remain_byte_exact(tmp_path):
    from skin_dast_data import ingest_dast
    raw = archive(tmp_path/'original.zip')
    result = ingest_dast(tmp_path/'original.zip', tmp_path/'out')
    assert result['subjects'] == 1 and result['photographs'] == 1
    row = result['images'][0]
    assert row['subject_id'] == '002'
    assert row['camera'] is None and row['exposure'] is None
    assert Path(row['path']).read_bytes() == raw
    assert row['role'] == 'external_diagnostic'
    assert row['ground_truth_lab'] is None
    assert row['filename_alias_used'] is True


@pytest.mark.parametrize('bad', ['../escape.jpg', '/abs.jpg', 'example-data/../x.jpg', 'C:/x.jpg'])
def test_dast_archive_rejects_unsafe_paths_before_writing(tmp_path, bad):
    from skin_dast_data import ingest_dast
    archive(tmp_path/'original.zip', bad)
    with pytest.raises(ValueError, match='path'):
        ingest_dast(tmp_path/'original.zip', tmp_path/'out')
    assert not (tmp_path/'out').exists()
