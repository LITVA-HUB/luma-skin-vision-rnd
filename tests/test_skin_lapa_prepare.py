import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_lapa_archive_paths_are_bounded(tmp_path):
    from skin_lapa_prepare import member_target
    assert member_target(tmp_path, 'LaPa/train/images/123_0.jpg').is_relative_to(tmp_path)
    for name in ['../outside', 'LaPa/train/images/../../bad.jpg', '/abs.jpg', 'LaPa/train/images/C:x.jpg']:
        with pytest.raises(ValueError, match='path'):
            member_target(tmp_path, name)


def records(split, stem, digest):
    return [dict(split=split, kind=kind, stem=stem, sha256=digest if kind == 'images' else kind+stem,
                 path=f'{split}/{kind}/{stem}', bytes=1)
            for kind in ['images', 'labels', 'landmarks']]


def test_lapa_preserves_official_test_and_excludes_source_group_or_byte_duplicates():
    from skin_lapa_prepare import select_rows
    source = (records('test', '123_0', 'a') + records('train', '123_1', 'b') +
              records('val', '500_0', 'a') + records('val', '700_0', 'c') +
              records('train', '800_0', 'c') + records('train', '900_0', 'd'))
    kept, dropped = select_rows(source)
    assert [(r['split'], r['stem']) for r in kept] == [('test','123_0'),('train','900_0'),('val','700_0')]
    assert len(dropped) == 3


def test_lapa_missing_or_duplicate_pairs_are_rejected():
    from skin_lapa_prepare import select_rows
    source = records('train', '1_0', 'a')
    with pytest.raises(ValueError, match='triplet'):
        select_rows(source[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        select_rows(source + source[:1])
