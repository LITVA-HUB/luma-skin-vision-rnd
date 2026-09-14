import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_only_original_training_roles_and_deterministic_order():
    from skin_uminho_train_expand_v2 import training_rows
    rows = [dict(name=f'{i}_reflectance.mat', role='train', size=i + 1) for i in range(19)]
    held = [dict(name='test.mat', role='test'), dict(name='val.mat', role='validation')]
    chosen = training_rows(list(reversed(rows)) + held)
    assert len(chosen) == 19
    assert all(r['role'] == 'train' for r in chosen)
    assert chosen == training_rows(rows + list(reversed(held)))
    with pytest.raises(ValueError, match='19'):
        training_rows(rows[:-1] + held)


@pytest.mark.parametrize('name', ['../escape.mat', 'C:\\escape.mat', 'folder/escape.mat', 'folder\\escape.mat'])
def test_source_paths_cannot_escape(name):
    from skin_uminho_train_expand_v2 import training_rows
    rows = [dict(name=f'{i}.mat', role='train', size=1) for i in range(19)]
    rows[0]['name'] = name
    with pytest.raises(ValueError, match='filename'):
        training_rows(rows)
