import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def fixture_text():
    return '\n'.join(['notes'] * 6 + [',Subject 1,,,', 'Wavelength (nm),R1,R2,R3,Average',
                                     '250,.1,.2,.3,.2', '253,.2,.3,.4,.3'])


def test_nist_triplicates_stay_with_the_same_subject():
    from skin_nist_data import parse_nist
    arrays = parse_nist(fixture_text())
    np.testing.assert_array_equal(arrays['wavelength_nm'], [250, 253])
    assert arrays['repeats'].shape == (1, 3, 2)
    np.testing.assert_allclose(arrays['average'], [[.2, .3]])
    assert arrays['partition'].shape == (1,)


def test_nist_checks_average_and_grid_without_clipping():
    from skin_nist_data import parse_nist
    with pytest.raises(ValueError, match='average'):
        parse_nist(fixture_text().replace('.3,.2', '.3,.9'))
    with pytest.raises(ValueError, match='wavelength'):
        parse_nist(fixture_text().replace('253,', '249,'))


def test_nist_retains_author_repeat_labels_including_r4():
    from skin_nist_data import parse_nist
    arrays = parse_nist(fixture_text().replace('R1,R2,R3', 'R2,R3,R4'))
    np.testing.assert_array_equal(arrays['repeat_labels'], [['R2', 'R3', 'R4']])
    with pytest.raises(ValueError, match='layout'):
        parse_nist(fixture_text().replace('R1,R2,R3', 'R1,R2,R2'))
    with pytest.raises(ValueError, match='layout'):
        parse_nist(fixture_text().replace('R1,R2,R3', 'R1,R2,Other'))
