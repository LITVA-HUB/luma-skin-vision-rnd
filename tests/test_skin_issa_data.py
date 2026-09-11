"""Boundary checks for subject grouping and restricted endpoint extraction."""
import xml.etree.ElementTree as ET

import pytest

from scripts.skin_issa_data import assign_roles, selected_cells, column_name


def test_roles_do_not_depend_on_row_order_or_record_multiplicity():
    rows = [{'B': str(o), 'C': str(o*100+s)} for o in (1, 2, 9, 10, 11) for s in range(20)]
    roles = assign_roles(rows)
    assert roles == assign_roles(list(reversed(rows)) + rows[:10])
    assert sum(v == 'train' for v in roles.values()) == 28
    assert sum(v == 'validation' for v in roles.values()) == 6
    assert sum(v == 'known_source_test' for v in roles.values()) == 6
    assert all(roles[(str(o), str(o*100+s))] == f'reserved_origin_{o}' for o in (9,10,11) for s in range(20))


def test_endpoint_columns_are_not_decoded_by_metadata_reader():
    row = ET.fromstring('<row xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><c r="A13"><v>1</v></c><c r="B13"><v>2</v></c><c r="R13" t="s"><v>999999</v></c></row>')
    assert selected_cells(row, [], {'A','B'}) == {'A': '1', 'B': '2'}
    with pytest.raises(IndexError):
        selected_cells(row, [], {'R'})


def test_subject_code_cannot_cross_origins_and_columns_are_bijective():
    with pytest.raises(ValueError, match='multiple origins'):
        assign_roles([{'B':'1','C':'9'}, {'B':'2','C':'9'}])
    assert [column_name(i) for i in (1, 26, 27, 43, 48, 56, 58, 65)] == ['A','Z','AA','AQ','AV','BD','BF','BM']
