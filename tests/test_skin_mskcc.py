import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/skin_mskcc_data.py'
spec = importlib.util.spec_from_file_location('skin_mskcc_data_test', SCRIPT)
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def test_patient_partition_is_order_independent_and_exact():
    devices = {**{f's{i}': 'SLR' for i in range(18)}, **{f'p{i}': 'ipod' for i in range(28)}}
    roles = data.patient_roles(devices)
    assert roles == data.patient_roles(dict(reversed(list(devices.items()))))
    assert {r: list(roles.values()).count(r) for r in set(roles.values())} == {
        'train': 24, 'validation': 6, 'calibration': 6, 'test': 10}
    assert sum(r == 'test' and devices[p] == 'SLR' for p, r in roles.items()) == 4


@pytest.mark.parametrize('role', ['test', 'calibration', 'all'])
def test_source_loader_denies_held_out_endpoints_before_io(role, monkeypatch):
    def forbidden():
        pytest.fail('Endpoint guard performed I/O first')
    monkeypatch.setattr(data, 'manifest', forbidden)
    with pytest.raises(ValueError, match='cannot open'):
        data.load_source(role)
