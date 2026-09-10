import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
import cc_v7_risk


def test_streaming_reader_preserves_risk_then_calibration_alignment(monkeypatch):
    def fake_read(path,key,indices,total):
        assert indices.tolist()==[1,3,4,7]
        return indices[:,None]*10
    monkeypatch.setattr(cc_v7_risk,"read_npz_rows",fake_read)
    actual=cc_v7_risk.read_in_role_order(Path("unused"),"gt",np.array([1,7,3,4]))
    np.testing.assert_array_equal(actual[:,0],[10,70,30,40])


def test_risk_fitting_refuses_test_roles_and_overlapping_calibration(tmp_path):
    values={"valid":np.ones(2,dtype=bool)}
    rows=[{"subset":"risk","group":"a"},{"subset":"test","group":"b"}]
    with pytest.raises(ValueError,match="wrong role"):
        cc_v7_risk.fit_heads(values,None,rows,1,tmp_path,"unused")
    rows[1]={"subset":"cal","group":"a"}
    with pytest.raises(ValueError,match="overlap"):
        cc_v7_risk.fit_heads(values,None,rows,1,tmp_path,"unused")
    assert not list(tmp_path.iterdir())
