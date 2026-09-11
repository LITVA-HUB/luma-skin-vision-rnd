import sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import skin_mskcc_selective_core as core
import skin_mskcc_selective_data as reserved


def test_grouped_oof_assignment_covers_each_person_once():
    people=[f'p{i}' for i in range(24)]
    folds=core.patient_folds(people)
    assert folds==core.patient_folds(people[::-1])
    assert sorted(list(folds.values()).count(i) for i in range(6))==[4]*6


def test_sealed_data_checks_lock_before_any_endpoint_io(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('wrong lock')
    def io():pytest.fail('Endpoint I/O before lock verification')
    monkeypatch.setattr(reserved,'verify_lock',fail);monkeypatch.setattr(reserved,'manifest',io)
    with pytest.raises(ValueError,match='wrong lock'):reserved.sealed('test','missing','bad')


def test_lock_checks_stage_and_bound_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'ROOT',tmp_path)
    value=tmp_path/'value';value.write_bytes(b'original')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'stage':'final','bindings':{'value':core.sha(value)}}))
    digest=core.sha(lock);core.verify_lock(lock,digest,'final')
    with pytest.raises(ValueError,match='stage'):core.verify_lock(lock,digest,'precalibration')
    value.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):core.verify_lock(lock,digest,'final')


def test_risk_feature_contract_keeps_common_inputs_identical():
    data={'color':np.ones((5,36))};p=np.arange(45).reshape(3,5,3);w=np.ones((3,5,8))
    d=core.designs(data,p,w,np.arange(5))
    for r in d.values():
        np.testing.assert_array_equal(r['C_plus'],r['Proposed'][:,:40])
    assert np.all(d['single17']['Proposed'][:,-1]==0)
    assert np.all(d['ensemble']['Proposed'][:,-1]>0)


def test_instrument_reference_retains_actual_single_measurement_without_imputation():
    row={f'{c}_{i}':str(v) if i==1 else '' for c,v in zip('lab',[50,12,20]) for i in [1,2,3]}
    row.update(dict(zip(['average_l','average_a','average_b'],['50','12','20'])))
    target,rep=reserved.instrument_reference(row)
    np.testing.assert_array_equal(target,[50,12,20])
    assert np.isnan(rep[1:]).all()
    row['l_2']='51'
    with pytest.raises(ValueError,match='Partial'):reserved.instrument_reference(row)


def test_instrument_reference_rejects_missing_and_inconsistent_ground_truth():
    row={f'{c}_{i}':'' for c in 'lab' for i in [1,2,3]}
    row.update({f'average_{c}':'0' for c in 'lab'})
    with pytest.raises(ValueError,match='No complete'):reserved.instrument_reference(row)
    for c in 'lab':row[f'{c}_1']='1'
    with pytest.raises(ValueError,match='inconsistent'):reserved.instrument_reference(row)
