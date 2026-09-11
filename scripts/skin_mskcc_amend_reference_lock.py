"""One-time, explicit pre-test amendment; never refits or selects a model."""
import json
from skin_mskcc_data import ROOT,sha
from skin_mskcc_selective_core import OUT,binding


def main():
    archive=OUT/'reference_amendment_archive'
    original=archive/'precalibration_lock.json'
    digest='f4c286c74b79e0643f6a50766cae0d8298e04df109b7ee54a2462cc15739b10f'
    assert sha(original)==digest and sha(OUT/'precalibration_lock.json')==digest
    assert not (OUT/'final_lock.json').exists()
    lock=json.loads(original.read_bytes())
    loader=str((ROOT/'scripts/skin_mskcc_selective_data.py').relative_to(ROOT))
    assert sha(archive/'skin_mskcc_selective_data.py.txt')==lock['bindings'][loader]
    for name,expected in lock['bindings'].items():
        if name!=loader:assert sha(ROOT/name)==expected,name
    files=[ROOT/name for name in lock['bindings']]
    files += [original,archive/'skin_mskcc_selective_data.py.txt',
              ROOT/'docs/research/skin_mskcc_reference_amendment_v1.md',
              ROOT/'scripts/skin_mskcc_amend_reference_lock.py']
    lock['bindings']=binding(files)
    lock['original_precalibration_lock_sha256']=digest
    lock['no_calibration_or_test_endpoints_used']=False
    lock['calibration_reference_completeness_inspected']=True
    lock['calibration_predictions_or_errors_inspected']=False
    lock['test_endpoints_opened']=False
    lock['amendment']='docs/research/skin_mskcc_reference_amendment_v1.md'
    (OUT/'precalibration_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
    print('Amended pre-calibration SHA256:',sha(OUT/'precalibration_lock.json'))


if __name__=='__main__':main()
