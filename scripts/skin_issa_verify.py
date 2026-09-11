"""Independent scalar color checks and exact-spectrum overlap diagnostic."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from skin_issa_data import OUT, PRIVATE, ROOT, checked_manifest, sha, write_json
from skin_issa_material import load_cache, verify_lock
from skin_mskcc_audit import scalar_de


def main():
    verify_lock();manifest=checked_manifest()
    tr,common,train_r,train_valid=load_cache('train')
    va,_,val_r,val_valid=load_cache('validation')
    assert not set(tr['subject'])&set(va['subject'])
    expected={role:{r['A'] for r in manifest if r['role']==role} for role in ('train','validation')}
    assert set(tr['record'])==expected['train'] and set(va['record'])==expected['validation']
    for r in manifest:
        if int(r['B'])>=9:assert r['role']==f"reserved_origin_{r['B']}"
    pred=np.load(PRIVATE/'material_validation_predictions.npz',allow_pickle=False)
    result=json.loads((OUT/'material_results.json').read_text(encoding='utf-8-sig'))
    for path,h in result['bindings'].items():assert sha(ROOT/path)==h,path
    fixture=np.loadtxt(ROOT/'tests/fixtures/ciede2000_sharma.txt')
    assert max(abs(scalar_de(x[:3],x[3:6])-x[6]) for x in fixture)<5e-5
    mask=np.all(va['declared_support'][val_valid]==common[None,:],axis=1)
    targets=va['lab'][val_valid][mask]
    color_people=va['subject'][val_valid][mask];color_origins=va['origin'][val_valid][mask]
    comparisons=0;gap=0.
    for record in result['records']:
        name=f"{record['method']}_{record['width']}"
        e=np.array([scalar_de(p,q) for p,q in zip(pred[name+'_lab'],targets,strict=True)])
        comparisons+=len(e);gap=max(gap,float(np.max(np.abs(e-pred[name+'_error']))))
        for key,value in [('mean',np.mean(e)),('median',np.median(e)),('p95',np.quantile(e,.95)),('max',np.max(e))]:
            gap=max(gap,abs(float(value)-record['common_support_native_delta_e00'][key]))
        person_score=np.mean([e[color_people==p].mean() for p in set(color_people)])
        gap=max(gap,abs(float(person_score)-record['mean_subject_delta_e00']))
        by_origin=[float(e[color_origins==o].mean()) for o in set(color_origins)]
        gap=max(gap,abs(max(by_origin)-record['worst_origin_mean_delta_e00']))
        r=val_r[val_valid];p=pred[name+'_spectra']
        rmse=np.sqrt(np.mean((p-r)**2,axis=1));rel=np.linalg.norm(p-r,axis=1)/np.linalg.norm(r,axis=1)
        for arr,key in [(rmse,'spectral_rmse_reflectance_fraction'),(rel,'spectral_relative_l2')]:
            gap=max(gap,abs(float(arr.mean())-record[key]['mean']))
    assert gap<1e-10
    # Exact common-band fingerprints diagnose copied records across labels.
    # This does not establish near-duplicate freedom or independent identities.
    def fingerprint(row):return hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest()
    train_prints=defaultdict(set)
    for r,p in zip(train_r,tr['subject'],strict=True):train_prints[fingerprint(r)].add(str(p))
    validation_matches=sum(fingerprint(r) in train_prints for r in val_r)
    within_train_shared=sum(len(v)>1 for v in train_prints.values())
    check={'status':'PASS','independent_scalar_delta_e00_cases':comparisons,'maximum_metric_gap':gap,
        'all_16_records_checked':len(result['records'])==16,'split_record_membership_checked':True,
        'validation_spectra_exactly_matching_training':validation_matches,
        'train_fingerprints_shared_by_multiple_subject_labels':within_train_shared,
        'identity_limitation':'Exact overlap diagnostic only; no independent person-identity verification.',
        'reserved_endpoint_values_read':False,
        'bindings':{p:sha(ROOT/p) for p in ['scripts/skin_issa_verify.py','scripts/skin_mskcc_audit.py',
            'tests/fixtures/ciede2000_sharma.txt','docs/benchmarks/skin_issa_v1/material_results.json']}}
    write_json(OUT/'independent_audit.json',check);print(json.dumps(check))


if __name__=='__main__':main()
