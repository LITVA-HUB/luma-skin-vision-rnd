"""Read-only OOF or independent-test audit; cannot fit or alter any model."""
import argparse
import json
import joblib
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,RUN,patient_folds,plain_predict,verify_lock,deployment_designs
from skin_mskcc_selective_data import sealed
from skin_mskcc_audit import scalar_de


def oof_audit():
    torch.set_num_threads(4);data=load('train');oof=np.load(RUN/'oof.npz');folds=patient_folds(data['patient'])
    assert np.array_equal(oof['fold'],[folds[p] for p in data['patient']])
    receipt=json.loads((OUT/'oof.json').read_bytes());maximum=0.;cases=0
    for row in receipt['fits']:
        f,s=row['fold'],row['seed'];fit=np.flatnonzero(oof['fold']!=f);held=np.flatnonzero(oof['fold']==f)
        assert set(data['patient'][fit]).isdisjoint(data['patient'][held])
        path=RUN/f'oof/fold{f}_seed{s}.pt';assert sha(path)==row['checkpoint_sha256']
        state=torch.load(path,weights_only=True,map_location='cpu')
        np.testing.assert_array_equal(state['target_mean'].numpy(),data['target'][fit].mean(0).astype(np.float32))
        np.testing.assert_array_equal(state['target_std'].numpy(),data['target'][fit].std(0).astype(np.float32))
        pred,wit=plain_predict(path,{'tokens':data['tokens'][held]});i=[17,29,43].index(s)
        maximum=max(maximum,float(np.max(np.abs(pred-oof['prediction'][i,held]))),float(np.max(np.abs(wit-oof['witness'][i,held]))));cases+=1
    assert maximum==0
    result={'status':'PASS','oof_checkpoint_replays':cases,'target_scalers_checked_against_excluded_subject_fits':cases,
            'max_replay_difference':maximum,'calibration_test_read':False}
    (OUT/'oof_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


def test_audit():
    torch.set_num_threads(4);lock_path=OUT/'final_lock.json';digest=sha(lock_path);lock=verify_lock(lock_path,digest,'final')
    data=sealed('test',lock_path,digest);record=json.loads((OUT/'test_results.json').read_bytes())
    assert record['final_lock_sha256']==digest
    maximum=0.;curves=0;arrays=0
    for path in sorted((RUN/'test').glob('*.npz')):
        d=np.load(path);np.testing.assert_array_equal(d['target'],data['target'])
        error=np.array([scalar_de(p,q) for p,q in zip(d['prediction'],d['target'])])
        maximum=max(maximum,float(np.max(np.abs(error-d['error']))));arrays+=1
        name=path.stem
        if name.startswith('color_'):
            reported=record['color_comparators'][name[6:]];score=d['density']
        else:reported=record['risk_models'][name];score=d['raw']
        order=sorted(range(len(error)),key=lambda i:(float(score[i]),str(data['image'][i])))
        for entry in reported['coverage']:
            n=int(np.ceil(entry['coverage']*len(error)));e=sorted(float(error[i]) for i in order[:n])
            mean=sum(e)/n;median=e[n//2] if n%2 else (e[n//2-1]+e[n//2])/2
            maximum=max(maximum,abs(mean-entry['mean']),abs(median-entry['median']))
            for q,key in [(.9,'p90'),(.95,'p95')]:
                z=(n-1)*q;l=int(z);value=e[l]+(e[min(l+1,n-1)]-e[l])*(z-l);maximum=max(maximum,abs(value-entry[key]))
            curves+=1
    # Independently replay all selected head+calibration objects.
    design=deployment_designs(data,load('train'));head_gap=0.
    for key,choice in lock['selected'].items():
        version,arm=key.split('__');d=np.load(RUN/'test'/(key+'.npz'))
        raw=np.maximum(joblib.load(ROOT/choice['path']).predict(design[version][arm]),0)
        cal=joblib.load(ROOT/lock['calibrators'][key]['path']).predict(raw)
        head_gap=max(head_gap,float(np.max(np.abs(raw-d['raw']))),float(np.max(np.abs(cal-d['calibrated']))))
    assert maximum<1e-10 and head_gap==0
    result={'status':'PASS','test_prediction_arrays':arrays,'independent_coverage_cases':curves,
            'max_scalar_metric_gap':maximum,'head_and_calibrator_replays':len(lock['selected']),'max_head_replay_gap':head_gap,
            'final_lock_sha256':digest,'test_results_sha256':sha(OUT/'test_results.json')}
    (OUT/'test_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['oof','test']);args=parser.parse_args()
    oof_audit() if args.stage=='oof' else test_audit()
