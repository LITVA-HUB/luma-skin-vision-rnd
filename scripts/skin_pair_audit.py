"""Read-only source replay, fit-boundary and independent DeltaE00 audit."""
import json
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_invariance import PairedColor,nuisance_transform
from skin_pair_train import OUT,RUN,PROTOCOL,subset,prediction


def main():
    torch.set_num_threads(4);tr,va=load('train'),load('validation')
    lock=json.loads((OUT/'source_lock.json').read_bytes())
    for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest,name
    records=sorted(OUT.glob('*/result.json'));assert len(records)==27
    maximum=0.;replays=0;initials={};comparisons=0
    for path in records:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if r['protocol']=='mixed':train,selection,evaluation=tr,va,va
        else:
            camera=r['protocol'][5:]
            train=subset(tr,tr['device']==camera);selection=subset(va,va['device']==camera);evaluation=subset(va,va['device']!=camera)
            assert set(train['device']).isdisjoint(evaluation['device'])
            assert set(selection['device']).isdisjoint(evaluation['device'])
        assert set(train['patient']).isdisjoint(selection['patient'])
        assert set(train['patient']).isdisjoint(evaluation['patient'])
        initials.setdefault((r['protocol'],r['seed']),set()).add(r['initial_backbone_sha256'])
        for label in ['best','final']:
            checkpoint=folder/(label+'.pt');assert sha(checkpoint)==r[label+'_sha256']
            s=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert s['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(s['target_mean'].numpy(),train['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),train['target'].std(0).astype(np.float32))
            center,scale,q,_=nuisance_transform(train['tokens'],train['site'],3 if r['arm']=='quotient3' else 0)
            if r['arm'] not in ['standardized','quotient3']:center=np.zeros(18);scale=np.ones(18);q=np.eye(18)
            for key,expected in [('center',center),('scale',scale),('quotient',q)]:np.testing.assert_array_equal(s[key].numpy(),expected)
            model=PairedColor(s['target_std'],r['arm']).cuda().eval();model.load_state_dict(s['state'])
            mean,std=s['target_mean'].cuda(),s['target_std'].cuda()
            endpoints=[(selection,folder/(label+'_selection.npz'))]
            if label=='best':endpoints.append((evaluation,folder/'evaluation.npz'))
            for data,array in endpoints:
                x=torch.from_numpy((((data['tokens'].astype(np.float64)-center)/scale)@q).astype(np.float32)).cuda()
                p=prediction(model,x,mean,std);saved=np.load(array)
                np.testing.assert_array_equal(saved['target'],data['target'])
                np.testing.assert_array_equal(saved['prediction'],p);replays+=1
                if array.name=='evaluation.npz':
                    e=np.array([scalar_de(a,b) for a,b in zip(p,data['target'])])
                    maximum=max(maximum,float(abs(e-saved['error']).max()),abs(float(e.mean())-r['full']['mean']))
                    comparisons+=len(e)
            del model,x,mean,std;torch.cuda.empty_cache()
    assert all(len(v)==1 for v in initials.values())
    assert maximum<1e-10
    result={'status':'PASS','fits':len(records),'exact_prediction_array_replays':replays,
        'independent_scalar_delta_e00_comparisons':comparisons,'maximum_metric_gap':maximum,
        'all_target_scalers_and_quotients_checked_against_training_only':True,
        'matched_backbone_initializations':True,'reserved_test_or_calibration_read':False,
        'source_lock_sha256':sha(OUT/'source_lock.json')}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
