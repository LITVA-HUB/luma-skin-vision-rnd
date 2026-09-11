"""Read-only replay and fit-boundary audit of the fixed 54-run factorial."""
import json
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_capture_model import CaptureColor,MODES
from skin_capture_train import OUT,RUN,PROTOCOL,prediction,subset


def main():
    torch.set_num_threads(4);tr,va=load('train'),load('validation')
    lock=json.loads((OUT/'source_lock.json').read_bytes())
    for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest,name
    records=sorted(OUT.glob('*/result.json'));assert len(records)==54
    gap=0.;replays=0;comparisons=0;initials={}
    for path in records:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if r['protocol']=='mixed':train,selection,evaluation=tr,va,va
        else:
            c=r['protocol'][5:];train=subset(tr,tr['device']==c);selection=subset(va,va['device']==c);evaluation=subset(va,va['device']!=c)
            assert set(train['device']).isdisjoint(evaluation['device']) and set(selection['device']).isdisjoint(evaluation['device'])
        assert set(train['patient']).isdisjoint(selection['patient']) and set(train['patient']).isdisjoint(evaluation['patient'])
        initials.setdefault(r['seed'],set()).add(r['initial_state_sha256'])
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert int(np.argmin([h['selection_patient_mean'] for h in history]))+1==r['best_epoch']
        for label in ['best','final']:
            file=folder/(label+'.pt');assert sha(file)==r[label+'_sha256'];s=torch.load(file,weights_only=True,map_location='cpu')
            assert s['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(s['target_mean'].numpy(),train['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),train['target'].std(0).astype(np.float32))
            model=CaptureColor(r['arch']).cuda().eval();model.load_state_dict(s['state']);mean=s['target_mean'].cuda();std=s['target_std'].cuda()
            endpoints=[(selection,folder/(label+'_selection.npz'))]
            if label=='best':endpoints.append((evaluation,folder/'evaluation.npz'))
            for data,file in endpoints:
                x=torch.from_numpy(data['tokens']).cuda();p,g,h=prediction(model,x,mean,std);saved=np.load(file)
                np.testing.assert_array_equal(saved['target'],data['target']);np.testing.assert_array_equal(saved['prediction'],p);replays+=1
                if file.name=='evaluation.npz':
                    np.testing.assert_array_equal(saved['gate'],g);np.testing.assert_array_equal(saved['hypotheses'],h)
                    e=np.array([scalar_de(a,b) for a,b in zip(p,data['target'])]);gap=max(gap,float(abs(e-saved['error']).max()),abs(float(e.mean())-r['full']['mean']));comparisons+=len(e)
                    accuracy=float((g.argmax(1)==np.array([MODES.index(str(v)) for v in data['mode']])).mean())
                    assert accuracy==r['mode_accuracy']
            del model,x,mean,std;torch.cuda.empty_cache()
    assert all(len(v)==1 for v in initials.values()) and gap<1e-10
    result={'status':'PASS','fits':len(records),'exact_prediction_array_replays':replays,'gate_and_hypothesis_replays':len(records),
        'independent_scalar_delta_e00_comparisons':comparisons,'maximum_metric_gap':gap,
        'fitting_only_target_scales_and_same_camera_selection_checked':True,'matched_initializations':True,
        'reserved_test_or_calibration_read':False,'source_lock_sha256':sha(OUT/'source_lock.json')}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
