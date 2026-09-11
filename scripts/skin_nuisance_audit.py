"""Independent metric and source-boundary audit plus exact GPU checkpoint replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_nuisance_model import NuisanceColor as SpatialColor
from skin_nuisance_train import OUT,RUN,PROTOCOL,prediction,subset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[12,36],default=36);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tr,va=load('train'),load('validation');lock=json.loads((OUT/'source_lock.json').read_bytes())
    for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest,name
    amendment=OUT/'test_precision_amendment'
    old=json.loads((amendment/'original_source_lock.json').read_bytes())
    changed=[k for k in old['bindings'] if old['bindings'][k]!=lock['bindings'][k]]
    assert len(changed)==1 and changed[0].replace('\\','/')=='tests/test_skin_nuisance.py'
    records=sorted(OUT.glob('*/result.json'));assert len(records)==args.fits
    gap=0.;arrays=0;comparisons=0;coverage_cases=0;initials={};identical_trajectories=0
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
            checkpoint=folder/(label+'.pt');assert sha(checkpoint)==r[label+'_sha256']
            s=torch.load(checkpoint,weights_only=True,map_location='cpu');assert s['protocol_sha256']==sha(PROTOCOL)
            if label=='final' and r['arm']=='learned':
                original_arm='graph_always'
                original=ROOT/'experiments/runs/skin_train_branch_v1'/f"{r['protocol']}__{original_arm}__s{r['seed']}"/'final.pt'
                old=torch.load(original,weights_only=True,map_location='cpu')['state']
                assert set(old)==set(s['state'])
                for name,value in s['state'].items():assert torch.equal(value,old[name]),name
                identical_trajectories+=1
            np.testing.assert_array_equal(s['target_mean'].numpy(),train['target'].mean(0).astype(np.float32))
            np.testing.assert_array_equal(s['target_std'].numpy(),train['target'].std(0).astype(np.float32))
            model=SpatialColor(r['arm']).cuda().eval();model.load_state_dict(s['state']);mean=s['target_mean'].cuda();std=s['target_std'].cuda()
            assert sum(p.numel() for p in model.parameters())==r['stored_parameters'] and model.active_parameters()==r['active_parameters']
            endpoints=[(selection,folder/(label+'_selection.npz'))]
            if label=='best':endpoints.append((evaluation,folder/'evaluation.npz'))
            for data,file in endpoints:
                x=torch.from_numpy(data['tokens']).cuda();p,risk=prediction(model,x,mean,std)
                with np.load(file) as saved:
                    np.testing.assert_array_equal(saved['target'],data['target'])
                    np.testing.assert_array_equal(saved['prediction'],p);np.testing.assert_array_equal(saved['risk'],risk);arrays+=1
                    if file.name=='evaluation.npz':
                        error=np.array([scalar_de(a,b) for a,b in zip(p,data['target'],strict=True)])
                        comparisons+=len(error);gap=max(gap,float(np.max(np.abs(error-saved['error']))),abs(error.mean()-r['full']['mean']))
                        order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
                        for cr in r['uncalibrated_dispersion_coverage']:
                            n=int(np.ceil(len(error)*cr['requested_coverage']));assert n==cr['accepted']
                            e=error[order[:n]]
                            for key,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:
                                gap=max(gap,abs(float(value)-cr[key]))
                            coverage_cases+=1
            del model,x,mean,std;torch.cuda.empty_cache()
    assert all(len(v)==1 for v in initials.values()) and gap<1e-10
    result={'status':'PASS','fits':len(records),'exact_prediction_arrays':arrays,'exact_risk_arrays':arrays,
            'independent_scalar_delta_e00_cases':comparisons,'independent_coverage_rows':coverage_cases,
            'maximum_metric_gap':float(gap),'identical_learned_control_trajectories':identical_trajectories,'fitting_only_target_scales_and_same_camera_selection_checked':True,
            'matched_initializations':True,'reserved_data_loaded':False,'source_lock_sha256':sha(OUT/'source_lock.json'),
            'audit_script_sha256':sha(Path(__file__)),'scalar_reference_sha256':sha(ROOT/'scripts/skin_mskcc_audit.py')}
    name='audit_mixed.json' if args.fits==12 else 'audit.json'
    (OUT/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
