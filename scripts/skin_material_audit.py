"""Replay real-image fits and independently verify skin color and coverage."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_material_train import OUT,RUN,PROTOCOL,prediction
from skin_material_model import MaterialImage,ARMS
from skin_material_prior import PRIOR
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[15,45],default=45);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    tr,va=load('train'),load('validation')
    for file in ('prior_receipt.json','source_lock.json'):
        for p,h in json.loads((OUT/file).read_bytes())['bindings'].items():assert sha(ROOT/p)==h,p
    records=sorted(OUT.glob('*/result.json'));assert len(records)==args.fits
    keys={(json.loads(p.read_bytes())['protocol'],json.loads(p.read_bytes())['arm'],json.loads(p.read_bytes())['seed']) for p in records}
    protocols=['mixed'] if args.fits==15 else ['mixed','from_SLR','from_ipod']
    assert keys=={(p,a,s) for p in protocols for a in ARMS for s in [17,29,43]}
    gap=0.;replays=0;scalar_cases=0;coverages=0;curve_points=0;initials={}
    prior=dict(np.load(PRIOR,allow_pickle=False))
    for path in records:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if r['protocol']=='mixed':train,selection,evaluation=tr,va,va
        else:
            camera=r['protocol'].removeprefix('from_')
            train=subset(tr,tr['device']==camera);selection=subset(va,va['device']==camera);evaluation=subset(va,va['device']!=camera)
            assert set(train['device']).isdisjoint(evaluation['device'])
        assert set(train['patient']).isdisjoint(selection['patient']) and set(train['patient']).isdisjoint(evaluation['patient'])
        ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
        torch.manual_seed(r['seed']);model=MaterialImage(r['arm'],prior,ym,ys)
        assert digest(model.state_dict())==r['initial_state_sha256']
        initials.setdefault((r['protocol'],r['seed']),set()).add(r['initial_state_sha256'])
        model=model.cuda();mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
        history=json.loads((path.parent/'history.json').read_bytes())
        assert len(history)==80
        assert min(history,key=lambda x:x['selection_patient_mean'])['epoch']==r['best_epoch']
        for tag in ('final','best'):
            assert sha(folder/f'{tag}.pt')==r[f'{tag}_sha256']
            state=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(state['target_mean'].numpy(),ym);np.testing.assert_array_equal(state['target_std'].numpy(),ys)
            for key in ('mu','basis','matrix','white','base','jacobian'):
                np.testing.assert_array_equal(state['state'][key].numpy(),np.asarray(prior[key],dtype=np.float32))
            model.load_state_dict(state['state'])
            p,_,_,_=prediction(model,torch.from_numpy(selection['tokens']).cuda(),mean,std)
            saved=np.load(folder/f'{tag}_selection.npz');np.testing.assert_array_equal(p,saved['prediction']);np.testing.assert_array_equal(saved['target'],selection['target']);replays+=1
        p,g,h,free=prediction(model,torch.from_numpy(evaluation['tokens']).cuda(),mean,std)
        saved=np.load(folder/'evaluation.npz');np.testing.assert_array_equal(p,saved['prediction']);replays+=1
        np.testing.assert_array_equal(saved['target'],evaluation['target'])
        for name,arr in [('gate',g),('hypotheses',h),('residual_or_free_coordinates',free)]:np.testing.assert_array_equal(saved[name],arr)
        error=np.array([scalar_de(a,b) for a,b in zip(p,evaluation['target'],strict=True)])
        scalar_cases+=len(error);gap=max(gap,float(np.max(np.abs(error-saved['error']))),abs(float(error.mean())-r['full']['mean']))
        risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
        np.testing.assert_array_equal(risk,saved['risk'])
        order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
        for c in r['uncalibrated_disagreement_coverage']:
            n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted']
            e=error[order[:n]]
            for key,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:
                gap=max(gap,abs(float(value)-c[key]))
            coverages+=1
        curve=np.load(folder/'risk_curve.npz')
        expected=np.cumsum(error[order])/np.arange(1,len(error)+1)
        gap=max(gap,float(np.max(np.abs(expected-curve['mean_error']))));curve_points+=len(error)
        np.testing.assert_array_equal(curve['coverage'],np.arange(1,len(error)+1)/len(error))
        del model;torch.cuda.empty_cache()
    assert gap<1e-10 and all(len(v)==1 for v in initials.values())
    result={'status':'PASS','fits':len(records),'exact_color_prediction_arrays':replays,
        'gate_hypothesis_free_coordinate_sets':len(records),'independent_scalar_delta_e00_cases':scalar_cases,
        'coverage_rows':coverages,'full_curve_points':curve_points,'maximum_metric_gap':gap,
        'fitting_only_target_scales_and_fixed_material_buffers_checked':True,'matched_initializations':True,
        'same_camera_epoch_selection_checked':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/('audit_mixed.json' if args.fits==15 else 'audit.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
