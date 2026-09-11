"""Replay image fields, source selections and independent scalar skin metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_loss_field_train import OUT,RUN,PRIOR,PROTOCOL,prediction,gpu_palette
from skin_loss_field import LossFieldImage,ARMS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--fits',choices=[12,36],type=int,default=36);args=p.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for path,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/path)==h,path
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==args.fits
    protocols=['mixed'] if args.fits==12 else ['mixed','from_SLR','from_ipod']
    keys={(json.loads(p.read_bytes())['protocol'],json.loads(p.read_bytes())['arm'],json.loads(p.read_bytes())['seed']) for p in paths}
    assert keys=={(p,a,s) for p in protocols for a in ARMS for s in (17,29,43)}
    tr,va=load('train'),load('validation');replay=scalar=coverage=curves=field_rows=0;gap=0.;fieldgap=0.;initials={}
    for path in paths:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if r['protocol']=='mixed':t,v,ev=tr,va,va
        else:
            c=r['protocol'].removeprefix('from_');t=subset(tr,tr['device']==c);v=subset(va,va['device']==c);ev=subset(va,va['device']!=c)
            assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
        assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
        palette=dict(np.load(PRIOR/f"{r['protocol']}.npz"));atoms,ids=np.unique(t['target'],axis=0,return_inverse=True)
        np.testing.assert_array_equal(atoms,palette['atoms']);np.testing.assert_array_equal(ids,palette['index'])
        ym=t['target'].mean(0).astype(np.float32);ys=t['target'].std(0).astype(np.float32)
        torch.manual_seed(r['seed']);model=LossFieldImage(len(atoms));assert digest(model.state_dict())==r['initial_sha256']
        initials.setdefault((r['protocol'],r['seed']),set()).add(r['initial_sha256'])
        model=model.cuda();pal=gpu_palette(palette);mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert min(history,key=lambda h:h['selection_patient_mean'])['epoch']==r['best_epoch']
        for tag in ('final','best'):
            assert sha(folder/f'{tag}.pt')==r[f'{tag}_sha256']
            state=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==sha(PROTOCOL) and state['palette_sha256']==sha(PRIOR/f"{r['protocol']}.npz")
            np.testing.assert_array_equal(state['target_mean'].numpy(),ym);np.testing.assert_array_equal(state['target_std'].numpy(),ys)
            model.load_state_dict(state['state']);out=prediction(model,torch.from_numpy(v['tokens']).cuda(),mean,std,pal,r['arm'])[0]
            np.testing.assert_array_equal(out,np.load(folder/f'{tag}_selection.npz')['prediction']);replay+=1
        assert sha(folder/'evaluation.npz')==r['evaluation_sha256'];a=dict(np.load(folder/'evaluation.npz'))
        pred,secondary,risk,w,idx=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std,pal,r['arm'])
        for key,arr in [('prediction',pred),('secondary',secondary),('risk',risk),('weights',w),('indices',idx),('target',ev['target'])]:np.testing.assert_array_equal(arr,a[key])
        replay+=1
        assert np.max(abs(w.sum(1)-1))<2e-6
        assert float((w<0).mean())==r['negative_weight_fraction'] and float((risk<0).mean())==r['negative_risk_fraction']
        if r['arm']!='direct':
            np.testing.assert_array_equal(pred,palette['grid'].astype(np.float32)[idx])
            # Validate selected action and risk against FP64 products with an
            # explicit accumulated FP32 roundoff bound, including affine cancellation.
            for i in (0,len(w)-1):
                row=w[i].astype(float)@palette['cost'].astype(float)
                gamma=len(w[i])*np.finfo(np.float32).eps/(1-len(w[i])*np.finfo(np.float32).eps)
                bound=gamma*(abs(w[i]).astype(float)@abs(palette['cost']).astype(float))
                assert abs(float(risk[i])-row[idx[i]]) <= bound[idx[i]]+1e-6
                assert row[idx[i]]-row.min() <= bound[idx[i]]+bound.max()+1e-6
                fieldgap=max(fieldgap,abs(float(risk[i])-row[idx[i]]));field_rows+=1
        for label,out in [('primary',pred),('secondary',secondary)]:
            error=np.array([scalar_de(x,y) for x,y in zip(out,ev['target'])]);scalar+=len(error)
            gap=max(gap,float(np.max(abs(error-a[label+'_error']))),abs(float(error.mean())-r['scores'][label]['full']['mean']))
            order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
            for c in r['scores'][label]['coverage']:
                n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted'];e=error[order[:n]]
                for key,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:gap=max(gap,abs(float(value)-c[key]))
                coverage+=1
            curve=np.cumsum(error[order])/np.arange(1,len(error)+1);gap=max(gap,float(np.max(abs(curve-a[label+'_curve']))));curves+=len(error)
        del model,pal;torch.cuda.empty_cache()
    assert gap<1e-9 and all(len(v)==1 for v in initials.values())
    result={'status':'PASS','fits':len(paths),'exact_color_replay_arrays':replay,'independent_scalar_color_cases':scalar,
        'coverage_rows':coverage,'curve_points':curves,'fp64_field_rows_with_fp32_bound':field_rows,
        'max_field_dot_product_gap':fieldgap,'max_metric_gap':gap,'shared_initialization_checked':True,
        'fit_only_dictionary_scalers_and_same_camera_selection_checked':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/('audit_mixed.json' if args.fits==12 else 'audit.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
