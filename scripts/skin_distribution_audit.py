"""Exact replay plus scalar skin-error and manual quadrature decision audit."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import torch
from skin_distribution_train import OUT,RUN,PROTOCOL,prediction,endpoints
from skin_distribution_model import ColorDistribution,ARMS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def manual_decision(means,scales,probabilities,order):
    means=np.asarray(means,dtype=float);scales=np.asarray(scales,dtype=float)
    p=np.asarray(probabilities,dtype=float);p=p/p.sum()
    center=sum(p[j]*means[j] for j in range(len(p)))
    sd=np.sqrt(sum(p[j]*(scales[j]**2+(means[j]-center)**2) for j in range(len(p))))
    candidates=[center]+list(means)
    for a in (.25,-.25,.5,-.5):
        for axis in range(3):
            c=center.copy();c[axis]+=a*sd[axis];candidates.append(c)
    nodes=[-1.,1.] if order==2 else [-math.sqrt(3),0.,math.sqrt(3)]
    weights=[.5,.5] if order==2 else [1/6,2/3,1/6]
    pts=[];ws=[]
    for k in range(len(p)):
        for idx in itertools.product(range(order),repeat=3):
            pts.append(means[k]+scales[k]*np.array([nodes[i] for i in idx]))
            ws.append(p[k]*math.prod(weights[i] for i in idx))
    risks=[sum(w*scalar_de(c,x) for w,x in zip(ws,pts)) for c in candidates]
    best=int(np.argmin(risks))
    return candidates[best],risks[best],risks[0],len(candidates)*len(pts)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[12,36],default=36);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for p,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/p)==h,p
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==args.fits
    protocols=['mixed'] if args.fits==12 else ['mixed','from_SLR','from_ipod']
    keys={(json.loads(p.read_bytes())['protocol'],json.loads(p.read_bytes())['arm'],json.loads(p.read_bytes())['seed']) for p in paths}
    assert keys=={(p,a,s) for p in protocols for a in ARMS for s in (17,29,43)}
    tr,va=load('train'),load('validation');replay=scalar=coverage=curves=manual=0;gap=0.;initials={}
    for path in paths:
        rec=json.loads(path.read_bytes());folder=RUN/path.parent.name
        if rec['protocol']=='mixed':t,v,ev=tr,va,va
        else:
            c=rec['protocol'].removeprefix('from_')
            t=subset(tr,tr['device']==c);v=subset(va,va['device']==c);ev=subset(va,va['device']!=c)
            assert set(t['device']).isdisjoint(ev['device'])
        assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
        ym=t['target'].mean(0).astype(np.float32);ys=t['target'].std(0).astype(np.float32)
        torch.manual_seed(rec['seed']);model=ColorDistribution()
        assert digest(model.state_dict())==rec['initial_sha256']
        initials.setdefault((rec['protocol'],rec['seed']),set()).add(rec['initial_sha256'])
        model=model.cuda();mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert min(history,key=lambda h:h['selection_patient_mean'])['epoch']==rec['best_epoch']
        for tag in ('final','best'):
            assert sha(folder/f'{tag}.pt')==rec[f'{tag}_sha256']
            state=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True)
            assert state['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(ym,state['target_mean']);np.testing.assert_array_equal(ys,state['target_std'])
            model.load_state_dict(state['state']);p=prediction(model,torch.from_numpy(v['tokens']).cuda(),mean,std)[0]
            np.testing.assert_array_equal(p,np.load(folder/f'{tag}_selection.npz')['prediction']);replay+=1
        assert sha(folder/'evaluation.npz')==rec['evaluation_sha256']
        saved=np.load(folder/'evaluation.npz');p,g,h,s=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std)
        for key,arr in [('point',p),('gate',g),('hypotheses',h),('scales',s),('target',ev['target'])]:np.testing.assert_array_equal(arr,saved[key])
        replay+=1
        ep,_=endpoints(rec['arm'],p,g,h,s)
        for label,(pred,risk) in ep.items():
            np.testing.assert_array_equal(pred,saved[label+'_prediction']);np.testing.assert_array_equal(risk,saved[label+'_risk'])
            error=np.array([scalar_de(a,b) for a,b in zip(pred,ev['target'])]);scalar+=len(error)
            gap=max(gap,float(np.max(abs(error-saved[label+'_error']))),abs(float(error.mean())-rec['scores'][label]['full']['mean']))
            order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
            for c in rec['scores'][label]['coverage']:
                n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted']
                e=error[order[:n]]
                for key,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:
                    gap=max(gap,abs(float(value)-c[key]))
                coverage+=1
            curve=np.cumsum(error[order])/np.arange(1,len(error)+1)
            gap=max(gap,float(np.max(abs(curve-saved[label+'_curve']))));curves+=len(error)
        if rec['arm'] in ('gaussian','mdn4'):
            if rec['arm']=='gaussian':h=p[:,None];s=s.mean(1,keepdims=True);g=np.ones((len(p),1))
            for i in [0,len(p)-1]:
                for order in (2,3):
                    pred,risk,meanrisk,n=manual_decision(h[i],s[i],g[i],order)
                    manual+=n
                    gap=max(gap,float(np.max(abs(pred-ep[f'decision{order}'][0][i]))),abs(risk-ep[f'decision{order}'][1][i]))
                    if order==3:gap=max(gap,abs(meanrisk-ep['mean'][1][i]))
        del model;torch.cuda.empty_cache()
    assert gap<1e-9 and all(len(v)==1 for v in initials.values())
    out={'status':'PASS','fits':len(paths),'exact_color_replay_arrays':replay,'gate_hypothesis_scale_sets':len(paths),
        'independent_scalar_color_cases':scalar,'coverage_rows':coverage,'curve_points':curves,
        'manual_quadrature_scalar_cases':manual,'maximum_gap':gap,'matched_initializations':True,
        'fit_only_scalers_and_same_camera_selection_checked':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/('audit_mixed.json' if args.fits==12 else 'audit.json'),out);print(json.dumps(out))


if __name__=='__main__':main()
