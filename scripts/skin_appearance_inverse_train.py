"""Frozen closed-form forward/inverse skin color experiment, source roles only."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_appearance_inverse import image_features,fit_map,predict_map,fit_forward,infer_forward
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import subset,write,rows
from skin_distribution_train import score
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_appearance_inverse_v1'
RUN=ROOT/'experiments/runs/skin_appearance_inverse_v1'
PROTOCOL=ROOT/'docs/research/skin_appearance_inverse_protocol_v1.md'
ALPHAS=(.01,1.,100.)


def configs():
    for kind in ('rgb','stats'):
        for direction in ('direct','global','mode'):
            for degree in (1,2):
                for alpha in ALPHAS:yield kind,direction,degree,alpha


def novelty(t,x):
    scale=np.maximum(t.std(0),1e-6)
    return np.sqrt(np.min(np.sum(((x[:,None]-t[None])/scale)**2,axis=2),axis=1))


def outputs(model,x,direction):
    if direction=='direct':return {'direct':(predict_map(model,x),None)},{}
    posterior,evidence,d=infer_forward(model,x)
    return {key:(d[key],d[key+'_risk']) for key in ('mean','medoid')},{'posterior':posterior,'log_evidence':evidence}


def run_fit(protocol,tr,va,ev,kind,direction,degree,alpha):
    name=f'{protocol}__{kind}__{direction}_d{degree}_a{alpha:g}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes())
        assert sha(folder/'model.npz')==r['model_sha256'] and sha(folder/'evaluation.npz')==r['evaluation_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient']) and set(tr['patient']).isdisjoint(ev['patient'])
    if protocol!='mixed':assert set(tr['device']).isdisjoint(ev['device']) and set(va['device']).isdisjoint(ev['device'])
    x=image_features(tr,kind);vx=image_features(va,kind);ex=image_features(ev,kind);start=time.perf_counter()
    model=fit_map(x,tr['target'],degree,alpha) if direction=='direct' else fit_forward(tr,kind,direction,degree,alpha)
    fit_seconds=time.perf_counter()-start;np.savez(folder/'model.npz',**model)
    vo,vs=outputs(model,vx,direction);eo,es=outputs(model,ex,direction);risk=novelty(x,ex)
    selection={k:summarize(delta_e00(p,va['target']),rows(va)) for k,(p,r) in vo.items()}
    arrays={'target':ev['target'],'patient':ev['patient'],'site':ev['site'],'common_risk':risk,**es};metrics={}
    for key,(p,ownrisk) in eo.items():
        m,error,order=score(p,risk,ev);m.pop('predicted_error_mae_or_dispersion_mae')
        metrics[key]={'common':m};arrays[key]=p;arrays[key+'_error']=error;arrays[key+'_curve']=np.cumsum(error[order])/np.arange(1,len(error)+1)
        if ownrisk is not None:
            om,_,order=score(p,ownrisk,ev);metrics[key]['posterior']=om
            arrays[key+'_risk']=ownrisk;arrays[key+'_posterior_curve']=np.cumsum(error[order])/np.arange(1,len(error)+1)
    np.savez(folder/'selection.npz',**{k:p for k,(p,r) in vo.items()})
    np.savez(folder/'evaluation.npz',**arrays)
    r={'protocol':protocol,'kind':kind,'direction':direction,'degree':degree,'alpha':alpha,'selection':selection,'metrics':metrics,
        'fit_seconds':fit_seconds,'model_sha256':sha(folder/'model.npz'),'selection_sha256':sha(folder/'selection.npz'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'model_bytes':(folder/'model.npz').stat().st_size,
        'inference_scalar_storage':sum(v.size for k,v in model.items() if k not in ('oof_prediction','oof_folds')),
        'oof_folds':int(model.get('oof_folds',0)),'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),
        'evaluation_people':len(set(ev['patient'])),'train_images':len(x),'evaluation_images':len(ex),
        'source_exploratory_only':True,'risk_calibrated':False,'reserved_endpoint_access':False,
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__))}
    if direction!='direct':
        p=es['posterior'];entropy=-np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)
        r['posterior_diagnostic']={'mean_effective_atoms':float(np.exp(entropy).mean()),'median_effective_atoms':float(np.median(np.exp(entropy))),
            'mean_top_atom_mass':float(p.max(1).mean()),'mean_log_evidence_standardized_units':float(es['log_evidence'].mean())}
    write(out/'result.json',r)
    print(json.dumps({'completed':name,'mean':{k:m['common']['full']['mean'] for k,m in metrics.items()},'fit_seconds':fit_seconds}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_appearance_inverse.py',ROOT/'tests/test_skin_appearance_inverse.py']
    files += [ROOT/'scripts'/n for n in ('skin_mskcc_pixels.py','skin_mskcc_data.py','skin_pair_train.py','skin_distribution_train.py','skin_mskcc_summary_pilot.py')]
    files += [ROOT/'src/luma_skin_vision/color.py']+[ROOT/'data/processed/skin_mskcc_pixels_v1'/f'{r}.npz' for r in ('train','validation')]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'configurations_per_protocol':36,'protocols':['mixed','from_SLR','from_ipod']})
        print('Frozen: no fitting or reserved endpoint access');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    tr,va=load('train'),load('validation')
    protocols=[('mixed',tr,va,va)]+[('from_'+c,subset(tr,tr['device']==c),subset(va,va['device']==c),subset(va,va['device']!=c)) for c in ('SLR','ipod')]
    with threadpool_limits(1):
        for protocol,t,v,e in protocols:
            for config in configs():run_fit(protocol,t,v,e,*config)


if __name__=='__main__':main()
