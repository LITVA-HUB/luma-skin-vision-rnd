"""Native skin color from conditional patch distributions; source-only screen."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_patch_likelihood import fit_patch,likelihood,color_output
from skin_appearance_inverse_train import novelty
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import subset,rows,write
from skin_mskcc_summary_pilot import summarize
from skin_distribution_train import score
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_patch_likelihood_v1'
RUN=ROOT/'experiments/runs/skin_patch_likelihood_v1'
PROTOCOL=ROOT/'docs/research/skin_patch_likelihood_protocol_v1.md'
STRENGTHS=(1,4,16,64)


def configs():
    for representation in ('mean','bag'):
        for degree in (1,2):
            for components in (1,3):
                for seed in ((0,) if components==1 else (17,29,43)):yield representation,degree,components,seed


def fit(protocol,t,v,ev,representation,degree,components,seed):
    name=f'{protocol}__{representation}_d{degree}_k{components}_s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'model.npz')==r['model_sha256'];assert sha(folder/'evaluation.npz')==r['evaluation_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
    if protocol!='mixed':assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
    start=time.perf_counter();model,history=fit_patch(t,representation,degree,components,seed);fit_seconds=time.perf_counter()-start
    np.savez(folder/'model.npz',**model);write(out/'history.json',history)
    vl=likelihood(model,v['tokens']);selection={};vp={}
    for strength in STRENGTHS:
        p,_=color_output(vl,model['palette'],strength);vp[f't{strength}']=p
        selection[str(strength)]=summarize(delta_e00(p,v['target']),rows(v))
    chosen=min(STRENGTHS,key=lambda z:(selection[str(z)]['patient_balanced_mean'],STRENGTHS.index(z)))
    np.savez(folder/'selection.npz',loglik=vl,**vp)
    el=vl if protocol=='mixed' else likelihood(model,ev['tokens']);cl=likelihood(model,ev['tokens'],collapse=True)
    tx=t['tokens'].astype(np.float64)[:,:,9:12].mean(1);ex=ev['tokens'].astype(np.float64)[:,:,9:12].mean(1);common=novelty(tx,ex)
    arrays={'target':ev['target'],'patient':ev['patient'],'site':ev['site'],'loglik':el,'collapse_loglik':cl,'common_risk':common};metrics={}
    for key,ll,strength in [(f't{s}',el,s) for s in STRENGTHS]+[('collapsed',cl,chosen)]:
        p,posterior=color_output(ll,model['palette'],strength)
        own=(posterior*delta_e00(p[:,None],model['palette'][None])).sum(1)
        m,e,order=score(p,common,ev);m.pop('predicted_error_mae_or_dispersion_mae');pm,_,po=score(p,own,ev)
        metrics[key]={'common':m,'posterior':pm};arrays[key]=p;arrays[key+'_error']=e;arrays[key+'_risk']=own
        arrays[key+'_curve']=np.cumsum(e[order])/np.arange(1,len(e)+1);arrays[key+'_posterior_curve']=np.cumsum(e[po])/np.arange(1,len(e)+1)
    np.savez(folder/'evaluation.npz',**arrays)
    r={'protocol':protocol,'representation':representation,'degree':degree,'components':components,'seed':seed,'selected_strength':chosen,
        'selection':selection,'metrics':metrics,'fit_seconds':fit_seconds,'stored_numeric_scalars':sum(a.size for a in model.values() if a.dtype.kind in 'fiu'),
        'model_bytes':(folder/'model.npz').stat().st_size,'model_sha256':sha(folder/'model.npz'),'selection_sha256':sha(folder/'selection.npz'),'evaluation_sha256':sha(folder/'evaluation.npz'),
        'train_people':len(set(t['patient'])),'selection_people':len(set(v['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(t['target']),'evaluation_images':len(ev['target']),'source_exploratory_only':True,'risk_calibrated':False,
        'reserved_endpoint_access':False,'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__))}
    write(out/'result.json',r);print(json.dumps({'completed':name,'strength':chosen,'mean':metrics[f't{chosen}']['common']['full']['mean'],'collapsed':metrics['collapsed']['common']['full']['mean'],'seconds':fit_seconds}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','run'));args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_patch_likelihood.py',ROOT/'tests/test_skin_patch_likelihood.py']
    files += [ROOT/'scripts'/n for n in ('skin_appearance_inverse.py','skin_appearance_inverse_train.py','skin_mskcc_pixels.py','skin_mskcc_data.py','skin_pair_train.py','skin_distribution_train.py','skin_mskcc_summary_pilot.py')]
    files += [ROOT/'src/luma_skin_vision/color.py']+[ROOT/'data/processed/skin_mskcc_pixels_v1'/f'{r}.npz' for r in ('train','validation')]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'fits':48,'strengths':STRENGTHS})
        print('Patch likelihood protocol frozen before fits');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    tr,va=load('train'),load('validation')
    protocols=[('mixed',tr,va,va)]+[('from_'+c,subset(tr,tr['device']==c),subset(va,va['device']==c),subset(va,va['device']!=c)) for c in ('SLR','ipod')]
    with threadpool_limits(1):
        for protocol,t,v,e in protocols:
            for config in configs():fit(protocol,t,v,e,*config)


if __name__=='__main__':main()
