"""Matched risk fitting -> locked calibration -> locked independent skin test."""
import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,MANIFEST,RAW,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,RUN,PROTOCOL,designs,deployment_designs,binding,verify_lock
from skin_mskcc_selective_data import sealed
from skin_mskcc_selective_metrics import selection,evaluate,accepted,COVERAGES
from skin_mskcc_color_registry import paths as color_paths,all_predictions


def write(path,record):
    Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf8')


def candidates():
    for alpha in [1,10]:
        yield f'mlp{alpha}',TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64,32),activation='tanh',solver='lbfgs',alpha=alpha,max_iter=2000,random_state=17)),transformer=StandardScaler())
    yield 'hgb',HistGradientBoostingRegressor(loss='squared_error',max_iter=100,learning_rate=.05,
        max_depth=2,min_samples_leaf=32,l2_regularization=10,random_state=17,early_stopping=False)


def score(model,x):
    y=np.maximum(model.predict(x),0)
    if not np.isfinite(y).all():raise ValueError('Nonfinite predicted color risk')
    return y


def fit_heads():
    torch.set_num_threads(4);train,val=load('train'),load('validation')
    head_dir=RUN/'heads';head_dir.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((OUT/'oof.json').read_bytes());assert receipt['oof_sha256']==sha(RUN/'oof.npz')
    oof=np.load(RUN/'oof.npz');assert np.array_equal(oof['target'],train['target'])
    assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['core_sha256']==sha(ROOT/'scripts/skin_mskcc_selective_core.py')
    td=designs(train,oof['prediction'],oof['witness'],oof['density']);vd=deployment_designs(val,train)
    selected={};records=[]
    for version in td:
        te=delta_e00(td[version]['prediction'],train['target']);ve=delta_e00(vd[version]['prediction'],val['target'])
        for arm in ['C_plus','Proposed']:
            key=version+'__'+arm;options=[]
            for name,model in candidates():
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(4):
                    warnings.simplefilter('always');model.fit(td[version][arm],te)
                raw=score(model,vd[version][arm]);risk80,aurc=selection(ve,raw,val)
                path=head_dir/(key+'__'+name+'.joblib');joblib.dump(model,path)
                if not np.array_equal(score(joblib.load(path),vd[version][arm]),raw):raise ValueError('Head replay differs')
                r={'version':version,'arm':arm,'candidate':name,'path':str(path.relative_to(ROOT)),
                   'sha256':sha(path),'input_features':td[version][arm].shape[1],'validation_risk80':risk80,
                   'validation_aurc60_100':aurc,'warnings':[str(w.message) for w in caught]}
                records.append(r);options.append(r)
                np.savez(head_dir/(key+'__'+name+'_validation.npz'),raw=raw,error=ve,prediction=vd[version]['prediction'])
            chosen=min(options,key=lambda r:(r['validation_risk80'],r['validation_aurc60_100'],r['candidate']))
            selected[key]=chosen
            print(json.dumps({'selected':key,'candidate':chosen['candidate'],'source_risk80':chosen['validation_risk80']}),flush=True)
    write(OUT/'head_search.json',{'candidates':records,'selected':selected,'scope':'SOURCE VALIDATION selection only; no calibration/test used'})
    stable=[PROTOCOL,MANIFEST,ROOT/'pyproject.toml',ROOT/'uv.lock',RUN/'oof.npz',OUT/'oof.json',OUT/'head_search.json']
    stable+=list((ROOT/'scripts').glob('skin_mskcc*.py'))
    stable+=list((ROOT/'src/luma_skin_vision').glob('color.py'))
    stable+=list((ROOT/'data/processed/skin_mskcc_pixels_v1').glob('*.npz'))
    stable+=[ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/cache.json',RAW/'image_receipts.json']
    stable+=color_paths()+list(head_dir.glob('*.joblib'))
    lock={'stage':'precalibration','protocol_sha256':sha(PROTOCOL),'bindings':binding(stable),
          'selected':selected,'primary_version':'ensemble','secondary_version':'single17',
          'no_calibration_or_test_endpoints_used':True}
    write(OUT/'precalibration_lock.json',lock)
    print(json.dumps({'precalibration_lock_sha256':sha(OUT/'precalibration_lock.json')}),flush=True)


def calibrate():
    torch.set_num_threads(4);pre=OUT/'precalibration_lock.json';digest=sha(pre)
    lock=verify_lock(pre,digest,'precalibration');data=sealed('calibration',pre,digest);train=load('train')
    dd=deployment_designs(data,train);folder=RUN/'calibration';folder.mkdir(exist_ok=False,parents=True)
    people,counts=np.unique(data['patient'],return_counts=True);count=dict(zip(people,counts));weights=np.array([1/count[p] for p in data['patient']])
    records={}
    for key,choice in lock['selected'].items():
        version,arm=key.split('__');raw=score(joblib.load(ROOT/choice['path']),dd[version][arm])
        error=delta_e00(dd[version]['prediction'],data['target'])
        model=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(raw,error,sample_weight=weights)
        path=folder/(key+'.joblib');joblib.dump(model,path);calibrated=model.predict(raw)
        thresholds={str(c):float(np.quantile(raw,c)) for c in COVERAGES}
        bins=np.quantile(calibrated,[.2,.4,.6,.8]).tolist()
        records[key]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'raw_coverage_thresholds':thresholds,'calibration_bin_edges':bins,
                      'calibration_fit_mae':float(np.mean(np.abs(calibrated-error))),
                      'scope':'In-sample calibration diagnostic, not independent performance'}
        np.savez(folder/(key+'.npz'),raw=raw,calibrated=calibrated,error=error,prediction=dd[version]['prediction'])
    write(OUT/'calibration.json',records)
    files=[ROOT/name for name in lock['bindings']]+[pre,OUT/'calibration.json',OUT/'calibration_data.json',ROOT/'data/processed/skin_mskcc_selective_v1/calibration.npz']
    files+=list(folder.glob('*.joblib'))+list(folder.glob('*.npz'))
    final={'stage':'final','bindings':binding(files),'precalibration_lock_sha256':digest,'primary_version':'ensemble',
           'selected':lock['selected'],'calibrators':records,'test_endpoints_opened':False,'protocol_sha256':sha(PROTOCOL)}
    write(OUT/'final_lock.json',final)
    print(json.dumps({'final_lock_sha256':sha(OUT/'final_lock.json'),'calibration_people':6,'test_not_opened':True}),flush=True)


def bootstrap_pair(error,score_a,score_b,data):
    rng=np.random.default_rng(20260911);people=np.unique(data['patient']);groups=[np.flatnonzero(data['patient']==p) for p in people]
    differences=[]
    for _ in range(2000):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))]);k=int(np.ceil(.8*len(ix)))
        a=np.lexsort((data['image'][ix],score_a[ix]))[:k];b=np.lexsort((data['image'][ix],score_b[ix]))[:k]
        differences.append(float(error[ix][a].mean()-error[ix][b].mean()))
    return {'proposed_minus_C_plus_risk80_ci95':np.quantile(differences,[.025,.975]).tolist(),
            'replicates':2000,'unit':'patient cluster; re-rank each resample','scope':'Paired confirmatory primary comparison; secondary variants/multiple comparisons not adjusted'}


def test(lock_sha):
    torch.set_num_threads(4);path=OUT/'final_lock.json';lock=verify_lock(path,lock_sha,'final')
    data=sealed('test',path,lock_sha);train=load('train');dd=deployment_designs(data,train)
    result={'final_lock_sha256':lock_sha,'population':{'images':len(data['target']),'people':len(set(data['patient']))},
            'known_cameras_only':True,'risk_models':{},'color_comparators':{},'paired':{}}
    folder=RUN/'test';folder.mkdir(exist_ok=False,parents=True);raws={}
    for key,choice in lock['selected'].items():
        version,arm=key.split('__');raw=score(joblib.load(ROOT/choice['path']),dd[version][arm]);raws[key]=raw
        error=delta_e00(dd[version]['prediction'],data['target'])
        calibration=lock['calibrators'][key];calibrated=joblib.load(ROOT/calibration['path']).predict(raw)
        r=evaluate(error,raw,data);r['selected_head']=choice['candidate'];r['predicted_error_mae']=float(np.mean(np.abs(calibrated-error)))
        r['predicted_error_rmse']=float(np.sqrt(np.mean((calibrated-error)**2)))
        r['calibration_threshold_outcomes']={c:accepted(error,raw<=t,data) for c,t in calibration['raw_coverage_thresholds'].items()}
        r['predicted_error_threshold_outcomes']={str(t):accepted(error,calibrated<=t,data) for t in [2,5]}
        bins=np.searchsorted(calibration['calibration_bin_edges'],calibrated,side='right');r['calibration_bins']=[]
        for b in range(5):
            ix=np.flatnonzero(bins==b)
            if len(ix):r['calibration_bins'].append({'bin':b,'n':len(ix),'mean_predicted_error':float(calibrated[ix].mean()),'mean_observed_error':float(error[ix].mean())})
        result['risk_models'][key]=r
        np.savez(folder/(key+'.npz'),prediction=dd[version]['prediction'],target=data['target'],error=error,raw=raw,calibrated=calibrated,
                 image=data['image'],patient=data['patient'])
    for version in dd:
        a=result['risk_models'][version+'__Proposed'];b=result['risk_models'][version+'__C_plus']
        assert a['full']==b['full']
        error=delta_e00(dd[version]['prediction'],data['target'])
        r=bootstrap_pair(error,raws[version+'__Proposed'],raws[version+'__C_plus'],data)
        r['proposed_minus_C_plus_full_mean']=0.
        r['proposed_minus_C_plus_risk80']=a['coverage'][3]['mean']-b['coverage'][3]['mean'];result['paired'][version]=r
    colors=all_predictions(data);dist=next(iter(dd.values()))['C_plus'][:,-1]
    for name,p in colors.items():
        error=delta_e00(p,data['target']);result['color_comparators'][name]=evaluate(error,dist,data)
        np.savez(folder/('color_'+name+'.npz'),prediction=p,target=data['target'],error=error,density=dist)
    result['lowest_observed_color_mean']=min(result['color_comparators'],key=lambda n:result['color_comparators'][n]['full']['mean'])
    result['comparison_note']='All color comparators frozen before test; lowest observed is descriptive, not a newly fitted selection.'
    write(OUT/'test_results.json',result)
    print(json.dumps({'population':result['population'],'primary_pair':result['paired']['ensemble'],
                      'primary_C_plus':result['risk_models']['ensemble__C_plus']['coverage'][3],
                      'primary_Proposed':result['risk_models']['ensemble__Proposed']['coverage'][3],
                      'lowest_observed_color_mean':result['lowest_observed_color_mean']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['heads','calibrate','test']);parser.add_argument('--lock-sha')
    args=parser.parse_args()
    if args.stage=='heads':fit_heads()
    elif args.stage=='calibrate':calibrate()
    else:
        if not args.lock_sha:raise ValueError('Explicit final lock SHA required')
        test(args.lock_sha)


if __name__=='__main__':main()
