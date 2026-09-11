"""Exact refits plus independent normal equations, densities and skin metrics."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from skin_appearance_inverse import fit_map,predict_map,fit_forward,image_features,forward_means
from skin_appearance_inverse_train import OUT,RUN,PROTOCOL,configs,outputs
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def check_map(model,x,y,alpha):
    np.testing.assert_array_equal(model['xc'],x.mean(0));np.testing.assert_array_equal(model['xs'],np.maximum(x.std(0),1e-6))
    np.testing.assert_array_equal(model['yc'],y.mean(0));np.testing.assert_array_equal(model['ys'],np.maximum(y.std(0),1e-6))
    z=(x-model['xc'])/model['xs'];columns=[np.ones(len(z))]+[z[:,i] for i in range(z.shape[1])]
    if int(model['degree'])==2:
        columns += [z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i,z.shape[1])]
    d=np.column_stack(columns);target=(y-model['yc'])/model['ys'];gradient=d.T@(d@model['weight']-target)
    penalty=alpha*model['weight'].copy();penalty[0]=0
    relative=float(np.linalg.norm(gradient+penalty)/max(1.,np.linalg.norm(d.T@target)))
    assert relative<1e-9,relative
    return relative


def main():
    for name,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/name)==h,name
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==108
    rr=[json.loads(p.read_bytes()) for p in paths]
    assert {(r['protocol'],r['kind'],r['direction'],r['degree'],r['alpha']) for r in rr}=={
        (p,*c) for p in ('mixed','from_SLR','from_ipod') for c in configs()}
    train,val=load('train'),load('validation');replays=scalar=coverage=curves=densities=cost_cases=folds=normal=0
    gap=normal_gap=0.;palettes={};costs={};novelty={}
    with threadpool_limits(1):
        for path,r in zip(paths,rr):
            protocol=r['protocol'];camera=protocol.removeprefix('from_')
            t=train if protocol=='mixed' else subset(train,train['device']==camera)
            v=val if protocol=='mixed' else subset(val,val['device']==camera)
            ev=val if protocol=='mixed' else subset(val,val['device']!=camera)
            assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
            if protocol!='mixed':assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
            kind,direction,degree,alpha=(r[k] for k in ('kind','direction','degree','alpha'))
            x=image_features(t,kind);vx=image_features(v,kind);ex=image_features(ev,kind);folder=RUN/path.parent.name
            for key in ('model','selection','evaluation'):assert sha(folder/f'{key}.npz')==r[key+'_sha256']
            model=dict(np.load(folder/'model.npz'))
            refit=fit_map(x,t['target'],degree,alpha) if direction=='direct' else fit_forward(t,kind,direction,degree,alpha)
            for key in model:np.testing.assert_array_equal(model[key],refit[key])
            if direction=='direct':normal_gap=max(normal_gap,check_map(model,x,t['target'],alpha));normal+=1
            else:
                labels=t['mode'] if direction=='mode' else np.zeros(len(x),dtype=int)
                own_folds=0
                for k,label in enumerate(np.unique(labels)):
                    ix=labels==label;prefix=f'map{k}_';m={key[len(prefix):]:a for key,a in model.items() if key.startswith(prefix)}
                    normal_gap=max(normal_gap,check_map(m,t['target'][ix],x[ix],alpha));normal+=1
                    for person in np.unique(t['patient'][ix]):
                        fit_rows=ix & (t['patient']!=person);held=ix & (t['patient']==person)
                        assert set(t['patient'][fit_rows]).isdisjoint(t['patient'][held])
                        mm=fit_map(t['target'][fit_rows],x[fit_rows],degree,alpha)
                        normal_gap=max(normal_gap,check_map(mm,t['target'][fit_rows],x[fit_rows],alpha));normal+=1
                        np.testing.assert_array_equal(predict_map(mm,t['target'][held]),model['oof_prediction'][held]);own_folds+=1
                    error=(x[ix]-model['oof_prediction'][ix])/model['xs']
                    cov=np.array([[sum(float(row[i])*float(row[j]) for row in error)/len(error) for j in range(x.shape[1])] for i in range(x.shape[1])])
                    cov=.8*cov+.2*np.diag(np.diag(cov))+np.eye(len(cov))*1e-4
                    np.testing.assert_allclose(cov,model['cov'][k],rtol=1e-12,atol=1e-12)
                assert own_folds==r['oof_folds'];folds+=own_folds
                _,first=np.unique(t['site'],return_index=True);np.testing.assert_array_equal(model['palette'],t['target'][first])
                if protocol not in costs:
                    palette=model['palette'];palettes[protocol]=palette
                    costs[protocol]=np.array([[scalar_de(a,b) for b in palette] for a in palette]);cost_cases+=len(palette)**2
                else:np.testing.assert_array_equal(model['palette'],palettes[protocol])
            for inp,tag,data in ((vx,'selection',v),(ex,'evaluation',ev)):
                oo,extra=outputs(model,inp,direction);saved=dict(np.load(folder/f'{tag}.npz'))
                for key,(p,risk) in oo.items():
                    np.testing.assert_array_equal(p,saved[key]);replays+=1
                    if tag=='selection':
                        errors=np.array([scalar_de(a,b) for a,b in zip(p,data['target'])])
                        patient_mean=np.mean([errors[data['patient']==person].mean() for person in np.unique(data['patient'])])
                        gap=max(gap,abs(float(patient_mean)-r['selection'][key]['patient_balanced_mean']))
                if tag=='evaluation':
                    for key,array in extra.items():np.testing.assert_array_equal(array,saved[key])
                    if direction!='direct':
                        n=min(12,len(inp));z=(inp[:n]-model['xc'])/model['xs'];means=forward_means(model)
                        ll=np.stack([np.column_stack([multivariate_normal.logpdf(z,mean=mu,cov=c) for mu in mus]) for mus,c in zip(means,model['cov'])])
                        lm=logsumexp(ll,axis=0)-np.log(len(means));lp=lm-logsumexp(lm,axis=1)[:,None]
                        np.testing.assert_allclose(np.exp(lp),saved['posterior'][:n],rtol=1e-7,atol=1e-10)
                        np.testing.assert_allclose(logsumexp(lm,axis=1)-np.log(means.shape[1]),saved['log_evidence'][:n],rtol=1e-8,atol=1e-8)
                        densities+=ll.size
                        expected=saved['posterior']@costs[protocol];index=np.argmin(expected,axis=1)
                        np.testing.assert_array_equal(saved['medoid'],model['palette'][index])
                        np.testing.assert_allclose(saved['medoid_risk'],expected[np.arange(len(inp)),index],rtol=1e-12,atol=1e-12)
                        independent_mean=np.array([sum(float(w)*scalar_de(p,a) for w,a in zip(weights,model['palette'])) for p,weights in zip(saved['mean'][:n],saved['posterior'][:n])])
                        np.testing.assert_allclose(independent_mean,saved['mean_risk'][:n],rtol=1e-12,atol=1e-12);cost_cases+=n*len(model['palette'])
                    np.testing.assert_array_equal(saved['target'],ev['target'])
                    nk=(protocol,kind)
                    if nk not in novelty:
                        scale=np.maximum(x.std(0),1e-6)
                        novelty[nk]=np.array([min(float(np.linalg.norm((a-b)/scale)) for b in x) for a in ex])
                    np.testing.assert_allclose(saved['common_risk'],novelty[nk],rtol=1e-12,atol=1e-12)
                    for key in oo:
                        errors=np.array([scalar_de(a,b) for a,b in zip(saved[key],ev['target'])]);scalar+=len(errors)
                        gap=max(gap,float(np.max(abs(errors-saved[key+'_error']))))
                        for ranking,metric in r['metrics'][key].items():
                            risk=saved['common_risk'] if ranking=='common' else saved[key+'_risk']
                            order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
                            for c in metric['coverage']:
                                n=int(np.ceil(c['requested_coverage']*len(errors)));assert n==c['accepted'];e=errors[order[:n]]
                                for name,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:gap=max(gap,abs(float(value)-c[name]))
                                coverage+=1
                            curve=np.cumsum(errors[order])/np.arange(1,len(errors)+1)
                            name=key+('_curve' if ranking=='common' else '_posterior_curve');gap=max(gap,float(np.max(abs(curve-saved[name]))));curves+=len(errors)
    assert gap<1e-8
    result={'status':'PASS','fits':108,'exact_prediction_arrays':replays,'independent_scalar_evaluation_cases':scalar,
        'coverage_rows':coverage,'curve_points':curves,'maximum_metric_gap':gap,'normal_equation_checks':normal,'maximum_relative_normal_equation_residual':normal_gap,
        'person_excluded_forward_folds_checked':folds,'independent_gaussian_density_cases':densities,'independent_palette_cost_cases':cost_cases,
        'exact_closed_form_and_oof_refits':108,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
