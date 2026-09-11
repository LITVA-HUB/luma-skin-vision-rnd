"""Exact conditional-mixture refits, independent densities and native skin error."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import skin_patch_likelihood as core
from skin_patch_likelihood_train import OUT,RUN,PROTOCOL,STRENGTHS,configs
from skin_appearance_inverse import design
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    for name,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/name)==h,name
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==48
    rr=[json.loads(p.read_bytes()) for p in paths]
    assert {(r['protocol'],r['representation'],r['degree'],r['components'],r['seed']) for r in rr}=={
        (p,*c) for p in ('mixed','from_SLR','from_ipod') for c in configs()}
    tr,va=load('train'),load('validation');replays=scalar=coverage=curves=densities=normal_checks=expected_cases=0
    gap=normal_gap=sufficiency_gap=0.;gaussian_models=0
    with threadpool_limits(1):
        for path,r in zip(paths,rr):
            protocol=r['protocol'];c=protocol.removeprefix('from_')
            t=tr if protocol=='mixed' else subset(tr,tr['device']==c)
            v=va if protocol=='mixed' else subset(va,va['device']==c)
            ev=va if protocol=='mixed' else subset(va,va['device']!=c)
            assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
            if protocol!='mixed':assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
            folder=RUN/path.parent.name
            for key in ('model','selection','evaluation'):assert sha(folder/f'{key}.npz')==r[key+'_sha256']
            model=dict(np.load(folder/'model.npz'));captured={};original=core.component_update
            def tracked(d,x,responsibility,alpha):
                captured.update(d=d,x=x,r=responsibility.copy(),alpha=alpha)
                return original(d,x,responsibility,alpha)
            core.component_update=tracked
            try:refit,history=core.fit_patch(t,r['representation'],r['degree'],r['components'],r['seed'])
            finally:core.component_update=original
            for key in model:np.testing.assert_array_equal(model[key],refit[key])
            assert history==json.loads((path.parent/'history.json').read_bytes())
            np.testing.assert_array_equal(model['yc'],t['target'].mean(0));np.testing.assert_array_equal(model['ys'],np.maximum(t['target'].std(0),1e-6))
            raw=t['tokens'].astype(np.float64)[:,:,9:12]
            if r['representation']=='mean':raw=raw.mean(1,keepdims=True)
            np.testing.assert_array_equal(model['xc'],raw.mean((0,1)));np.testing.assert_array_equal(model['xs'],np.maximum(raw.std((0,1)),1e-6))
            _,first=np.unique(t['site'],return_index=True);np.testing.assert_array_equal(model['palette'],t['target'][first])
            d,x,responsibility=captured['d'],captured['x'],captured['r'];patches=x.shape[1]
            for k in range(r['components']):
                mass=responsibility[:,:,k].sum(1)/patches;response=np.sum(responsibility[:,:,k,None]*x,axis=1)/patches
                rhs=d.T@response;gradient=d.T@(mass[:,None]*(d@model['weight'][k]))-rhs
                penalty=model['weight'][k].copy();penalty[0]=0
                relative=np.linalg.norm(gradient+penalty)/max(1.,np.linalg.norm(rhs));normal_gap=max(normal_gap,float(relative));assert relative<1e-9
                residual=(x-(d@model['weight'][k])[:,None]).reshape(-1,3);w=responsibility[:,:,k].reshape(-1)
                moment=residual.T@(residual*w[:,None])/w.sum();cov=.9*moment+.1*np.diag(np.diag(moment))+np.eye(3)*1e-3
                np.testing.assert_allclose(cov,model['cov'][k],rtol=1e-10,atol=1e-11);normal_checks+=1
            np.testing.assert_allclose(model['pi'],(responsibility.sum((0,1))+1)/(responsibility.shape[0]*patches+r['components']),rtol=1e-13)
            selection=dict(np.load(folder/'selection.npz'));saved=dict(np.load(folder/'evaluation.npz'))
            vl=core.likelihood(model,v['tokens']);el=core.likelihood(model,ev['tokens']);cl=core.likelihood(model,ev['tokens'],collapse=True)
            for a,b in ((vl,selection['loglik']),(el,saved['loglik']),(cl,saved['collapse_loglik'])):np.testing.assert_array_equal(a,b)
            selected=min(STRENGTHS,key=lambda s:(r['selection'][str(s)]['patient_balanced_mean'],STRENGTHS.index(s)));assert selected==r['selected_strength']
            for strength in STRENGTHS:
                p,_=core.color_output(vl,model['palette'],strength);np.testing.assert_array_equal(p,selection[f't{strength}']);replays+=1
                e=np.array([scalar_de(a,b) for a,b in zip(p,v['target'])]);value=np.mean([e[v['patient']==person].mean() for person in np.unique(v['patient'])])
                gap=max(gap,abs(float(value)-r['selection'][str(strength)]['patient_balanced_mean']))
            # Independent scipy densities on every patch of four actual images,
            # across sixteen actual measured color hypotheses.
            z=ev['tokens'][:4].astype(np.float64)[:,:,9:12]
            if r['representation']=='mean':z=z.mean(1,keepdims=True)
            z=(z-model['xc'])/model['xs'];phi=design((model['palette'][:16]-model['yc'])/model['ys'],r['degree'])
            mean=np.einsum('aq,kqd->akd',phi,model['weight']);ll=[]
            for a in range(len(phi)):
                terms=np.stack([multivariate_normal.logpdf(z.reshape(-1,3),mean=mean[a,k],cov=model['cov'][k])+np.log(model['pi'][k]) for k in range(r['components'])],axis=-1)
                ll.append(logsumexp(terms,axis=-1).reshape(z.shape[:2]).mean(1));densities+=terms.size
            np.testing.assert_allclose(np.column_stack(ll),el[:4,:len(phi)],rtol=1e-9,atol=1e-8)
            if r['components']==1:
                difference=el-cl;value=float(np.max(abs(difference-difference[:,:1])));sufficiency_gap=max(sufficiency_gap,value);assert value<1e-8;gaussian_models+=1
            tx=t['tokens'].astype(np.float64)[:,:,9:12].mean(1);ex=ev['tokens'].astype(np.float64)[:,:,9:12].mean(1);scale=np.maximum(tx.std(0),1e-6)
            independent_risk=np.array([min(float(np.linalg.norm((a-b)/scale)) for b in tx) for a in ex])
            np.testing.assert_allclose(independent_risk,saved['common_risk'],rtol=1e-12,atol=1e-12)
            for key,ll,strength in [(f't{s}',el,s) for s in STRENGTHS]+[('collapsed',cl,selected)]:
                p,posterior=core.color_output(ll,model['palette'],strength);np.testing.assert_array_equal(p,saved[key]);replays+=1
                e=np.array([scalar_de(a,b) for a,b in zip(p,ev['target'])]);scalar+=len(e);gap=max(gap,float(np.max(abs(e-saved[key+'_error']))))
                expected=np.array([sum(float(w)*scalar_de(color,atom) for w,atom in zip(weights,model['palette'])) for color,weights in zip(p[:12],posterior[:12])])
                gap=max(gap,float(np.max(abs(expected-saved[key+'_risk'][:12]))));expected_cases+=12*len(model['palette'])
                for ranking,m in r['metrics'][key].items():
                    risk=saved['common_risk'] if ranking=='common' else saved[key+'_risk'];order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
                    for row in m['coverage']:
                        n=int(np.ceil(row['requested_coverage']*len(e)));assert n==row['accepted'];accepted=e[order[:n]]
                        for name,value in [('mean',accepted.mean()),('median',np.median(accepted)),('p95',np.quantile(accepted,.95)),('above_5_fraction',(accepted>5).mean()),('above_10_fraction',(accepted>10).mean())]:gap=max(gap,abs(float(value)-row[name]))
                        coverage+=1
                    curve=np.cumsum(e[order])/np.arange(1,len(e)+1);name=key+('_curve' if ranking=='common' else '_posterior_curve')
                    gap=max(gap,float(np.max(abs(curve-saved[name]))));curves+=len(e)
    assert gap<1e-8
    result={'status':'PASS','fits':48,'exact_model_history_refits':48,'exact_color_arrays':replays,'scalar_evaluation_color_cases':scalar,
        'coverage_rows':coverage,'curve_points':curves,'maximum_metric_gap':gap,'weighted_normal_equation_checks':normal_checks,
        'maximum_relative_normal_equation_residual':normal_gap,'independent_gaussian_density_cases':densities,'independent_expected_color_costs':expected_cases,
        'single_gaussian_sufficiency_models':gaussian_models,'maximum_sufficiency_gap':sufficiency_gap,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
