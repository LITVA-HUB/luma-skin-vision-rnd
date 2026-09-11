"""Independent weighted augmented solves and scalar skin metrics."""
import json,math
from pathlib import Path
import numpy as np
from skin_local_reference_run import OUT,RUN,PREVIOUS,bindings
from skin_local_reference import predict_bank
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def weighted_solve(x,y,q,w):
    mass=math.fsum(w);xm=w@x/mass;ym=w@y/mass
    design=np.vstack([np.sqrt(w)[:,None]*(x-xm),np.eye(x.shape[1])])
    target=np.vstack([np.sqrt(w)[:,None]*(y-ym),np.zeros((x.shape[1],3))])
    coef=np.linalg.lstsq(design,target,rcond=None)[0]
    return ym+(q-xm)@coef


def verify_bank(x,y,q,saved,pred):
    x=x.astype(float);q=q.astype(float);mean=x.mean(0);std=np.maximum(x.std(0),1e-6);center=y.mean(0)
    for k,v in [('mean',mean),('std',std),('center',center)]:np.testing.assert_array_equal(v,saved[k])
    xx=(x-mean)/std;qq=(q-mean)/std;n=len(x);gap=0.;wgap=0.;color_cases=0
    base=qq@saved['coef']+center;bank=xx@saved['coef']+center
    np.testing.assert_array_equal(bank,saved['bank_prediction']);np.testing.assert_array_equal(base,pred['global_ridge'])
    np.testing.assert_array_equal(np.broadcast_to(center,base.shape),pred['global_mean'])
    uniform_gap=0.
    for j,query in enumerate(qq):
        d2=np.mean((xx-query)**2,1)
        d=np.array([scalar_de(base[j],p) for p in bank]);color_cases+=len(d)
        for affinity,logs in [('appearance',-.5*d2),('color',-.5*(d/5)**2)]:
            a=np.array([math.exp(v-max(logs)) for v in logs]);w=n*a/math.fsum(a)
            wgap=max(wgap,float(abs(w-saved['weights_'+affinity][j]).max()))
            np.testing.assert_allclose(w,saved['weights_'+affinity][j],atol=1e-10,rtol=1e-10)
            estimated=w@y/math.fsum(w);local=weighted_solve(xx,y,query,w)
            gap=max(gap,float(abs(estimated-pred[affinity+'_mean'][j]).max()),float(abs(local-pred[affinity+'_affine'][j]).max()))
        uniform=weighted_solve(xx,y,query,np.ones(n));uniform_gap=max(uniform_gap,float(abs(uniform-base[j]).max()))
        assert abs(math.sqrt(d2.min())-saved['risk'][j])<1e-12
    assert gap<1e-8 and wgap<1e-10 and uniform_gap<1e-8
    return gap,wgap,color_cases,uniform_gap


def verify_metrics(errors,person,site,record):
    mean=math.fsum(errors)/len(errors)
    expected={'mean':mean,'median':np.median(errors),'p95':np.quantile(errors,.95),'fraction_above10':np.mean(errors>10),
        'person_mean':np.mean([np.mean(errors[person==p]) for p in np.unique(person)]),
        'site_mean':np.mean([np.mean(errors[site==s]) for s in np.unique(site)])}
    for key,v in expected.items():assert abs(v-record[key])<1e-10,(key,v,record[key])


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    r=json.loads((OUT/'results.json').read_bytes());z=load('train');p=dict(np.load(RUN/'predictions.npz'))
    for file,h in r['arrays'].items():assert sha(RUN/file)==h
    for key in ('target','patient','site'):np.testing.assert_array_equal(p[key],z[key])
    np.testing.assert_array_equal(p['global_ridge'],np.load(PREVIOUS/'ridge36.npz')['features'])
    np.testing.assert_array_equal(p['global_mean'],np.load(PREVIOUS/'mean_lab.npz')['features'])
    names=[m['method'] for m in r['methods']];color_cases=0;gap=0.;wgap=0.;uniform=0.;perturbations=0
    for j,identity in enumerate(np.unique(z['patient'])):
        s=dict(np.load(RUN/f'fold{j}.npz'));tr=np.flatnonzero(z['patient']!=identity);ev=np.flatnonzero(z['patient']==identity)
        np.testing.assert_array_equal(s['train_indices'],tr);np.testing.assert_array_equal(s['held_indices'],ev)
        assert set(z['patient'][tr]).isdisjoint(z['patient'][ev]) and set(z['site'][tr]).isdisjoint(z['site'][ev])
        g,w,c,u=verify_bank(z['color'][tr],z['target'][tr],z['color'][ev],s,{name:p[name][ev] for name in names})
        gap=max(gap,g);wgap=max(wgap,w);uniform=max(uniform,u);color_cases+=c
        for affinity in ('appearance','color'):
            mass=s['weights_'+affinity]/len(tr);photo=1/(mass**2).sum(1)
            pm=np.column_stack([mass[:,z['patient'][tr]==a].sum(1) for a in np.unique(z['patient'][tr])]);pe=1/(pm**2).sum(1)
            expected={'photo_effective_min':photo.min(),'photo_effective_median':np.median(photo),'people_effective_min':pe.min(),'people_effective_median':np.median(pe),'largest_person_mass':pm.max()}
            for key,v in expected.items():assert abs(v-r['folds'][j]['support'][affinity][key])<1e-10
        if j in (0,23):
            changed=z['target'].copy();changed[ev]+=1e5;got=predict_bank(z['color'][tr],changed[tr],z['color'][ev])
            for name in names:np.testing.assert_array_equal(got['predictions'][name],p[name][ev])
            perturbations+=1
    scalar_gap=0.;coverage=0
    baseline=np.array([scalar_de(a,b) for a,b in zip(p['global_ridge'],z['target'])]);color_cases+=len(baseline)
    for m in r['methods']:
        name=m['method'];s=dict(np.load(RUN/(name+'.npz')));e=np.array([scalar_de(a,b) for a,b in zip(p[name],z['target'])]);color_cases+=len(e)
        scalar_gap=max(scalar_gap,float(abs(e-s['error']).max()));verify_metrics(e,z['patient'],z['site'],m['full'])
        order=np.argsort(p['risk'],kind='stable');np.testing.assert_array_equal(order,s['order'])
        curve=np.array([math.fsum(e[order[:i]])/i for i in range(1,len(e)+1)])
        np.testing.assert_allclose(curve,s['curve'],atol=1e-11,rtol=1e-12)
        assert abs(curve.mean()-m['discrete_mean_risk'])<1e-10
        for c,row in zip((1.,.95,.9,.8,.7,.6),m['coverage']):
            n=math.ceil(c*len(e));ix=order[:n];assert row['coverage']==c and row['accepted']==n
            verify_metrics(e[ix],z['patient'][ix],z['site'][ix],row);coverage+=1
        pd=np.array([(e-baseline)[z['patient']==a].mean() for a in np.unique(z['patient'])])
        np.testing.assert_allclose(pd,s['person_differences'],atol=1e-11,rtol=1e-10)
        assert int((pd<0).sum())==m['person_improved_vs_ridge']
        assert abs(pd.mean()-m['person_delta_mean'])<1e-10 and abs((e-baseline).mean()-m['mean_delta_vs_ridge'])<1e-10
    assert scalar_gap<1e-10 and coverage==36 and json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    audit={'status':'PASS','excluded_person_folds':24,'weighted_augmented_solves':1932,'uniform_limit_checks':966,'held_target_perturbations':perturbations,
        'exact_previous_control_replays':2,'scalar_color_cases':color_cases,'coverage_rows':coverage,'max_scalar_gap':scalar_gap,
        'max_local_prediction_gap':gap,'max_weight_gap':wgap,'max_uniform_prediction_gap':uniform,'train_only':True,'independent_accuracy_changed':False,
        'bindings':{str(a.relative_to(ROOT)):sha(a) for a in [Path(__file__),OUT/'results.json',OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
