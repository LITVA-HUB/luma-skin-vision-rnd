"""Independent control enumeration, augmented ridge solves and scalar color."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import itertools,json,math
from pathlib import Path
import numpy as np
import torch
from skin_relational_probe_run import OUT,RUN,MODELS,bindings,context_features
from skin_relational_probe import loo_ridge
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def independent_correlation(x,y):
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    def ranks(a):
        _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
        starts=np.cumsum(counts)-counts
        return (starts+(counts-1)/2)[inverse]
    a,b=ranks(x),ranks(y);a-=a.mean();b-=b.mean()
    return float(a@b/math.sqrt((a@a)*(b@b)))


def independent_scaled(x,person):
    result=np.empty_like(x,dtype=float)
    for p in sorted(set(person)):
        fit=x[person!=p];held=person==p;mean=np.array([math.fsum(fit[:,j])/len(fit) for j in range(fit.shape[1])])
        std=np.sqrt(np.mean((fit-mean)**2,0));result[held]=(x[held]-mean)/np.maximum(std,1e-6)
    return result


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    record=json.loads((OUT/'results.json').read_bytes());z=load('train');pairs=dict(np.load(RUN/'pairs.npz'))
    for name,h in record['arrays'].items():assert sha(RUN/name)==h
    rng=np.random.default_rng(69117);pos=[];random=[];near=[];candidate_checks=0;scalar_cases=0;scalar_gap=0.;feature_gap=0.;ridge_gap=0.
    for site in sorted(set(z['site'])):
        ix=np.flatnonzero(z['site']==site)
        for a,b in itertools.combinations(ix,2):
            options=[c for c in range(len(z['target'])) if z['patient'][c]==z['patient'][a] and z['site'][c]!=site
                     and z['mode'][c]==z['mode'][b] and z['image_type'][c]==z['image_type'][b]]
            if not options:continue
            distances=[scalar_de(z['target'][a],z['target'][c]) for c in options];scalar_cases+=len(options)
            pos.append((a,b));random.append((a,int(rng.choice(options))));near.append((a,options[int(np.argmin(distances))]));candidate_checks+=len(options)
            assert len(options)==pairs['candidate_counts'][len(pos)-1]
    for name,expected in [('positive',pos),('uniform',random),('near',near)]:np.testing.assert_array_equal(expected,pairs[name])
    between=[(a,b) for p in sorted(set(z['patient'])) for a,b in itertools.combinations(np.flatnonzero(z['patient']==p),2) if z['site'][a]!=z['site'][b]]
    np.testing.assert_array_equal(between,pairs['between']);pair_names=('positive','uniform','near','between')
    for k in pair_names:
        e=np.array([scalar_de(z['target'][a],z['target'][b]) for a,b in pairs[k]]);scalar_cases+=len(e)
        scalar_gap=max(scalar_gap,float(abs(e-pairs[k+'_truth']).max()))
    assert record['inventory']['positive_pairs']==1421 and len(pos)==1421 and len(between)==18229
    for t,n in record['inventory']['near_threshold_counts'].items():assert int((pairs['near_truth']<=float(t)).sum())==n
    assert record['inventory']['same_site_cross_camera_pairs']==0
    assert all(z['device'][a]==z['device'][b] for a,b in pos)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    raw={'median_rgb':z['median'].astype(float),'color36':independent_scaled(z['color'].astype(float),z['patient']),
         'patch54':independent_scaled(np.concatenate([z['tokens'].mean(1),z['tokens'].max(1),z['tokens'].std(1)],1).astype(float),z['patient'])}
    for m in MODELS:
        suffix='image' if '__image__' in m else 'combined';c,p=context_features(m,z);saved=np.load(RUN/('raw_context_'+suffix+'.npz'))
        np.testing.assert_array_equal(c,saved['context']);np.testing.assert_array_equal(p,saved['prediction'])
        raw['context_'+suffix]=independent_scaled(c,z['patient']);raw['frozen_lab_'+suffix]=p
    fold_checks=0;perturbation_checks=0
    for rep,source in [('ridge3','median'),('ridge36','color')]:
        p=np.empty_like(z['target'])
        for j,identity in enumerate(np.unique(z['patient'])):
            saved=np.load(RUN/f'{rep}__fold{j}.npz');tr=np.flatnonzero(z['patient']!=identity);ev=np.flatnonzero(z['patient']==identity)
            np.testing.assert_array_equal(tr,saved['train_indices']);np.testing.assert_array_equal(ev,saved['held_indices'])
            assert set(z['patient'][tr]).isdisjoint(z['patient'][ev]) and set(z['site'][tr]).isdisjoint(z['site'][ev])
            x=z[source].astype(float);mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-6);center=z['target'][tr].mean(0)
            for k,v in [('mean',mean),('std',std),('center',center)]:np.testing.assert_array_equal(v,saved[k])
            design=np.vstack([(x[tr]-mean)/std,np.eye(x.shape[1])]);target=np.vstack([z['target'][tr]-center,np.zeros((x.shape[1],3))])
            coef=np.linalg.lstsq(design,target,rcond=None)[0]
            ridge_gap=max(ridge_gap,float(abs(coef-saved['coef']).max()))
            np.testing.assert_allclose(coef,saved['coef'],atol=1e-10,rtol=1e-9)
            p[ev]=(x[ev]-mean)/std@coef+center;fold_checks+=1
            if j in (0,23):
                changed=z['target'].copy();changed[ev]+=np.array([10000.,-10000.,20000.])
                predicted,_=loo_ridge(x,changed,z['patient'])
                np.testing.assert_allclose(predicted[ev],p[ev],atol=1e-10,rtol=1e-12);perturbation_checks+=1
        raw[rep]=p
    raw['mean_lab']=np.stack([z['target'][z['patient']!=p].mean(0) for p in z['patient']])
    comparison_checks=0;rank_checks=0;absolute_checks=0
    for r in record['representations']:
        name=r['representation'];s=dict(np.load(RUN/(name+'.npz')));f=raw[name]
        feature_gap=max(feature_gap,float(abs(f-s['features']).max()));np.testing.assert_allclose(f,s['features'],atol=1e-9,rtol=1e-9)
        # Scalar color is checked on persisted features, avoiding changes from an alternative solve's roundoff.
        f=s['features'];dist={};is_lab='lab' in name or name.startswith('ridge')
        for key in pair_names:
            if is_lab:
                dist[key]=np.array([scalar_de(f[a],f[b]) for a,b in pairs[key]]);scalar_cases+=len(dist[key])
                scalar_gap=max(scalar_gap,float(abs(dist[key]-s[key]).max()))
            else:
                dist[key]=np.linalg.norm(f[pairs[key][:,0]]-f[pairs[key][:,1]],axis=1)/math.sqrt(f.shape[1])
                np.testing.assert_allclose(dist[key],s[key],atol=1e-11,rtol=1e-12)
        patient=z['patient'][pairs['positive'][:,0]]
        for c in r['comparisons']:
            control=c['control'];threshold=c['maximum_reference_delta_e00'];mask=np.ones(len(pos),bool) if threshold is None else pairs[control+'_truth']<=threshold
            assert mask.sum()==c['pairs'];comparison_checks+=1
            if not mask.any():continue
            wins=(dist['positive'][mask]<dist[control][mask]).astype(float)+.5*(dist['positive'][mask]==dist[control][mask]);ids=patient[mask]
            assert abs(math.fsum(wins)/len(wins)-c['pooled_preference'])<1e-12
            local=[math.fsum(wins[ids==p])/sum(ids==p) for p in np.unique(ids)]
            assert len(local)==c['people'] and abs(math.fsum(local)/len(local)-c['person_mean_preference'])<1e-12
        corr=independent_correlation(s['between'],pairs['between_truth']);rank_checks+=1
        assert corr is None and r['between_color_spearman'] is None or corr is not None and abs(corr-r['between_color_spearman'])<1e-12
        values=[]
        for p in np.unique(z['patient']):
            ix=z['patient'][pairs['between'][:,0]]==p;v=independent_correlation(s['between'][ix],pairs['between_truth'][ix]);rank_checks+=1
            if v is not None:values.append(v)
        assert len(values)==r['correlation_people']
        if values:assert abs(np.mean(values)-r['between_color_person_mean_spearman'])<1e-12
        else:assert r['between_color_person_mean_spearman'] is None
        if 'absolute_error' in s:
            e=np.array([scalar_de(a,b) for a,b in zip(f,z['target'])]);scalar_cases+=len(e);absolute_checks+=len(e)
            scalar_gap=max(scalar_gap,float(abs(e-s['absolute_error']).max()));metric=r['absolute_color_error']
            for key,value in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('person_mean',np.mean([e[z['patient']==p].mean() for p in np.unique(z['patient'])]))]:assert abs(value-metric[key])<1e-10
        print(json.dumps({'verified':name,'scalar_cases':scalar_cases}),flush=True)
    assert scalar_gap<1e-10 and ridge_gap<1e-10 and feature_gap<1e-9
    assert fold_checks==48 and comparison_checks==100 and rank_checks==250 and perturbation_checks==4
    f=raw['frozen_lab_image'];a=np.arange(32);b=np.arange(32,64)
    anchored=np.mean(f[a,None]-f[None,b]+z['target'][None,b],1)
    expected=f[a]+np.mean(z['target'][b]-f[b],0)
    assert abs(anchored-expected).max()<1e-12
    edges=f[a,None]-f[None,a];potential=f[a]-f[a].mean(0)
    np.testing.assert_allclose(edges,potential[:,None]-potential[None],atol=1e-12,rtol=1e-12)
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    audit={'status':'PASS','candidate_checks':candidate_checks,'control_triplets':1421,'all_between_site_pairs':18229,
           'augmented_least_squares_refits':fold_checks,'held_target_perturbations':perturbation_checks,'context_replays':2,
           'scalar_color_cases':scalar_cases,'absolute_color_cases':absolute_checks,'preference_checks':comparison_checks,
           'rank_checks':rank_checks,'max_scalar_gap':scalar_gap,'max_ridge_coefficient_gap':ridge_gap,'max_feature_replay_gap':feature_gap,
           'train_only':True,'independent_accuracy_changed':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'results.json',OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
