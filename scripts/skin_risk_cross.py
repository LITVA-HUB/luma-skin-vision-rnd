"""Fixed two-model crossing of real skin color and Gaussian risk."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from luma_skin_vision.color import delta_e00
from skin_distribution_train import OUT as SOURCE,RUN as SOURCE_RUN,score
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write

OUT=ROOT/'docs/benchmarks/skin_risk_cross_v1'
RUN=ROOT/'experiments/runs/skin_risk_cross_v1'
PROTOCOL=ROOT/'docs/research/skin_risk_cross_protocol_v1.md'


def expected_error(prediction,mean,scale,nodes):
    prediction=np.asarray(prediction,dtype=float);mean=np.asarray(mean,dtype=float);scale=np.asarray(scale,dtype=float)
    if prediction.shape!=mean.shape or mean.shape!=scale.shape:raise ValueError('Unaligned predictions')
    out=[]
    for begin in range(0,len(mean),8):
        end=min(begin+8,len(mean))
        samples=mean[begin:end,None]+scale[begin:end,None]*nodes[None]
        out.extend(delta_e00(prediction[begin:end,None],samples).mean(1))
    return np.asarray(out)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[p for p in SOURCE.glob('*/result.json') if json.loads(p.read_bytes())['arm'] in ('mse_mode','gaussian')]
    assert len(paths)==18
    files=[PROTOCOL,Path(__file__),ROOT/'tests/test_skin_risk_cross.py',ROOT/'scripts/skin_distribution_train.py',
        ROOT/'scripts/skin_mskcc_audit.py',ROOT/'src/luma_skin_vision/color.py',SOURCE/'source_lock.json']+paths
    for p in paths:files.append(SOURCE_RUN/p.parent.name/'evaluation.npz')
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'lock.json'
    if not lock.exists():write(lock,{'bindings':bindings});print('Frozen crossing; checkpoint before second invocation');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    for p,h in json.loads((SOURCE/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/p)==h
    RUN.mkdir(parents=True,exist_ok=True);val=load('validation')
    u=qmc.Sobol(3,scramble=True,seed=71131).random_base2(11);z=ndtri(u);z=np.concatenate([z,-z])
    results=[];gap=0.;count=0;cases=0;coverages=0;curve_points=0
    for protocol in ('mixed','from_SLR','from_ipod'):
        ev=val if protocol=='mixed' else subset(val,val['device']!=protocol.removeprefix('from_'))
        for seed,other in ((17,29),(29,43),(43,17)):
            def source(arm,sd):
                name=f'{protocol}__{arm}__s{sd}';rec=json.loads((SOURCE/name/'result.json').read_bytes());f=SOURCE_RUN/name/'evaluation.npz'
                assert sha(f)==rec['evaluation_sha256'];a=dict(np.load(f))
                np.testing.assert_array_equal(a['target'],ev['target']);np.testing.assert_array_equal(a['patient'],ev['patient']);np.testing.assert_array_equal(a['site'],ev['site'])
                return a
            a=source('mse_mode',seed);b=source('mse_mode',other);g=source('gaussian',seed)
            p=a['point'].astype(float);q=b['point'].astype(float);m=g['point'].astype(float);s=g['scales'].mean(1).astype(float)
            pair=(p+q)/2;cross=(p+m)/2;dis=delta_e00(p,m)
            ordinary_exp=expected_error(p,m,s,z);ordinary_cov=expected_error(p,p,s,z)
            cross_exp=expected_error(cross,m,s,z);cross_cov=expected_error(cross,cross,s,z);own=expected_error(m,m,s,z)
            variants={
                'ordinary_dispersion':(p,a['mean_risk'],1,929297),
                'ordinary_gaussian_expected':(p,ordinary_exp,2,929297+935453),
                'ordinary_gaussian_covariance':(p,ordinary_cov,2,929297+935453),
                'ordinary_cross_disagreement':(p,dis,2,929297+935453),
                'ordinary_pair_disagreement':(pair,delta_e00(p,q),2,2*929297),
                'ordinary_pair_dispersion':(pair,(a['mean_risk']+b['mean_risk'])/2,2,2*929297),
                'mixed_pair_expected':(cross,cross_exp,2,929297+935453),
                'mixed_pair_covariance':(cross,cross_cov,2,929297+935453),
                'mixed_pair_disagreement':(cross,dis,2,929297+935453),
                'gaussian_expected':(m,own,1,935453)}
            arrays={'target':ev['target'],'patient':ev['patient'],'site':ev['site']}
            # Independent integral, first/last image, each Gaussian risk type.
            for pred,center,risks in ((p,m,ordinary_exp),(p,p,ordinary_cov),(cross,m,cross_exp),(cross,cross,cross_cov),(m,m,own)):
                for i in (0,len(p)-1):
                    manual=np.mean([scalar_de(pred[i],center[i]+s[i]*n) for n in z]);count+=len(z)
                    gap=max(gap,abs(float(manual)-risks[i]))
            for method,(pred,risk,models,parameters) in variants.items():
                scores,e,order=score(pred,risk,ev);ind=np.array([scalar_de(x,y) for x,y in zip(pred,ev['target'])]);cases+=len(e)
                gap=max(gap,float(np.max(abs(ind-e))))
                for c in scores['coverage']:
                    selected=ind[order[:c['accepted']]]
                    for key,v in [('mean',selected.mean()),('median',np.median(selected)),('p95',np.quantile(selected,.95)),('above_10_fraction',(selected>10).mean())]:gap=max(gap,abs(float(v)-c[key]))
                    coverages+=1
                curve=np.cumsum(e[order])/np.arange(1,len(e)+1);curve_points+=len(e)
                arrays.update({method+'_prediction':pred,method+'_risk':risk,method+'_error':e,method+'_curve':curve})
                results.append({'protocol':protocol,'seed':seed,'ordinary_partner_seed':other,'method':method,
                    'models_required':models,'active_parameters':parameters,'scores':scores})
            file=RUN/f'{protocol}__s{seed}.npz';np.savez(file,**arrays)
            print(json.dumps({'crossed':protocol,'seed':seed}),flush=True)
    assert gap<1e-9 and len(results)==90
    write(OUT/'results.json',{'scope':'POSTHOC SOURCE; no fitting, no calibrated C+', 'records':results,
        'independent_scalar_integral_cases':count,'independent_scalar_color_cases':cases,'coverage_rows':coverages,
        'curve_points':curve_points,'maximum_gap':gap,'bindings':bindings,
        'artifacts':{str(p.relative_to(ROOT)):sha(p) for p in RUN.glob('*.npz')}})


if __name__=='__main__':main()
