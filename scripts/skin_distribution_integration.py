"""Post-hoc fixed antithetic Sobol integration; no fitting or held-out loading."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from luma_skin_vision.color import delta_e00
from skin_distribution_train import OUT,RUN,score
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import subset,write
from skin_mskcc_audit import scalar_de

PROTOCOL=ROOT/'docs/research/skin_distribution_integration_diagnostic_v1.md'


def integrate(m,s,p,z):
    p=p.astype(float);p=p/p.sum(1,keepdims=True);m=m.astype(float);s=s.astype(float)
    center=(m*p[...,None]).sum(1)
    sd=np.sqrt(((s*s+(m-center[:,None])**2)*p[...,None]).sum(1))
    offsets=np.concatenate([np.eye(3)*a for a in (.25,-.25,.5,-.5)])
    candidates=np.concatenate([center[:,None],m,center[:,None]+sd[:,None]*offsets[None]],1)
    risks=np.zeros(candidates.shape[:2]);gap=0.;count=0
    for begin in range(0,len(m),4):
        end=min(begin+4,len(m));c=candidates[begin:end]
        for k in range(m.shape[1]):
            nodes=m[begin:end,k,None]+s[begin:end,k,None]*z[None]
            loss=delta_e00(c[:,:,None],nodes[:,None])
            risks[begin:end]+=loss.mean(-1)*p[begin:end,k,None]
            if begin==0:
                for a in (0,len(c[0])-1):
                    for b in (0,len(z)//2,len(z)-1):
                        gap=max(gap,abs(float(loss[0,a,b])-scalar_de(c[0,a],nodes[0,b])));count+=1
    index=risks.argmin(1)
    return candidates[np.arange(len(m)),index],risks[np.arange(len(m)),index],index,risks,gap,count


def main():
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==36
    files=[Path(__file__),PROTOCOL,OUT/'source_lock.json',ROOT/'scripts/skin_distribution_train.py',
        ROOT/'scripts/skin_mskcc_audit.py',ROOT/'src/luma_skin_vision/color.py']
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'integration_lock.json'
    if not lock.exists():
        write(lock,{'bindings':bindings,'scope':'posthoc integration only, all18density fits'});print('Lock written; run again after checkpoint');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    u=qmc.Sobol(3,scramble=True,seed=71131).random_base2(11);base=ndtri(u)
    nodes={n:np.concatenate([base[:n//2],-base[:n//2]]) for n in (1024,4096)}
    for z in nodes.values():assert np.max(abs(z.mean(0)))<1e-14
    val=load('validation');records=[];gap=0.;scalar=0
    for path in paths:
        r=json.loads(path.read_bytes());arm=r['arm']
        if arm not in ('gaussian','mdn4'):continue
        folder=RUN/path.parent.name;assert sha(folder/'evaluation.npz')==r['evaluation_sha256']
        a=dict(np.load(folder/'evaluation.npz'));m,s,p=a['hypotheses'],a['scales'],a['gate']
        if arm=='gaussian':m=a['point'][:,None];s=s.mean(1,keepdims=True);p=np.ones((len(m),1))
        ev=val if r['protocol']=='mixed' else subset(val,val['device']!=r['protocol'].removeprefix('from_'))
        np.testing.assert_array_equal(ev['target'],a['target'])
        arrays={};reports={};integrals={}
        for n,z in nodes.items():
            pred,risk,idx,risks,localgap,count=integrate(m,s,p,z);scalar+=count;gap=max(gap,localgap)
            rec,e,order=score(pred,risk,ev);independent=np.array([scalar_de(x,y) for x,y in zip(pred,ev['target'])]);scalar+=len(e)
            gap=max(gap,float(np.max(abs(independent-e))))
            for c in rec['coverage']:
                ix=order[:c['accepted']];gap=max(gap,abs(float(independent[ix].mean())-c['mean']))
            reports[str(n)]=rec;integrals[n]=risks
            arrays.update({f'prediction{n}':pred,f'risk{n}':risk,f'index{n}':idx,f'error{n}':e})
        # Reconstruct old finite decision index against the same candidate set.
        # Exact target-free helper is already frozen and independently audited.
        from skin_distribution_model import color_decision
        old=color_decision(m,s,p,3)['decision_index']
        regret=integrals[4096][np.arange(len(m)),old]-arrays['risk4096']
        np.savez(folder/'integration_diagnostic.npz',**arrays,old_order3_regret=regret)
        record={'name':path.parent.name,'protocol':r['protocol'],'arm':arm,'seed':r['seed'],'scores':reports,
            'candidate_choice_change_fraction':float(np.mean(arrays['index1024']!=arrays['index4096'])),
            'mean_delta_e00_between_rules':float(delta_e00(arrays['prediction1024'],arrays['prediction4096']).mean()),
            'maximum_candidate_expected_error_difference':float(np.max(abs(integrals[1024]-integrals[4096]))),
            'order3_mean_quadrature_regret_under4096':float(regret.mean()),
            'source_evaluation_sha256':sha(folder/'evaluation.npz'),'diagnostic_sha256':sha(folder/'integration_diagnostic.npz')}
        records.append(record);print(json.dumps({'completed_integration':record['name'],'mean4096':reports['4096']['full']['mean']}),flush=True)
    assert len(records)==18 and gap<1e-9
    result={'scope':'POSTHOC numeric integration sensitivity; no refit or primary replacement',
        'records':records,'independent_scalar_cases':scalar,'maximum_scalar_gap':gap,
        'normal_node_covariances':{str(n):(z.T@z/len(z)).tolist() for n,z in nodes.items()},
        'antithetic_mean_checked':True,'reserved_endpoint_access':False,'bindings':bindings}
    write(OUT/'integration_diagnostic.json',result)


if __name__=='__main__':main()
