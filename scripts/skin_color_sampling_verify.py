"""Scalar color geometry, sampling mass identities and exact state audit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_color_sampling_train import OUT,RUN,bindings
from skin_support_curve_train import predict,novelty
from skin_support_curve import SkinRepresentation,patient_roles
from skin_support_curve_verify import check_summary
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write


def independent_probability(tr,arm):
    sites=sorted(set(tr['site']));colors=[];owners=[]
    for s in sites:
        ix=np.flatnonzero(tr['site']==s);assert np.all(tr['target'][ix]==tr['target'][ix[0]])
        assert len(set(tr['patient'][ix]))==1;colors.append(tr['target'][ix[0]]);owners.append(tr['patient'][ix[0]])
    s=len(sites);density=np.array([math.fsum(math.exp(-.5*(scalar_de(a,b)/5)**2) for b in colors)/s for a in colors])
    weight=np.clip(np.median(density)/density,1/3,3);color_mass=.5/s+.5*weight/weight.sum()
    q=np.empty(len(tr['target']));site_q=np.empty_like(q)
    for j,site in enumerate(sites):
        mask=tr['site']==site;n=int(mask.sum());site_q[mask]=1/(s*n)
        if arm=='image':q[mask]=1/len(q)
        elif arm=='site':q[mask]=1/(s*n)
        elif arm=='person_site':q[mask]=1/(len(set(owners))*sum(p==owners[j] for p in owners)*n)
        else:q[mask]=color_mass[j]/n
    correction=site_q/q if arm=='color_ipw' else np.ones(len(q))
    return q,correction,density if arm in ('color','color_ipw') else np.ones(s),site_q,s*s


def indices_for(q,arm,seed,epoch):
    rng=np.random.default_rng(seed*1000+epoch)
    if arm=='image':return rng.integers(0,len(q),(31,32),dtype=np.int64)
    cdf=np.cumsum(q);cdf[-1]=1
    return np.searchsorted(cdf,rng.random((31,32)),side='right').astype(np.int64)


def refit(tr,arm,seed,q,correction,saved):
    torch.manual_seed(seed);np.random.seed(seed);model=SkinRepresentation('baseline').cuda()
    mean=torch.from_numpy(tr['target'].mean(0).astype(np.float32)).cuda();std=torch.from_numpy(tr['target'].std(0).astype(np.float32)).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();rgb=torch.from_numpy(tr['rgb']).cuda().float()/255
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std;weight=torch.tensor(correction,dtype=torch.float32,device='cuda')
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    for epoch in range(1,31):
        model.train()
        for batch in indices_for(q,arm,seed,epoch):
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True);p,g,_=model(x[ix],rgb[ix])
            if arm=='color_ipw':loss=(((p-y[ix]).square().mean(1)+.1*torch.nn.functional.cross_entropy(g,mode[ix],reduction='none'))*weight[ix]).mean()
            else:loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);opt.step()
        scheduler.step()
    assert digest({k:v.cpu() for k,v in model.state_dict().items()})==digest(saved['state'])
    del model,opt,x,rgb,y;torch.cuda.empty_cache()


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    data=load('train');paths=sorted(OUT.glob('*/result.json'));assert len(paths)==15
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    scalar_cases=0;density_cases=0;gap=0.;prob_gap=0.;exact_old=0;refits=0;core={};draws={};common=None
    for path in paths:
        r=json.loads(path.read_bytes());arm=r['arm'];seed=r['seed'];folder=RUN/path.parent.name
        a,b=patient_roles(data['patient'],data['device'],18,seed);tr,ev=subset(data,a),subset(data,b)
        assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
        assert hashlib.sha256(np.packbits(a).tobytes()+np.packbits(b).tobytes()).hexdigest()==r['role_sha256']
        q,w,rho,site_q,cases=independent_probability(tr,arm);density_cases+=cases
        for file,key in [('final.pt','checkpoint_sha256'),('evaluation.npz','evaluation_sha256'),('sampling.npz','sampling_sha256')]:assert sha(folder/file)==r[key]
        z=dict(np.load(folder/'evaluation.npz'));sampling=dict(np.load(folder/'sampling.npz'))
        prob_gap=max(prob_gap,float(np.max(abs(q-sampling['proposal']))),float(np.max(abs(w-sampling['correction']))))
        assert prob_gap<1e-12;np.testing.assert_allclose(rho,sampling['density'],atol=1e-14,rtol=1e-12)
        if arm=='color_ipw':
            assert w.max()<=2;np.testing.assert_allclose(q*w,site_q,rtol=1e-14,atol=1e-14)
            # Actual native-color probes, equality for several per-image loss functions.
            probes=np.column_stack([tr['target'],tr['target']**2,np.ones(len(q))])
            np.testing.assert_allclose((q*w)@probes,site_q@probes,rtol=1e-13,atol=1e-11)
        trace=hashlib.sha256();counts=np.zeros(len(q),np.int64)
        for epoch in range(1,31):
            ix=indices_for(q,arm,seed,epoch);trace.update(ix.tobytes());counts+=np.bincount(ix.ravel(),minlength=len(q))
        assert trace.hexdigest()==r['draw_sha256'];np.testing.assert_array_equal(counts,sampling['counts'])
        assert counts.sum()==r['sampled_images']==29760 and r['optimizer_updates']==930
        draws[seed,arm]=r['draw_sha256']
        for key in ('target','patient','site'):np.testing.assert_array_equal(z[key],ev[key])
        saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
        np.testing.assert_array_equal(saved['target_mean'].numpy(),tr['target'].mean(0).astype(np.float32))
        np.testing.assert_array_equal(saved['target_std'].numpy(),tr['target'].std(0).astype(np.float32))
        torch.manual_seed(seed);model=SkinRepresentation('baseline');assert digest(model.state_dict())==r['initial_sha256']
        core.setdefault(seed,set()).add(r['initial_sha256']);model.load_state_dict(saved['state']);model=model.cuda()
        p=predict(model,torch.from_numpy(ev['tokens']).cuda(),torch.from_numpy(ev['rgb']).cuda().float()/255,saved['target_mean'].cuda(),saved['target_std'].cuda())
        np.testing.assert_array_equal(p,z['prediction'])
        e=np.array([scalar_de(x,y) for x,y in zip(p,ev['target'])]);scalar_cases+=len(e);gap=max(gap,float(np.max(abs(e-z['error']))))
        assert gap<1e-10;gap=max(gap,check_summary(e,ev['patient'],ev['site'],r['scores']['full']))
        site_error=np.mean([e[ev['site']==s].mean() for s in np.unique(ev['site'])]);assert abs(site_error-r['scores']['site_balanced_mean'])<1e-10
        # Independently recompute standardized input-neighbor ranking.
        train=tr['tokens'].mean(1).astype(float);val=ev['tokens'].mean(1).astype(float);scale=np.maximum(train.std(0),1e-6)
        risk=np.array([min(np.linalg.norm((x-y)/scale) for y in train) for x in val]);np.testing.assert_allclose(risk,z['risk'],rtol=1e-12,atol=1e-12)
        if common is not None:np.testing.assert_array_equal(z['risk'],common)
        common=z['risk'];order=np.argsort(risk,kind='stable')
        for c,rec in zip((1,.95,.9,.8,.7,.6),r['scores']['coverage']):
            k=math.ceil(c*len(e));ix=order[:k];assert rec['accepted']==k and rec['requested_coverage']==c
            gap=max(gap,check_summary(e[ix],ev['patient'][ix],ev['site'][ix],rec))
        np.testing.assert_allclose([math.fsum(e[order[:k]])/k for k in range(1,len(e)+1)],z['curve'],rtol=1e-12,atol=1e-12)
        if arm=='image':
            old_folder=ROOT/'experiments/runs/skin_support_curve_v1'/f'n18__baseline__s{seed}'
            old=torch.load(old_folder/'final.pt',weights_only=True,map_location='cpu')
            assert digest(old['state'])==digest(saved['state']);np.testing.assert_array_equal(np.load(old_folder/'evaluation.npz')['prediction'],p);exact_old+=1
        del model;torch.cuda.empty_cache()
        if seed==17 and arm in ('color','color_ipw'):refit(tr,arm,seed,q,w,saved);refits+=1
    for seed in (17,29,43):assert len(core[seed])==1 and draws[seed,'color']==draws[seed,'color_ipw']
    audit={'status':'PASS','fits':15,'exact_prediction_arrays':15,'exact_old_control_states':exact_old,'exact_weighted_and_unweighted_full_refits':refits,
        'scalar_evaluation_color_cases':scalar_cases,'scalar_training_density_pairs':density_cases,'coverage_rows':90,'curve_points':scalar_cases,
        'maximum_color_gap':gap,'maximum_probability_gap':prob_gap,'all930_step_draw_counts_replayed':True,'importance_identity_and_bounded_weight_checked':True,
        'shared_initialization_and_acceptance_checked':True,'only_original_TRAIN_loaded':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'scripts/skin_support_curve_verify.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
