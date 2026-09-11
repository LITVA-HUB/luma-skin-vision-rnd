"""Independent NumPy heads/augmented solves, scalar colors and deterministic refits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_neural_reference_train import OUT,RUN,PROTOCOL,bindings,prepare_core,risk_order
from skin_neural_reference import ColorAdapter,DeployedColor
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def numpy_head(x,state):
    h=x.astype(np.float64)
    for i in (0,2,4):
        h=h@state[f'adapter.features.{i}.weight'].numpy().astype(np.float64).T+state[f'adapter.features.{i}.bias'].numpy()
        h=h/(1+np.exp(-h)) if i!=4 else np.tanh(h)
    gate=np.tanh(h@state['adapter.output.weight'].numpy().astype(np.float64).T+state['adapter.output.bias'].numpy())
    return h,gate


def independent_local(q,b,y,keep,kind):
    b=b[keep];y=y[keep];logs=-.5*np.mean((q-b)**2,axis=1);w=np.exp(logs-logs.max());w/=sum(w)
    ym=w@y
    if kind=='mean':return ym
    xm=w@b
    design=np.vstack([(b-xm)*np.sqrt(w[:,None]),.1*np.eye(b.shape[1])])
    target=np.vstack([(y-ym)*np.sqrt(w[:,None]),np.zeros((b.shape[1],3))])
    coef=np.linalg.lstsq(design,target,rcond=None)[0]
    return ym+(q-xm)@coef


def numeric_metrics(e,people,sites):
    return {'mean':float(np.mean(e)),'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),
            'fraction_above10':float(np.mean(e>10)),
            'person_mean':float(np.mean([np.mean(e[people==p]) for p in np.unique(people)])),
            'site_mean':float(np.mean([np.mean(e[sites==s]) for s in np.unique(sites)]))}


def compare_metrics(a,b):
    for k in a:np.testing.assert_allclose(a[k],b[k],atol=2e-12,rtol=2e-12)


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    source=json.loads((OUT/'results.json').read_bytes());assert source['status']=='COMPLETE'
    assert source['bindings']==bindings()==json.loads((OUT/'source_lock.json').read_bytes())['bindings']
    tr,va=load('train'),load('validation');protocols={p:(t,d) for p,t,d in banks(tr,va)}
    counts={'core_feature_replays':0,'head_numpy_replays':0,'weighted_augmented_solves':0,'excluded_person_perturbations':0,
            'scalar_color_cases':0,'coverage_rows':0,'exact_prediction_arrays':0,'full_affine_refits':0}
    gaps={'numpy_native_prediction':0.,'numpy_bank_embedding':0.,'scalar_color':0.,'deployment_prediction':0.}
    for record in source['records']:
        protocol,seed=record['protocol'],record['seed'];name=f'{protocol}__s{seed}';folder=RUN/name;t,domains=protocols[protocol]
        assert sha(folder/'training_cache.npz')==record['training_cache_sha256']
        cache=dict(np.load(folder/'training_cache.npz'))
        core,tokens,vtokens,mean,std,p,vp,x,vx,y,fm,fs,_=prepare_core(protocol,seed,t,va)
        for key,tensor in [('features',x),('base',p),('target',y),('validation_features',vx),('validation_base',vp)]:
            np.testing.assert_array_equal(cache[key],tensor.cpu().numpy())
        assert digest(core.state_dict())==record['core_state_sha256'];counts['core_feature_replays']+=1
        groups=np.unique(t['patient'],return_inverse=True)[1];np.testing.assert_array_equal(groups,cache['groups'])
        draws=np.random.default_rng(seed).integers(0,len(x),size=(300,32),dtype=np.int64)
        np.testing.assert_array_equal(draws,cache['indices']);assert hashlib.sha256(draws.tobytes()).hexdigest()==record['draw_sha256']
        exclusions=int(sum((groups[ix,None]==groups[None]).sum() for ix in draws))
        risk=risk_order(t,va);base=(vp*std+mean).cpu().numpy()
        residual=(y-p).cpu().numpy()
        for arm,row in record['arms'].items():
            if arm=='base':pred=base
            else:
                file=folder/(arm+'.pt');assert sha(file)==row['checkpoint_sha256'];saved=torch.load(file,map_location='cpu',weights_only=True)
                state=saved['state'];deployed=DeployedColor(arm,saved['bank_size']).cuda().eval();deployed.load_state_dict(state)
                assert sum(v.numel() for v in deployed.parameters())==row['parameters']==1123092<=1129297
                assert row['trainable_parameters']==193795 and row['optimizer_steps']==300
                assert file.stat().st_size==row['checkpoint_bytes']
                assert row['bank_scalars']==19*saved['bank_size'] and row['fixed_scale_scalars']==1102
                assert row['excluded_query_person_bank_entries']==exclusions
                assert digest(deployed.core.state_dict())==record['core_state_sha256']
                with torch.no_grad():
                    vc=torch.from_numpy(va['color']).cuda()
                    pred=torch.cat([deployed(a,b) for a,b in zip(vtokens.split(64),vc.split(64))]).cpu().numpy()
                z,gate=numpy_head(cache['validation_features'],state);bz,_=numpy_head(cache['features'],state)
                if arm=='residual':correction=gate
                else:
                    np.testing.assert_allclose(bz,state['bank_z'].numpy(),atol=2e-5,rtol=2e-5)
                    gaps['numpy_bank_embedding']=max(gaps['numpy_bank_embedding'],float(abs(bz-state['bank_z'].numpy()).max()))
                    np.testing.assert_array_equal(residual,state['bank_residual'].numpy())
                    correction=np.array([independent_local(q,bz,residual,np.ones(len(bz),bool),arm) for q in z])*gate
                    if arm=='affine':counts['weighted_augmented_solves']+=len(z)
                    # Perturb ALL references of the query's person, then rebuild the correction.
                    for ix in (0,len(x)-1):
                        keep=groups!=groups[ix];q,_=numpy_head(cache['features'][ix:ix+1],state)
                        altered=residual.copy();altered[~keep]+=100000
                        a=independent_local(q[0],bz,residual,keep,arm);b=independent_local(q[0],bz,altered,keep,arm)
                        np.testing.assert_array_equal(a,b);counts['excluded_person_perturbations']+=1
                independent=(cache['validation_base']+correction)*std.cpu().numpy()+mean.cpu().numpy()
                gap=float(abs(independent-pred).max());gaps['numpy_native_prediction']=max(gaps['numpy_native_prediction'],gap)
                np.testing.assert_allclose(independent,pred,atol=3e-4,rtol=1e-5);counts['head_numpy_replays']+=1
                # Full optimization replay, independently reconstructing support masks and draws.
                if arm=='affine' and seed==17:
                    torch.manual_seed(seed);head=ColorAdapter(arm).cuda();assert digest(head.state_dict())==row['initial_head_sha256']
                    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
                    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,300,eta_min=.00001)
                    for batch in draws:
                        ix=torch.from_numpy(batch).cuda();keep=torch.from_numpy(groups[batch,None]!=groups[None]).cuda()
                        opt.zero_grad(set_to_none=True);delta=head(x[ix],x,y-p,keep)
                        loss=(p[ix]+delta-y[ix]).square().mean();loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True);opt.step();scheduler.step()
                    assert digest(head.state_dict())==row['final_head_sha256'];counts['full_affine_refits']+=1
                    del head,opt
                del deployed
            for domain,d in row['evaluations'].items():
                file=folder/f'{arm}__{domain}.npz';assert sha(file)==d['array_sha256'];array=dict(np.load(file));mask=domains[domain]
                gap=float(abs(array['prediction']-pred[mask]).max());gaps['deployment_prediction']=max(gaps['deployment_prediction'],gap)
                np.testing.assert_allclose(array['prediction'],pred[mask],atol=2e-5,rtol=0)
                counts['exact_prediction_arrays']+=int(np.array_equal(array['prediction'],pred[mask]))
                for key in ('target','patient','site'):np.testing.assert_array_equal(array[key],va[key][mask])
                np.testing.assert_array_equal(array['risk'],risk[mask]);order=np.argsort(risk[mask],kind='stable')
                np.testing.assert_array_equal(array['order'],order)
                errors=np.array([scalar_de(a,b) for a,b in zip(array['prediction'],array['target'])])
                gap=float(abs(errors-array['error']).max());gaps['scalar_color']=max(gaps['scalar_color'],gap)
                np.testing.assert_allclose(errors,array['error'],atol=2e-12,rtol=2e-12);counts['scalar_color_cases']+=len(errors)
                compare_metrics(numeric_metrics(errors,array['patient'],array['site']),d['full'])
                np.testing.assert_allclose(np.cumsum(errors[order])/np.arange(1,len(errors)+1),array['curve'],atol=2e-12,rtol=2e-12)
                for row_c in d['coverage']:
                    n=int(np.ceil(row_c['coverage']*len(errors)));assert n==row_c['accepted'];ix=order[:n]
                    compare_metrics(numeric_metrics(errors[ix],array['patient'][ix],array['site'][ix]),row_c);counts['coverage_rows']+=1
        print(json.dumps({'verified':name}),flush=True)
    assert counts['head_numpy_replays']==27 and counts['full_affine_refits']==3 and counts['coverage_rows']==360
    bound={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',OUT/'results.json',ROOT/'scripts/skin_mskcc_audit.py']}
    write(OUT/'audit.json',{'status':'PASS','counts':counts,'max_gaps':gaps,'bindings':bound,'scope':'Exploratory source evidence only; no independent TEST/CAL accessed'})
    print(json.dumps({'status':'PASS','counts':counts,'max_gaps':gaps}),flush=True)


if __name__=='__main__':main()
