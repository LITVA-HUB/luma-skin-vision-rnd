"""Exact model/plan replay with independent observed-token provenance checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_expert_anchor_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_expert_anchor import ARMS,make_model,objective
from skin_capture_support_verify import verify_plans as original_verify_plans
from skin_capture_support import make_plan,apply_plan
from skin_capture_model import CaptureColor,MODES
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de


def verify_plans(t,seed,records):
    converted=[]
    for r in records:
        rr=dict(r);rr['arm']='baseline' if r['augmentation']=='raw' else 'paired_stratified';converted.append(rr)
    return original_verify_plans(t,seed,converted)


def independent_novelty(train,ev):
    a=np.array([[sum(float(v) for v in image[:,j])/64 for j in range(18)] for image in train['tokens']])
    b=np.array([[sum(float(v) for v in image[:,j])/64 for j in range(18)] for image in ev['tokens']])
    scale=np.maximum(np.std(a,axis=0),1e-6)
    return np.array([min(float(np.linalg.norm((x-y)/scale)) for y in a) for x in b])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[24,72],default=72);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for path,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/path)==h,path
    paths=sorted(OUT.glob('*/result.json'));rr=[json.loads(p.read_bytes()) for p in paths];assert len(paths)==args.fits
    protocols=['mixed'] if args.fits==24 else ['mixed','from_SLR','from_ipod']
    assert {(r['protocol'],r['arm'],r['seed']) for r in rr}=={(p,a,s) for p in protocols for a in ARMS for s in (17,29,43)}
    tr,va=load('train'),load('validation');replays=scalar=coverage=curves=observed=0;gap=0.;initials={}
    for protocol in protocols:
        t=tr if protocol=='mixed' else subset(tr,tr['device']==protocol.removeprefix('from_'))
        for seed in (17,29,43):observed+=verify_plans(t,seed,[r for r in rr if r['protocol']==protocol and r['seed']==seed])
    novelty={}
    for protocol in protocols:
        t=tr if protocol=='mixed' else subset(tr,tr['device']==protocol.removeprefix('from_'))
        ev=va if protocol=='mixed' else subset(va,va['device']!=protocol.removeprefix('from_'))
        novelty[protocol]=independent_novelty(t,ev)
    original_replays=0;anchor_cases=0
    for path,r in zip(paths,rr):
        folder=RUN/path.parent.name
        if r['protocol']=='mixed':t,v,ev=tr,va,va
        else:
            c=r['protocol'].removeprefix('from_');t=subset(tr,tr['device']==c);v=subset(va,va['device']==c);ev=subset(va,va['device']!=c)
            assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
        assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
        ym=t['target'].mean(0).astype(np.float32);ys=t['target'].std(0).astype(np.float32)
        torch.manual_seed(r['seed']);model=make_model(r['mechanism']);assert digest(model.state_dict())==r['initial_sha256']
        backbone=digest({k:v for k,v in model.state_dict().items() if k.startswith(('local.','context.','votes.0.'))});assert backbone==r['backbone_initial_sha256']
        initials.setdefault((r['protocol'],r['seed']),set()).add(backbone);model=model.cuda()
        mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert min(history,key=lambda h:h['selection_patient_mean'])['epoch']==r['best_epoch']
        for tag in ('final','best'):
            assert sha(folder/f'{tag}.pt')==r[f'{tag}_sha256']
            state=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True);assert state['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(state['target_mean'].numpy(),ym);np.testing.assert_array_equal(state['target_std'].numpy(),ys)
            model.load_state_dict(state['state']);p=prediction(model,torch.from_numpy(v['tokens']).cuda(),mean,std)[0]
            np.testing.assert_array_equal(p,np.load(folder/f'{tag}_selection.npz')['prediction']);replays+=1
        assert sha(folder/'evaluation.npz')==r['evaluation_sha256'];saved=dict(np.load(folder/'evaluation.npz'))
        p,g,h=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std)
        for key,array in [('prediction',p),('gate',g),('hypotheses',h),('target',ev['target'])]:np.testing.assert_array_equal(array,saved[key])
        replays+=1;error=np.array([scalar_de(a,b) for a,b in zip(p,ev['target'])]);scalar+=len(error)
        gap=max(gap,float(np.max(abs(error-saved['error']))),abs(float(error.mean())-r['scores']['full']['mean']))
        risk=novelty[r['protocol']];np.testing.assert_allclose(risk,saved['risk'],rtol=1e-12,atol=1e-12)
        order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
        for c in r['scores']['coverage']:
            n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted'];e=error[order[:n]]
            for key,val in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:gap=max(gap,abs(float(val)-c[key]))
            coverage+=1
        curve=np.cumsum(error[order])/np.arange(1,len(error)+1);gap=max(gap,float(np.max(abs(curve-saved['curve']))));curves+=len(error)
        if r['mechanism']=='baseline':
            old_arm='baseline' if r['augmentation']=='raw' else 'paired_stratified'
            old=ROOT/'experiments/runs/skin_capture_support_v1'/f"{r['protocol']}__{old_arm}__s{r['seed']}"
            np.testing.assert_array_equal(p,np.load(old/'evaluation.npz')['prediction'])
            for tag in ('best','final'):
                orig=torch.load(old/f'{tag}.pt',map_location='cpu',weights_only=True)['state']
                now=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True)['state']
                assert digest(orig)==digest(now)
            original_replays+=1
        # Independent arithmetic of the anchor objective on actual evaluation hypotheses.
        # This verifies code, not a teacher-assisted model performance claim.
        n=min(32,len(p));hh=torch.from_numpy(h[:n]).double();yy=torch.from_numpy(ev['target'][:n]).double()
        pp=torch.from_numpy(p[:n]).double();gg=torch.from_numpy(np.log(np.maximum(g[:n],1e-30))).double()
        q=torch.eye(4,dtype=torch.float64)[[MODES.index(str(v)) for v in ev['mode'][:n]]]
        for mech in ('conditional','uniform_anchor'):
            qq=q.numpy() if mech=='conditional' else np.full((n,4),.25)
            primary=sum(sum((float(p[i,j])-float(ev['target'][i,j]))**2 for j in range(3))/3 for i in range(n))/n
            anchor=sum(sum(qq[i,k]*sum((float(h[i,k,j])-float(ev['target'][i,j]))**2 for j in range(3))/3 for k in range(4)) for i in range(n))/n if h.shape[1]==4 else None
            if anchor is None:continue
            logprob=gg.log_softmax(1).numpy();ce=-sum(sum(q[i,k].item()*logprob[i,k] for k in range(4)) for i in range(n))/n
            assert abs(float(objective(mech,pp,gg,hh,yy,q))-(.5*primary+.5*anchor+.1*ce))<1e-9
            anchor_cases+=n
        del model;torch.cuda.empty_cache()
    assert gap<1e-9 and all(len(v)==1 for v in initials.values())
    result={'status':'PASS','fits':len(paths),'exact_color_replay_arrays':replays,'gate_hypothesis_sets':len(paths),
        'independent_scalar_color_cases':scalar,'coverage_rows':coverage,'curve_points':curves,'maximum_gap':gap,
        'observed_patch_tokens_independently_checked':observed,'all_epoch_pair_and_plan_digests_and_counts_replayed':True,
        'all_training_pairs_same_site_and_same_reference_checked':True,'same_camera_selection_and_fit_only_scales_checked':True,
        'same_shared_backbone_initialization':True,'exact_original_baseline_refits':original_replays,'independent_anchor_objective_cases':anchor_cases,'common_input_ranking_independently_checked':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'scripts/skin_capture_support_verify.py']}}
    write(OUT/('audit_mixed.json' if args.fits==24 else 'audit.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
