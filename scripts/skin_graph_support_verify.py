"""Exact model/plan replay with independent observed-token provenance checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_graph_support_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_graph_support_model import GraphSupportColor,ARMS
from skin_capture_support_verify import verify_plans as original_verify_plans
from skin_expert_anchor_verify import independent_novelty
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
        rr=dict(r);rr['arm']='baseline' if r['arm'].endswith('_raw') else 'paired_stratified';converted.append(rr)
    return original_verify_plans(t,seed,converted)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[12,36],default=36);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for path,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/path)==h,path
    paths=sorted(OUT.glob('*/result.json'));rr=[json.loads(p.read_bytes()) for p in paths];assert len(paths)==args.fits
    protocols=['mixed'] if args.fits==12 else ['mixed','from_SLR','from_ipod']
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
    branch_checks=0
    for path,r in zip(paths,rr):
        folder=RUN/path.parent.name
        if r['protocol']=='mixed':t,v,ev=tr,va,va
        else:
            c=r['protocol'].removeprefix('from_');t=subset(tr,tr['device']==c);v=subset(va,va['device']==c);ev=subset(va,va['device']!=c)
            assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
        assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
        ym=t['target'].mean(0).astype(np.float32);ys=t['target'].std(0).astype(np.float32)
        torch.manual_seed(r['seed']);model=GraphSupportColor();assert digest(model.state_dict())==r['initial_sha256']
        initials.setdefault((r['protocol'],r['seed']),set()).add(r['initial_sha256']);model=model.cuda()
        mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
        history=json.loads((path.parent/'history.json').read_bytes());assert len(history)==80
        assert min(history,key=lambda h:h['selection_patient_mean'])['epoch']==r['best_epoch']
        for tag in ('final','best'):
            assert sha(folder/f'{tag}.pt')==r[f'{tag}_sha256']
            state=torch.load(folder/f'{tag}.pt',map_location='cpu',weights_only=True);assert state['protocol_sha256']==sha(PROTOCOL)
            np.testing.assert_array_equal(state['target_mean'].numpy(),ym);np.testing.assert_array_equal(state['target_std'].numpy(),ys)
            model.load_state_dict(state['state']);p=prediction(model,torch.from_numpy(v['tokens']).cuda(),mean,std)
            np.testing.assert_array_equal(p,np.load(folder/f'{tag}_selection.npz')['prediction']);replays+=1
        assert sha(folder/'evaluation.npz')==r['evaluation_sha256'];saved=dict(np.load(folder/'evaluation.npz'))
        p=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std)
        for key,array in [('prediction',p),('target',ev['target'])]:np.testing.assert_array_equal(array,saved[key])
        replays+=1;error=np.array([scalar_de(a,b) for a,b in zip(p,ev['target'])]);scalar+=len(error)
        gap=max(gap,float(np.max(abs(error-saved['error']))),abs(float(error.mean())-r['scores']['full']['mean']))
        risk=novelty[r['protocol']];np.testing.assert_allclose(risk,saved['risk'],rtol=1e-12,atol=1e-12)
        order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
        for c in r['scores']['coverage']:
            n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted'];e=error[order[:n]]
            for key,val in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:gap=max(gap,abs(float(val)-c[key]))
            coverage+=1
        curve=np.cumsum(error[order])/np.arange(1,len(error)+1);gap=max(gap,float(np.max(abs(curve-saved['curve']))));curves+=len(error)
        # Runtime check on actual source images: graph input stays original,
        # and the augmented pass executes with steps=0.
        steps=int(np.ceil(len(t['target'])/32));aa,bb=pair_indices(t['site'],steps*16,np.random.default_rng(r['seed']*1000+1))
        ix=np.r_[aa[:16],bb[:16]];raw=torch.from_numpy(t['tokens'][ix]).cuda()
        modes=torch.eye(4,device='cuda')[[MODES.index(str(m)) for m in t['mode'][ix]]]
        partner=torch.tensor(np.r_[16:32,0:16],device='cuda');plan=make_plan(32,64,np.random.default_rng(r['seed']*100000+1))
        support='baseline' if r['arm'].endswith('_raw') else 'paired_stratified'
        mixed,_,_,_=apply_plan(raw,modes,partner,plan,support);seen=[]
        hook=model.local.register_forward_pre_hook(lambda module,args:seen.append((args[0].detach().clone(),model.steps)))
        model.train()
        with torch.no_grad():model(raw,branch=r['arm'].startswith('graph'));model(mixed,branch=False)
        hook.remove();assert len(seen)==2
        torch.testing.assert_close(seen[0][0],raw,rtol=0,atol=0);torch.testing.assert_close(seen[1][0],mixed,rtol=0,atol=0)
        assert seen[0][1]==(3 if r['arm'].startswith('graph') else 0) and seen[1][1]==0
        model.eval()
        with torch.no_grad():torch.testing.assert_close(model(raw,branch=True)[0],model(raw,branch=False)[0],rtol=0,atol=0)
        assert model.inference_parameters()==924932 and not r['graph_at_inference'];branch_checks+=1
        del model;torch.cuda.empty_cache()
    assert gap<1e-9 and all(len(v)==1 for v in initials.values())
    result={'status':'PASS','fits':len(paths),'exact_color_replay_arrays':replays,'original_graph_input_and_disabled_mixed_inference_checks':branch_checks,
        'independent_scalar_color_cases':scalar,'coverage_rows':coverage,'curve_points':curves,'maximum_gap':gap,
        'observed_patch_tokens_independently_checked':observed,'all_epoch_pair_and_plan_digests_and_counts_replayed':True,
        'all_training_pairs_same_site_and_same_reference_checked':True,'same_camera_selection_and_fit_only_scales_checked':True,
        'same_initialization':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py',ROOT/'scripts/skin_capture_support_verify.py',ROOT/'scripts/skin_expert_anchor_verify.py']}}
    write(OUT/('audit_mixed.json' if args.fits==12 else 'audit.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
