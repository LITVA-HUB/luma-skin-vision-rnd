"""Exact model/plan replay with independent observed-token provenance checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_capture_support_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_capture_support import ARMS,make_plan,apply_plan
from skin_capture_model import CaptureColor,MODES
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de


def verify_plans(t,seed,records):
    steps=int(np.ceil(len(t['target'])/32));trace=hashlib.sha256();pairs=hashlib.sha256();checked=0
    counts={arm:{'processed_bags':0,'partner_patch_slots':0,'softened_mode_bags':0,'requested_input_augmented_bags':0} for arm in ARMS}
    partner=np.r_[16:32,0:16];all_mode=np.eye(4,dtype=np.float32)[[MODES.index(str(m)) for m in t['mode']]]
    for epoch in range(1,81):
        a,b=pair_indices(t['site'],steps*16,np.random.default_rng(seed*1000+epoch));pairs.update(a.tobytes());pairs.update(b.tobytes())
        rng=np.random.default_rng(seed*100000+epoch)
        for offset in range(0,len(a),16):
            ix=np.r_[a[offset:offset+16],b[offset:offset+16]]
            assert np.array_equal(t['site'][ix],t['site'][ix[partner]])
            np.testing.assert_array_equal(t['target'][ix],t['target'][ix[partner]])
            plan=make_plan(32,64,rng);update_digest(trace,plan);m=all_mode[ix]
            planned=(plan['coin']<plan['fraction'][:,None]) & plan['augment'][:,None]
            fraction=planned.mean(1,keepdims=True).astype(np.float32)
            mixed=m*(1-fraction)+m[partner]*fraction
            for arm in ARMS:
                c=counts[arm];c['processed_bags']+=32
                if arm in ('paired_union','paired_stratified'):c['partner_patch_slots']+=int(planned.sum())
                if arm in ('soft_mode_control','paired_union','paired_stratified'):c['softened_mode_bags']+=int(np.any(mixed!=m,axis=1).sum())
                if arm in ('self_bootstrap','paired_union','paired_stratified'):c['requested_input_augmented_bags']+=int(plan['augment'].sum())
                if offset==0 and epoch in (1,40,80):
                    x=t['tokens'][ix];actual,target,origin,indices=apply_plan(torch.from_numpy(x),torch.from_numpy(m),torch.from_numpy(partner),plan,arm)
                    expected=np.empty_like(x)
                    for i in range(32):
                        for j in range(64):
                            take_partner=bool(planned[i,j]) and arm in ('paired_union','paired_stratified')
                            source=partner[i] if take_partner else i
                            if arm in ('baseline','soft_mode_control','paired_stratified') or not plan['augment'][i]:index=j
                            else:index=int(plan['second'][i,j] if take_partner else plan['first'][i,j])
                            expected[i,j]=x[source,index]
                    np.testing.assert_array_equal(actual.numpy(),expected)
                    np.testing.assert_array_equal(target.numpy(),m if arm in ('baseline','self_bootstrap') else mixed)
                    np.testing.assert_array_equal(target.numpy().sum(1),np.ones(32));checked+=32*64
    for r in records:
        assert r['plan_sha256']==trace.hexdigest() and r['pair_draw_sha256']==pairs.hexdigest()
        assert r['augmentation_counts']==counts[r['arm']],(r['arm'],r['augmentation_counts'],counts[r['arm']])
    return checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fits',type=int,choices=[15,45],default=45);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    for path,h in json.loads((OUT/'source_lock.json').read_bytes())['bindings'].items():assert sha(ROOT/path)==h,path
    paths=sorted(OUT.glob('*/result.json'));rr=[json.loads(p.read_bytes()) for p in paths];assert len(paths)==args.fits
    protocols=['mixed'] if args.fits==15 else ['mixed','from_SLR','from_ipod']
    assert {(r['protocol'],r['arm'],r['seed']) for r in rr}=={(p,a,s) for p in protocols for a in ARMS for s in (17,29,43)}
    tr,va=load('train'),load('validation');replays=scalar=coverage=curves=observed=0;gap=0.;initials={}
    for protocol in protocols:
        t=tr if protocol=='mixed' else subset(tr,tr['device']==protocol.removeprefix('from_'))
        for seed in (17,29,43):observed+=verify_plans(t,seed,[r for r in rr if r['protocol']==protocol and r['seed']==seed])
    for path,r in zip(paths,rr):
        folder=RUN/path.parent.name
        if r['protocol']=='mixed':t,v,ev=tr,va,va
        else:
            c=r['protocol'].removeprefix('from_');t=subset(tr,tr['device']==c);v=subset(va,va['device']==c);ev=subset(va,va['device']!=c)
            assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
        assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
        ym=t['target'].mean(0).astype(np.float32);ys=t['target'].std(0).astype(np.float32)
        torch.manual_seed(r['seed']);model=CaptureColor('mixture');assert digest(model.state_dict())==r['initial_sha256']
        initials.setdefault((r['protocol'],r['seed']),set()).add(r['initial_sha256']);model=model.cuda()
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
        risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1));np.testing.assert_array_equal(risk,saved['risk'])
        order=sorted(range(len(risk)),key=lambda i:(float(risk[i]),i))
        for c in r['scores']['coverage']:
            n=int(np.ceil(len(error)*c['requested_coverage']));assert n==c['accepted'];e=error[order[:n]]
            for key,val in [('mean',e.mean()),('median',np.median(e)),('p95',np.quantile(e,.95)),('above_5_fraction',(e>5).mean()),('above_10_fraction',(e>10).mean())]:gap=max(gap,abs(float(val)-c[key]))
            coverage+=1
        curve=np.cumsum(error[order])/np.arange(1,len(error)+1);gap=max(gap,float(np.max(abs(curve-saved['curve']))));curves+=len(error)
        del model;torch.cuda.empty_cache()
    assert gap<1e-9 and all(len(v)==1 for v in initials.values())
    result={'status':'PASS','fits':len(paths),'exact_color_replay_arrays':replays,'gate_hypothesis_sets':len(paths),
        'independent_scalar_color_cases':scalar,'coverage_rows':coverage,'curve_points':curves,'maximum_gap':gap,
        'observed_patch_tokens_independently_checked':observed,'all_epoch_pair_and_plan_digests_and_counts_replayed':True,
        'all_training_pairs_same_site_and_same_reference_checked':True,'same_camera_selection_and_fit_only_scales_checked':True,
        'same_initialization':True,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/('audit_mixed.json' if args.fits==15 else 'audit.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
