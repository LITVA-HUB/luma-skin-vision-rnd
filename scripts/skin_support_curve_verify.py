"""Independent scalar metrics, source patch provenance and fixed-state replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_support_curve_train import OUT,RUN,PROTOCOL,bindings,predict
from skin_support_curve import SkinRepresentation,patient_roles,pixel_patches
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
from skin_capture_model import MODES


def check_summary(errors,patient,site,rec):
    e=sorted(map(float,errors));n=len(e)
    def quant(q):
        t=(n-1)*q;j=int(t);return e[j]+(e[min(j+1,n-1)]-e[j])*(t-j)
    want={'n':n,'people':len(set(patient)),'sites':len(set(site)),'mean':math.fsum(e)/n,'median':quant(.5),
        'p90':quant(.9),'p95':quant(.95),'above_5_fraction':sum(x>5 for x in e)/n,'above_10_fraction':sum(x>10 for x in e)/n,
        'patient_balanced_mean':float(np.mean([np.mean(np.asarray(errors)[patient==p]) for p in np.unique(patient)]))}
    gap=max(abs(want[k]-rec[k]) for k in want);assert gap<1e-10;return gap


def refit(tr,ev,arm,seed,saved):
    torch.manual_seed(seed);np.random.seed(seed)
    model=SkinRepresentation(arm).cuda();ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    x=torch.from_numpy(tr['tokens']).cuda();rgb=torch.from_numpy(tr['rgb']).cuda().float()/255
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda();y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    for epoch in range(1,31):
        model.train()
        for batch in np.random.default_rng(seed*1000+epoch).integers(0,len(x),size=(31,32),dtype=np.int64):
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True);p,g,_=model(x[ix],rgb[ix])
            loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);opt.step()
        sched.step()
    assert digest({k:v.cpu() for k,v in model.state_dict().items()})==digest(saved['state'])
    del model,opt,x,rgb,y;torch.cuda.empty_cache()


def main():
    lock=json.loads((OUT/'source_lock.json').read_bytes());assert lock['bindings']==bindings()
    data=load('train');records=sorted(OUT.glob('*/result.json'));assert len(records)==27
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    gap=0.;scalar_cases=0;support_cases=0;replays=0;refits=0;patches=0;curves=0
    common={};hold_people=None;selected={};states={}
    for path in records:
        r=json.loads(path.read_bytes());name=path.parent.name;folder=RUN/name;arm=r['arm'];seed=r['seed'];count=r['train_people']
        a,b=patient_roles(data['patient'],data['device'],count,seed);tr,ev=subset(data,a),subset(data,b)
        assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
        assert len(set(tr['patient']))==count and len(set(ev['patient']))==6
        assert hold_people is None or hold_people==set(ev['patient']);hold_people=set(ev['patient'])
        selected[seed,count]=set(tr['patient'])
        assert r['role_sha256']==hashlib.sha256(np.packbits(a).tobytes()+np.packbits(b).tobytes()).hexdigest()
        assert r['train_images']==len(tr['target']) and r['holdout_images']==len(ev['target'])
        assert r['optimizer_updates']==930 and r['sampled_images']==29760
        trace=hashlib.sha256()
        for epoch in range(1,31):trace.update(np.random.default_rng(seed*1000+epoch).integers(0,len(tr['target']),size=(31,32),dtype=np.int64).tobytes())
        assert r['draw_sha256']==trace.hexdigest()
        assert sha(folder/'final.pt')==r['checkpoint_sha256'] and sha(folder/'evaluation.npz')==r['evaluation_sha256']
        saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');z=dict(np.load(folder/'evaluation.npz'))
        for key in ('target','patient','site'):np.testing.assert_array_equal(z[key],ev[key])
        np.testing.assert_array_equal(saved['target_mean'].numpy(),tr['target'].mean(0).astype(np.float32))
        np.testing.assert_array_equal(saved['target_std'].numpy(),tr['target'].std(0).astype(np.float32))
        torch.manual_seed(seed);model=SkinRepresentation(arm)
        assert digest(model.state_dict())==r['initial_sha256'] and digest(model.core.state_dict())==r['core_initial_sha256']
        assert sum(p.numel() for p in model.parameters())==r['parameters']
        states.setdefault(seed,set()).add(r['core_initial_sha256'])
        model.load_state_dict(saved['state']);model=model.cuda()
        p=predict(model,torch.from_numpy(ev['tokens']).cuda(),torch.from_numpy(ev['rgb']).cuda().float()/255,saved['target_mean'].cuda(),saved['target_std'].cuda())
        np.testing.assert_array_equal(p,z['prediction']);replays+=1
        # Independent source indexing, not reshape/transposition replay of the implementation.
        rgb=torch.from_numpy(ev['rgb'][:2]).float();actual=pixel_patches(rgb)
        expected=torch.stack([im[i:i+16,j:j+16].permute(2,0,1) for im in rgb for i in range(0,128,16) for j in range(0,128,16)])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);patches+=len(actual)
        e=np.array([scalar_de(x,y) for x,y in zip(p,ev['target'])]);scalar_cases+=len(e)
        gap=max(gap,float(np.max(abs(e-z['error']))));assert gap<1e-10
        train=tr['tokens'].mean(1).astype(float);val=ev['tokens'].mean(1).astype(float);scale=np.maximum(train.std(0),1e-6)
        risk=np.array([min(np.linalg.norm((x-y)/scale) for y in train) for x in val])
        np.testing.assert_allclose(risk,z['risk'],rtol=1e-12,atol=1e-12)
        if (seed,count) in common:np.testing.assert_array_equal(common[seed,count],z['risk'])
        common[seed,count]=z['risk'];order=np.argsort(risk,kind='stable')
        gap=max(gap,check_summary(e,ev['patient'],ev['site'],r['scores']['full']))
        for c,rec in zip((1,.95,.9,.8,.7,.6),r['scores']['coverage']):
            n=math.ceil(len(e)*c);ix=order[:n];assert rec['accepted']==n and rec['requested_coverage']==c
            gap=max(gap,check_summary(e[ix],ev['patient'][ix],ev['site'][ix],rec))
        curve=np.array([math.fsum(e[order[:k]])/k for k in range(1,len(e)+1)])
        np.testing.assert_allclose(curve,z['curve'],rtol=1e-12,atol=1e-12);curves+=len(curve)
        unique=np.unique(tr['site'],return_index=True)[1];palette=tr['target'][unique]
        support=np.array([min(scalar_de(x,y) for y in palette) for x in ev['target']]);support_cases+=len(palette)*len(support)
        np.testing.assert_allclose(support,z['reference_support_diagnostic'],rtol=1e-10,atol=1e-10)
        assert abs(support.mean()-r['support_diagnostic_mean'])<1e-10
        del model;torch.cuda.empty_cache()
        if seed==17 and count==6:refit(tr,ev,arm,seed,saved);refits+=1
    for seed in (17,29,43):assert selected[seed,6]<selected[seed,12]<selected[seed,18] and len(states[seed])==1
    result={'status':'PASS','fits':27,'exact_prediction_arrays':replays,'exact_full_930_step_refits':refits,'independent_scalar_color_cases':scalar_cases,
        'independent_support_color_distances':support_cases,'coverage_rows':27*6,'curve_points':curves,'source_patch_provenance_instances':patches,
        'maximum_metric_gap':gap,'same_core_initialization_and_draws':True,'nested_person_and_fixed_holdout_checked':True,
        'only_original_TRAIN_loaded':True,'reserved_endpoint_access':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
