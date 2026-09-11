"""Matched removal versus conditional expert native skin color supervision."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_capture_model import CaptureColor,MODES
from skin_capture_support import make_plan,apply_plan
from skin_expert_anchor import ARMS,make_model,objective
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,rows,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_summary_pilot import summarize
from skin_distribution_train import score
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_expert_anchor_v1'
RUN=ROOT/'experiments/runs/skin_expert_anchor_v1'
PROTOCOL=ROOT/'docs/research/skin_expert_anchor_protocol_v1.md'
SEEDS=(17,29,43)


def update_digest(h,plan):
    for k in sorted(plan):h.update(k.encode());h.update(np.ascontiguousarray(plan[k]).tobytes())


def prediction(model,x,mean,std):
    model.eval();out=[[],[],[]]
    with torch.no_grad():
        for part in x.split(64):
            p,g,h=model(part)
            for bucket,value in zip(out,(p*std+mean,g.softmax(1),h*std+mean)):bucket.append(value.cpu().numpy())
    return tuple(np.concatenate(v) for v in out)


def common_risk(train,evaluation):
    a=train['tokens'].astype(np.float64).mean(1)
    b=evaluation['tokens'].astype(np.float64).mean(1)
    scale=np.maximum(a.std(0),1e-6)
    return np.sqrt(np.min(np.sum(((b[:,None]-a[None])/scale)**2,axis=2),axis=1))


def fit(arm,seed,tr,va,protocol,evaluation=None):
    mechanism,augmentation=arm.rsplit('_',1)
    support_arm='baseline' if augmentation=='raw' else 'paired_stratified'
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'best.pt')==r['best_sha256'] and sha(folder/'final.pt')==r['final_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient'])
    if evaluation is not None:assert set(tr['device']).isdisjoint(evaluation['device'])
    for site in np.unique(tr['site']):assert np.all(tr['target'][tr['site']==site]==tr['target'][tr['site']==site][0])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();vx=torch.from_numpy(va['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    soft_mode=torch.nn.functional.one_hot(mode,len(MODES)).float();partner=torch.tensor(np.r_[16:32,0:16],device='cuda')
    model=make_model(mechanism);initial=digest(model.state_dict());backbone_initial=digest({k:v for k,v in model.state_dict().items() if k.startswith(('local.','context.','votes.0.'))});model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
            'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');trace=hashlib.sha256();pair_trace=hashlib.sha256()
    counts={'processed_bags':0,'partner_patch_slots':0,'softened_mode_bags':0,'requested_input_augmented_bags':0}
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,81):
        model.train();a,b=pair_indices(tr['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        pair_trace.update(a.tobytes());pair_trace.update(b.tobytes());rng=np.random.default_rng(seed*100000+epoch)
        for offset in range(0,len(a),16):
            indices=np.r_[a[offset:offset+16],b[offset:offset+16]]
            assert np.array_equal(tr['site'][indices[:16]],tr['site'][indices[16:]])
            ix=torch.tensor(indices,device='cuda');plan=make_plan(32,64,rng);update_digest(trace,plan)
            inp,target,origin,index=apply_plan(x[ix],soft_mode[ix],partner,plan,support_arm)
            counts['processed_bags']+=32;counts['partner_patch_slots']+=int(origin.sum())
            counts['softened_mode_bags']+=int((target!=soft_mode[ix]).any(1).sum())
            if augmentation=='paired':counts['requested_input_augmented_bags']+=int(plan['augment'].sum())
            opt.zero_grad(set_to_none=True);p,g,h=model(inp)
            if mechanism=='baseline' and augmentation=='raw':
                loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            else:loss=objective(mechanism,p,g,h,y[ix],target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite support loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std)[0]
        metric=summarize(delta_e00(p,va['target']),rows(va));value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_patient_mean':value,'selection_mean':metric['mean']})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0])
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=va if evaluation is None else evaluation
    p,g,h=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std)
    risk=common_risk(tr,ev);scores,e,order=score(p,risk,ev)
    np.savez(folder/'evaluation.npz',prediction=p,gate=g,hypotheses=h,risk=risk,error=e,curve=np.cumsum(e[order])/np.arange(1,len(e)+1),
        target=ev['target'],patient=ev['patient'],site=ev['site'])
    r={'arm':arm,'mechanism':mechanism,'augmentation':augmentation,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'scores':scores,'initial_sha256':initial,'backbone_initial_sha256':backbone_initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'plan_sha256':trace.hexdigest(),'pair_draw_sha256':pair_trace.hexdigest(),
        'augmentation_counts':counts,'parameters':sum(q.numel() for q in model.parameters()),'checkpoint_bytes':(folder/'best.pt').stat().st_size,
        'fit_peak_allocated_mib':peak,'fit_seconds':seconds,'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),
        'evaluation_people':len(set(ev['patient'])),'train_images':len(x),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(tr['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'augmentation_code_sha256':sha(ROOT/'scripts/skin_capture_support.py'),
        'source_exploratory_only':True,'risk_calibrated':False,'reserved_endpoint_access':False}
    write(out/'result.json',r);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':scores['full']['mean'],'at80':scores['coverage'][3]['mean'],'seconds':seconds}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','mixed','transfer']);stage=parser.parse_args().stage
    OUT.mkdir(parents=True,exist_ok=True)
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_capture_support.py',ROOT/'scripts/skin_expert_anchor.py',ROOT/'tests/test_skin_expert_anchor.py']
    files += [ROOT/'scripts'/p for p in ('skin_capture_model.py','skin_pair_invariance.py','skin_pair_train.py','skin_mskcc_data.py','skin_mskcc_pixels.py',
        'skin_mskcc_train_pixels.py','skin_mskcc_summary_pilot.py','skin_distribution_train.py')]
    files += [ROOT/'src/luma_skin_vision/color.py']+[ROOT/'data/processed/skin_mskcc_pixels_v1'/f'{role}.npz' for role in ('train','validation')]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'arms':ARMS,'seeds':SEEDS})
        print('Expert protocol/code frozen; no image model fitting');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    tr,va=load('train'),load('validation')
    protocols=[('mixed',tr,va,None)] if stage=='mixed' else [
        ('from_'+c,subset(tr,tr['device']==c),subset(va,va['device']==c),subset(va,va['device']!=c)) for c in ('SLR','ipod')]
    for protocol,t,v,e in protocols:
        for seed in SEEDS:
            for arm in ARMS:fit(arm,seed,t,v,protocol,e)


if __name__=='__main__':main()
