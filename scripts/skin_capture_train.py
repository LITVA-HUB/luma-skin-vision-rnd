"""Frozen full factorial of latent-capture mixture and direct perceptual loss."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import digest
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,rows,write,pair_error
from skin_capture_model import CaptureColor,ARCHES,MODES,delta_e00_squared

OUT=ROOT/'docs/benchmarks/skin_capture_v1'
RUN=ROOT/'experiments/runs/skin_capture_v1'
PROTOCOL=ROOT/'docs/research/skin_capture_protocol_v1.md'
SEEDS=[17,29,43]
OBJECTIVES=['mse','de2']


def prediction(model,x,mean,std):
    model.eval();ps=[];gs=[];hs=[]
    with torch.no_grad():
        for part in x.split(64):
            p,g,h=model(part);ps.append((p*std+mean).cpu().numpy());gs.append(g.softmax(1).cpu().numpy());hs.append((h*std+mean).cpu().numpy())
    return np.concatenate(ps),np.concatenate(gs),np.concatenate(hs)


def fit(arch,objective,seed,train,val,protocol_name,evaluation=None):
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:
        assert set(train['device']).isdisjoint(evaluation['device']) and set(val['device']).isdisjoint(evaluation['device'])
    name=f'{protocol_name}__{arch}_{objective}__s{seed}';folder=RUN/name;out=OUT/name
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    x=torch.from_numpy(train['tokens']).cuda();vx=torch.from_numpy(val['tokens']).cuda()
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda();native=torch.from_numpy(train['target']).cuda()
    y=(native.float()-mean)/std;mode=torch.tensor([MODES.index(str(v)) for v in train['mode']],device='cuda')
    model=CaptureColor(arch);initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arch':arch,'objective':objective,'seed':seed,'epoch':epoch,
                'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    for epoch in range(1,81):
        model.train();a,b=pair_indices(train['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[];norms=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);pred,logits,_=model(x[ix])
            color=(pred-y[ix]).square().mean() if objective=='mse' else delta_e00_squared(pred*std+mean,native[ix]).mean()/25
            loss=color if arch=='plain' else color+.1*torch.nn.functional.cross_entropy(logits,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();p,_,_=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0],target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    assert np.array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h=prediction(model,ex,mean,std);e=delta_e00(p,ev['target']);m=np.array([MODES.index(str(v)) for v in ev['mode']])
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,gate=g,hypotheses=h,patient=ev['patient'],site=ev['site'])
    record={'arch':arch,'objective':objective,'seed':seed,'protocol':protocol_name,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'paired_repeatability':pair_error(p,ev),'mode_accuracy':float((g.argmax(1)==m).mean()),
        'mean_gate_entropy':float(-(g*np.log(np.maximum(g,1e-12))).sum(1).mean()),
        'mean_hypothesis_rms_delta_e76':float(np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1)).mean()),
        'initial_state_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'stored_parameters':sum(p.numel() for p in model.parameters()),'gate_parameters':sum(p.numel() for p in model.gate.parameters()),
        'peak_train_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.perf_counter()-start,
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),'model_script_sha256':sha(ROOT/'scripts/skin_capture_model.py'),
        'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False}
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type','mode']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'mode_accuracy':record['mode_accuracy'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['mixed','transfer']);args=parser.parse_args()
    train,val=load('train'),load('validation');OUT.mkdir(parents=True,exist_ok=True)
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_capture_model.py',ROOT/'scripts/skin_pair_invariance.py',ROOT/'scripts/skin_pair_train.py',ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'src/luma_skin_vision/color.py']
    files+=list((ROOT/'data/processed/skin_mskcc_pixels_v1').glob('*.npz'))
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if args.stage=='mixed':
        if lock.exists():raise ValueError('Already started; inspect authoritative progress before resuming')
        write(lock,{'bindings':bindings,'arches':ARCHES,'objectives':OBJECTIVES,'seeds':SEEDS,'reserved_data_access':False})
        protocols=[('mixed',train,val,None)]
    else:
        assert json.loads(lock.read_bytes())['bindings']==bindings
        assert len(list(OUT.glob('mixed__*/result.json')))==18
        protocols=[('from_'+c,subset(train,train['device']==c),subset(val,val['device']==c),subset(val,val['device']!=c)) for c in ['SLR','ipod']]
    for protocol,tr,va,ev in protocols:
        for seed in SEEDS:
            for objective in OBJECTIVES:
                for arch in ARCHES:fit(arch,objective,seed,tr,va,protocol,evaluation=ev)


if __name__=='__main__':main()
