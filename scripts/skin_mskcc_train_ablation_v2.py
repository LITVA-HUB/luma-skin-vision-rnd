"""Source-only pixel networks: standard CNN and matched0/3-step patch voting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_vote_v2 import GlobalColorMLP, VoteAblation
ABLATION_PROTOCOL = ROOT / "docs/research/skin_mskcc_pixel_ablation_v2.md"


def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict(model,x,mean,std,arch):
    model.eval();pred=[];risk=[]
    with torch.no_grad():
        for part in x.split(64):
            if arch=='cnn':
                p=model(part);r=torch.zeros(len(part),device=x.device)
            else:p,r,_=model(part,details=True)
            pred.append((p*std+mean).cpu().numpy());risk.append(r.cpu().numpy())
    return np.concatenate(pred),np.concatenate(risk)


def fit(arch,seed,train,val,epochs):
    run=ROOT/f'experiments/runs/skin_mskcc_pixel_ablation_v2/{arch}_seed{seed}'
    out=ROOT/f'docs/benchmarks/skin_mskcc_pixel_ablation_v2/{arch}_seed{seed}'
    run.mkdir(exist_ok=False,parents=True);out.mkdir(exist_ok=False,parents=True)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    target_mean=train['target'].mean(0).astype(np.float32);target_std=train['target'].std(0).astype(np.float32)
    if arch=='global_mlp': model=GlobalColorMLP()
    else:model=VoteAblation(target_std,local_only=arch.startswith('local_'),steps=0 if arch=='local_mean' else 3)
    init_hash=digest(model.state_dict());model=model.cuda()
    def inputs(data):
        if arch=='cnn':return torch.from_numpy(data['rgb'].transpose(0,3,1,2).copy()).float().cuda()/255
        return torch.from_numpy(data['color' if arch=='global_mlp' else 'tokens']).float().cuda()
    x,vx=inputs(train),inputs(val)
    mean=torch.from_numpy(target_mean).cuda();std=torch.from_numpy(target_std).cuda()
    y=(torch.from_numpy(train['target']).float().cuda()-mean)/std
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=.00001)
    rows=[{k:val[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(val['target']))]
    best=float('inf');best_epoch=None;history=[];start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    def checkpoint(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'target_mean':mean.cpu(),'target_std':std.cpu(),'arch':arch,'seed':seed,'epoch':epoch,
                'protocol_sha256':sha(PROTOCOL),'ablation_protocol_sha256':sha(ABLATION_PROTOCOL)}
    for epoch in range(1,epochs+1):
        model.train();total=0.
        order=np.random.default_rng(seed*1000+epoch).permutation(len(x))
        for offset in range(0,len(order),32):
            ids=order[offset:offset+32]
            indices=torch.tensor(ids,device='cuda');batch=x[indices]
            if arch=='cnn':
                if torch.rand(())<.5:batch=batch.flip(-1)
                if torch.rand(())<.5:batch=batch.flip(-2)
            optimizer.zero_grad(set_to_none=True);pred=model(batch);loss=(pred-y[indices]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids)
        schedule.step();p,risk=predict(model,vx,mean,std,arch)
        error=delta_e00(p,val['target']);metric=summarize(error,rows)
        score=metric['patient_balanced_mean']
        if score<best:
            best=score;best_epoch=epoch;torch.save(checkpoint(epoch),run/'best.pt')
            np.savez(run/'best_validation.npz',prediction=p,target=val['target'],risk=risk,
                     delta_e00=error,image=val['image'],patient=val['patient'])
        history.append({'epoch':epoch,'loss':total/len(x),'validation_mean_delta_e00':metric['mean'],
                        'validation_patient_mean_delta_e00':score,'elapsed_seconds':time.perf_counter()-start})
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf8')
        if epoch%10==0 or epoch==1:
            print(json.dumps({'arch':arch,'seed':seed,'epoch':epoch,'val_mean':metric['mean'],'best':best,'seconds':time.perf_counter()-start}),flush=True)
    torch.save(checkpoint(epochs),run/'final.pt')
    final_pred,final_risk=predict(model,vx,mean,std,arch)
    np.savez(run/'final_validation.npz',prediction=final_pred,target=val['target'],risk=final_risk)
    best_state=torch.load(run/'best.pt',weights_only=True,map_location='cpu');model.load_state_dict(best_state['state'])
    p,risk=predict(model,vx,mean,std,arch);saved=np.load(run/'best_validation.npz')
    replay_gap=float(np.max(np.abs(p-saved['prediction'])))
    if replay_gap!=0:raise ValueError('Checkpoint replay mismatch')
    error=delta_e00(p,val['target'])
    record={'arch':arch,'seed':seed,'epochs':epochs,'best_epoch':best_epoch,
            'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':init_hash,
            'best_sha256':sha(run/'best.pt'),'final_sha256':sha(run/'final.pt'),
            'best_file_bytes':(run/'best.pt').stat().st_size,
            'train_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'elapsed_seconds':time.perf_counter()-start,'best_replay_max_gap':replay_gap,
            'full':summarize(error,rows),'strata':{},'protocol_sha256':sha(PROTOCOL),'ablation_protocol_sha256':sha(ABLATION_PROTOCOL),
            'training_script_sha256':sha(Path(__file__)),'vote_script_sha256':sha(ROOT/'scripts/skin_mskcc_vote_v2.py'),
            'source_validation_only':True,'calibrated_error_head':'NOT FIT; reserved subsequent phase'}
    for key in ['device','image_type']:
        record['strata'][key]={}
        for value in np.unique(val[key]):
            ix=np.flatnonzero(val[key]==value);record['strata'][key][str(value)]=summarize(error[ix],[rows[i] for i in ix])
    if arch!='cnn':
        record['disagreement_coverage']=[];order=np.lexsort((val['image'],risk))
        for coverage in [1,.95,.9,.8,.7,.6]:
            ix=order[:int(np.ceil(coverage*len(error)))];record['disagreement_coverage'].append({'coverage':coverage,**summarize(error[ix],[rows[i] for i in ix])})
        record['mean_intrinsic_score_zero_for_global_mlp']=float(risk.mean())
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'completed':arch,'seed':seed,'best_epoch':best_epoch,'mean':record['full']['mean'],'median':record['full']['median'],'peak_MiB':record['train_peak_allocated_mib']}),flush=True)
    del model,optimizer,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arch',choices=['all','global_mlp','shared_tight','local_mean','local_tight'],default='all')
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]);parser.add_argument('--epochs',type=int,default=80)
    args=parser.parse_args();train,val=load('train'),load('validation')
    for seed in args.seeds:
        for arch in (['global_mlp','shared_tight','local_mean','local_tight'] if args.arch=='all' else [args.arch]):
            fit(arch,seed,train,val,args.epochs)


if __name__=='__main__':main()
