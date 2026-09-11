"""Frozen real-image material decoder versus matched tangent controls."""
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
from skin_capture_model import MODES
from skin_material_model import MaterialImage,ARMS
from skin_material_prior import PRIOR

OUT=ROOT/'docs/benchmarks/skin_material_image_v1'
RUN=ROOT/'experiments/runs/skin_material_image_v1'
PROTOCOL=ROOT/'docs/research/skin_material_image_protocol_v1.md'
SEEDS=[17,29,43]


def prediction(model,x,mean,std):
    model.eval();ps=[];gs=[];hs=[];rs=[]
    with torch.no_grad():
        for part in x.split(64):
            p,g,h,r=model(part)
            ps.append((p*std+mean).cpu().numpy());gs.append(g.softmax(1).cpu().numpy())
            hs.append((h*std+mean).cpu().numpy());rs.append((r*std).cpu().numpy())
    return tuple(np.concatenate(a) for a in (ps,gs,hs,rs))


def fit(arm,seed,train,val,protocol,evaluation=None):
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        rec=json.loads((out/'result.json').read_bytes())
        assert rec['best_sha256']==sha(folder/'best.pt') and rec['final_sha256']==sha(folder/'final.pt')
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:assert set(train['device']).isdisjoint(evaluation['device']) and set(val['device']).isdisjoint(evaluation['device'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    x=torch.from_numpy(train['tokens']).cuda();vx=torch.from_numpy(val['tokens']).cuda()
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    y=(torch.from_numpy(train['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(v)) for v in train['mode']],device='cuda')
    model=MaterialImage(arm,dict(np.load(PRIOR)),ym,ys);initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
            'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    for epoch in range(1,81):
        model.train();a,b=pair_indices(train['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[];norms=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);p,g,h,r=model(x[ix])
            loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if arm.endswith('_residual'):loss=loss+.01*r.square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();p,_,_,_=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0],target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    assert np.array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h,r=prediction(model,ex,mean,std);e=delta_e00(p,ev['target'])
    risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,risk=risk,gate=g,hypotheses=h,
        residual_or_free_coordinates=r,patient=ev['patient'],site=ev['site'])
    order=np.argsort(risk,kind='stable')
    curve=np.cumsum(e[order])/np.arange(1,len(e)+1)
    np.savez(folder/'risk_curve.npz',coverage=np.arange(1,len(e)+1)/len(e),mean_error=curve)
    record={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'paired_repeatability':pair_error(p,ev),
        'initial_state_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'stored_parameters':sum(q.numel() for q in model.parameters()),'active_parameters':model.active_parameters(),
        'model_bytes':(folder/'best.pt').stat().st_size,'peak_train_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'elapsed_seconds':time.perf_counter()-start,
        'mean_native_residual_or_free_rms':float(np.sqrt(np.mean(np.sum(r*r,axis=2),axis=1)).mean()),
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),'model_script_sha256':sha(ROOT/'scripts/skin_material_model.py'),
        'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False,
        'uncalibrated_disagreement_coverage':[]}
    for coverage in [1.,.95,.9,.8,.7,.6]:
        n=int(np.ceil(len(e)*coverage));ix=order[:n]
        record['uncalibrated_disagreement_coverage'].append({'requested_coverage':coverage,'accepted':n,**summarize(e[ix],rows(subset(ev,ix)))})
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type','mode']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','mixed','transfer']);stage=parser.parse_args().stage
    prior_receipt=json.loads((OUT/'prior_receipt.json').read_bytes())
    for p,h in prior_receipt['bindings'].items():assert sha(ROOT/p)==h,p
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_material_model.py',ROOT/'tests/test_skin_material_model.py',
        ROOT/'scripts/skin_capture_model.py',ROOT/'scripts/skin_pair_invariance.py',ROOT/'scripts/skin_pair_train.py',
        ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'src/luma_skin_vision/color.py',OUT/'prior_receipt.json',PRIOR,
        ROOT/'scripts/skin_mskcc_data.py',ROOT/'scripts/skin_mskcc_summary_pilot.py',ROOT/'scripts/skin_mskcc_train_pixels.py']
    files += [ROOT/'data/processed/skin_mskcc_pixels_v1'/f'{role}.npz' for role in ('train','validation')]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'arms':ARMS,'seeds':SEEDS,'reserved_data_access':False})
        print('Source/code/prior lock prepared; no image model fitting.');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    tr,va=load('train'),load('validation')
    if stage=='mixed':protocols=[('mixed',tr,va,None)]
    else:
        assert len(list(OUT.glob('mixed__*/result.json')))==15
        protocols=[('from_'+c,subset(tr,tr['device']==c),subset(va,va['device']==c),subset(va,va['device']!=c)) for c in ('SLR','ipod')]
    for protocol,t,v,e in protocols:
        for seed in SEEDS:
            for arm in ARMS:fit(arm,seed,t,v,protocol,evaluation=e)


if __name__=='__main__':main()
