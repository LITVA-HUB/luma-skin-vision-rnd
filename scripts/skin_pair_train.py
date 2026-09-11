"""Exploratory source fits only; does not import or open reserved skin data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import digest
from skin_pair_invariance import ARMS,PairedColor,pair_indices,nuisance_transform,vicreg_loss

OUT=ROOT/'docs/benchmarks/skin_pair_v1'
RUN=ROOT/'experiments/runs/skin_pair_v1'
PROTOCOL=ROOT/'docs/research/skin_pair_invariance_protocol_v1.md'
SEEDS=[17,29,43]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def subset(data,mask):return {k:v[mask] for k,v in data.items()}


def rows(data):
    return [{k:data[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(data['target']))]


def prediction(model,x,mean,std):
    model.eval()
    with torch.no_grad():return np.concatenate([(model(t)*std+mean).cpu().numpy() for t in x.split(64)])


def pair_error(pred,data):
    values=[]
    for site in np.unique(data['site']):
        ids=np.flatnonzero(data['site']==site)
        values.extend(float(delta_e00(pred[a],pred[b])) for a,b in itertools.combinations(ids,2))
    return {'pairs':len(values),'mean_delta_e00_between_captures':float(np.mean(values))}


def fit(arm,seed,train,val,protocol_name,evaluation=None):
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:
        assert set(train['device']).isdisjoint(evaluation['device'])
        assert set(val['device']).isdisjoint(evaluation['device'])
    name=f'{protocol_name}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    center,scale,q,spectrum=nuisance_transform(train['tokens'],train['site'],3 if arm=='quotient3' else 0)
    if arm not in ['standardized','quotient3']:center=np.zeros(18);scale=np.ones(18);q=np.eye(18)
    def inputs(data):
        return torch.from_numpy((((data['tokens'].astype(np.float64)-center)/scale)@q).astype(np.float32)).cuda()
    x,vx=inputs(train),inputs(val);mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    y=(torch.from_numpy(train['target']).float().cuda()-mean)/std
    model=PairedColor(ys,arm);initial=digest(model.backbone.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
                'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),
                'center':torch.from_numpy(center),'scale':torch.from_numpy(scale),'quotient':torch.from_numpy(q),
                'protocol_sha256':sha(PROTOCOL)}
    for epoch in range(1,81):
        model.train();a,b=pair_indices(train['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        for offset in range(0,len(a),16):
            ids=np.r_[a[offset:offset+16],b[offset:offset+16]];ix=torch.tensor(ids,device='cuda')
            opt.zero_grad(set_to_none=True);pred,context=model(x[ix],features=True)
            supervised=(pred-y[ix]).square().mean();loss=supervised
            if arm in ['output','vicreg']:loss=loss+.5*(pred[:16]-pred[16:]).square().mean()
            if arm=='vicreg':
                z=model.projection(context);loss=loss+.01*vicreg_loss(z[:16],z[16:])
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt')
            np.savez(folder/'best_selection.npz',prediction=p,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std),target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    p=prediction(model,vx,mean,std);assert np.array_equal(p,np.load(folder/'best_selection.npz')['prediction'])
    # Camera-held-out endpoint is evaluated only after selection has finished.
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else inputs(ev)
    p=prediction(model,ex,mean,std);e=delta_e00(p,ev['target'])
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,patient=ev['patient'],site=ev['site'])
    record={'arm':arm,'seed':seed,'protocol':protocol_name,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'paired_repeatability':pair_error(p,ev),
        'initial_backbone_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'inference_parameters':sum(p.numel() for p in model.backbone.parameters()),'training_parameters':sum(p.numel() for p in model.parameters()),
        'peak_train_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.perf_counter()-start,
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'nuisance_eigenvalues':spectrum.tolist(),'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),
        'model_script_sha256':sha(ROOT/'scripts/skin_pair_invariance.py'),'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False}
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'patient_mean':record['full']['patient_balanced_mean'],'pair_gap':record['paired_repeatability']['mean_delta_e00_between_captures'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['mixed','transfer']);args=parser.parse_args()
    train,val=load('train'),load('validation');OUT.mkdir(parents=True,exist_ok=True)
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_pair_invariance.py',ROOT/'scripts/skin_mskcc_vote.py',ROOT/'scripts/skin_mskcc_pixels.py']
    files+=list((ROOT/'data/processed/skin_mskcc_pixels_v1').glob('*.npz'))
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files}
    if args.stage=='mixed':
        lock=OUT/'source_lock.json'
        if lock.exists():raise ValueError('Source experiment already started; inspect completed jobs before any resume')
        write(lock,{'bindings':bindings,'arms':ARMS,'seeds':SEEDS,'reserved_data_access':False})
        for seed in SEEDS:
            for arm in ARMS:fit(arm,seed,train,val,'mixed')
    else:
        assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings
        scores={arm:float(np.mean([json.loads((OUT/f'mixed__{arm}__s{seed}'/'result.json').read_bytes())['full']['patient_balanced_mean'] for seed in SEEDS])) for arm in ARMS}
        challenger=min((a for a in ARMS if a!='raw'),key=lambda a:(scores[a],a))
        choice=OUT/'transfer_choice.json'
        if choice.exists():raise ValueError('Transfer already started; inspect progress before resume')
        write(choice,{'source_patient_means':scores,'challenger':challenger,'camera_screen_fits_not_started':True})
        for camera in ['SLR','ipod']:
            tr=subset(train,train['device']==camera);va=subset(val,val['device']==camera);ev=subset(val,val['device']!=camera)
            for seed in SEEDS:
                for arm in ['raw',challenger]:fit(arm,seed,tr,va,'from_'+camera,evaluation=ev)


if __name__=='__main__':main()
