"""Frozen-core, equal-capacity local-reference skin color screen, final fits only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_neural_reference import ColorAdapter,DeployedColor,core_features,ARMS
from skin_capture_model import CaptureColor
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_pair_train import write,subset
from skin_local_reference_transfer import banks
from skin_local_reference_run import summarize
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_neural_reference_v1'
RUN=ROOT/'experiments/runs/skin_neural_reference_v1'
SOURCE=ROOT/'experiments/runs/skin_capture_v1'
PROTOCOL=ROOT/'docs/research/skin_neural_reference_protocol_v1.md'
SEEDS=(17,29,43)
STEPS=300


def source_file(protocol,seed):return SOURCE/f'{protocol}__mixture_mse__s{seed}'/'best.pt'


def bindings():
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_neural_reference.py',ROOT/'tests/test_skin_neural_reference.py',
           ROOT/'scripts/skin_capture_model.py',ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'scripts/skin_mskcc_data.py',
           ROOT/'scripts/skin_pair_train.py',ROOT/'scripts/skin_local_reference_transfer.py',ROOT/'scripts/skin_local_reference_run.py',
           ROOT/'scripts/skin_mskcc_train_pixels.py',ROOT/'src/luma_skin_vision/color.py',
           ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',ROOT/'data/processed/skin_mskcc_pixels_v1/validation.npz']
    for protocol in ('mixed','from_SLR','from_ipod'):
        for seed in SEEDS:
            files.extend([source_file(protocol,seed),ROOT/f'docs/benchmarks/skin_capture_v1/{protocol}__mixture_mse__s{seed}/result.json'])
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def features(core,tokens):
    parts=[];contexts=[]
    with torch.no_grad():
        for x in tokens.split(64):
            p,c=core_features(core,x);parts.append(p);contexts.append(c)
    return torch.cat(parts),torch.cat(contexts)


def prepare_core(protocol,seed,tr,va):
    original=json.loads((ROOT/f'docs/benchmarks/skin_capture_v1/{protocol}__mixture_mse__s{seed}/result.json').read_bytes())
    file=source_file(protocol,seed);assert sha(file)==original['best_sha256']
    saved=torch.load(file,map_location='cpu',weights_only=True)
    core=CaptureColor('mixture');core.load_state_dict(saved['state']);core=core.cuda().eval().requires_grad_(False)
    mean=saved['target_mean'].cuda();std=saved['target_std'].cuda()
    np.testing.assert_array_equal(mean.cpu().numpy(),tr['target'].mean(0).astype(np.float32))
    np.testing.assert_array_equal(std.cpu().numpy(),tr['target'].std(0).astype(np.float32))
    tokens=torch.from_numpy(tr['tokens']).cuda();vtokens=torch.from_numpy(va['tokens']).cuda()
    p,c=features(core,tokens);vp,vc=features(core,vtokens)
    f=torch.cat([c,torch.from_numpy(tr['color']).cuda()],1)
    vf=torch.cat([vc,torch.from_numpy(va['color']).cuda()],1)
    fm=f.mean(0);fs=f.std(0,correction=0).clamp_min(1e-6)
    x=torch.cat([(f-fm)/fs,p],1);vx=torch.cat([(vf-fm)/fs,vp],1)
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    # Reproduce the historical core on exactly its original evaluation people.
    mask=np.ones(len(va['target']),bool) if protocol=='mixed' else va['device']!=protocol.removeprefix('from_')
    historical=np.load(file.parent/'evaluation.npz')['prediction']
    replay=(vp*std+mean).cpu().numpy()[mask]
    gap=float(np.max(abs(replay-historical)));assert gap<2e-5
    return core,tokens,vtokens,mean,std,p,vp,x,vx,y,fm,fs,gap


def risk_order(tr,va):
    scale=np.maximum(tr['color'].astype(np.float64).std(0),1e-6)
    return np.sqrt(np.min(np.mean(((va['color'][:,None]-tr['color'][None])/scale)**2,axis=2),axis=1))


def score_arrays(pred,va,risk,domains,folder,arm):
    results={}
    for domain,mask in domains.items():
        ev=subset(va,mask);p=pred[mask];r=risk[mask];e=delta_e00(p,ev['target'])
        order=np.argsort(r,kind='stable');curve=np.cumsum(e[order])/np.arange(1,len(e)+1)
        coverage=[]
        for c in (1.,.95,.9,.8,.7,.6):
            n=int(np.ceil(c*len(e)));ix=order[:n]
            coverage.append({'coverage':c,'accepted':n,**summarize(e[ix],ev['patient'][ix],ev['site'][ix])})
        file=folder/f'{arm}__{domain}.npz'
        np.savez(file,prediction=p,target=ev['target'],patient=ev['patient'],site=ev['site'],error=e,risk=r,order=order,curve=curve)
        results[domain]={'full':summarize(e,ev['patient'],ev['site']),'coverage':coverage,'array_sha256':sha(file),
                         'people':len(set(ev['patient'])),'images':len(e),'cameras':np.unique(ev['device']).tolist()}
    return results


def deployment(core,head,mean,std,fm,fs,x,residual):
    size=0 if head.arm=='residual' else len(x)
    deployed=DeployedColor(head.arm,size).cuda().eval()
    deployed.core.load_state_dict(core.state_dict());deployed.adapter.load_state_dict(head.state_dict())
    with torch.no_grad():
        for name,value in [('feature_mean',fm),('feature_std',fs),('target_mean',mean),('target_std',std)]:getattr(deployed,name).copy_(value)
        if size:
            deployed.bank_z.copy_(head.features(x));deployed.bank_residual.copy_(residual)
    return deployed


def profile(deployed,tokens,color):
    # Inputs already on device; decode/feature construction excluded.
    with torch.inference_mode():
        for _ in range(10):deployed(tokens,color)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
        allocated_before=torch.cuda.memory_allocated()
        for _ in range(50):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();answer=deployed(tokens,color);end.record();end.synchronize();times.append(start.elapsed_time(end))
        assert torch.isfinite(answer).all()
    peak=torch.cuda.max_memory_allocated()
    return {'batch1_median_ms':float(np.median(times)),'batch1_p95_ms':float(np.quantile(times,.95)),
            'inference_process_peak_allocated_mib':peak/2**20,'inference_incremental_peak_mib':(peak-allocated_before)/2**20,
            'scope':'Prepared tokens/descriptors resident on GPU; includes core and cached reference solve, excludes JPEG/feature extraction; process includes training caches',
            'warmups':10,'repetitions':50}


def run_bank(protocol,seed,tr,va,domains):
    name=f'{protocol}__s{seed}';out=OUT/name;folder=RUN/name
    if (out/'result.json').exists():
        record=json.loads((out/'result.json').read_bytes())
        for arm,a in record['arms'].items():
            if arm!='base':assert sha(folder/(arm+'.pt'))==a['checkpoint_sha256']
            for domain,v in a['evaluations'].items():assert sha(folder/f'{arm}__{domain}.npz')==v['array_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return record
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    core,tokens,vtokens,mean,std,p,vp,x,vx,y,fm,fs,replay_gap=prepare_core(protocol,seed,tr,va)
    core_hash=digest(core.state_dict());risk=risk_order(tr,va);residual=y-p
    groups=torch.tensor(np.unique(tr['patient'],return_inverse=True)[1],device='cuda')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(x),size=(STEPS,32),dtype=np.int64)
    trace=hashlib.sha256(indices.tobytes()).hexdigest()
    np.savez(folder/'training_cache.npz',features=x.cpu().numpy(),base=p.cpu().numpy(),target=y.cpu().numpy(),
             groups=groups.cpu().numpy(),validation_features=vx.cpu().numpy(),validation_base=vp.cpu().numpy(),indices=indices)
    all_allowed=torch.ones(len(vx),len(x),dtype=torch.bool,device='cuda')
    record={'protocol':protocol,'seed':seed,'train_people':len(set(tr['patient'])),'train_images':len(x),'train_cameras':np.unique(tr['device']).tolist(),
            'core_checkpoint_sha256':sha(source_file(protocol,seed)),'core_replay_max_gap':replay_gap,'core_state_sha256':core_hash,
            'draw_sha256':trace,'training_cache_sha256':sha(folder/'training_cache.npz'),'arms':{},
            'source_exploratory_only':True,'core_previously_selected_on_source_validation':True,'adapter_validation_selection':False,
            'reserved_test_or_calibration_loaded':False,'risk_calibrated':False,'device':torch.cuda.get_device_name()}
    record['arms']['base']={'parameters':929297,'evaluations':score_arrays((vp*std+mean).cpu().numpy(),va,risk,domains,folder,'base')}
    for arm in ARMS:
        torch.manual_seed(seed);head=ColorAdapter(arm).cuda();initial=digest(head.state_dict())
        with torch.no_grad():assert torch.equal(head(vx,x,residual,all_allowed),torch.zeros_like(vp))
        opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00001)
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats();losses=[];exclusions=0
        for batch in indices:
            ix=torch.from_numpy(batch).cuda();allowed=groups[ix,None]!=groups[None]
            assert bool(allowed.any(1).all());exclusions+=int((~allowed).sum())
            opt.zero_grad(set_to_none=True);correction=head(x[ix],x,residual,allowed)
            loss=(p[ix]+correction-y[ix]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite adapter loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();scheduler.step();losses.append(float(loss.detach()))
        seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
        assert digest(core.state_dict())==core_hash and all(v.grad is None for v in core.parameters())
        head.eval()
        with torch.no_grad():
            cp=torch.cat([head(z,x,residual,torch.ones(len(z),len(x),dtype=torch.bool,device='cuda')) for z in vx.split(64)])
            prediction=((vp+cp)*std+mean).cpu().numpy()
        deployed=deployment(core,head,mean,std,fm,fs,x,residual)
        vcolor=torch.from_numpy(va['color']).cuda()
        with torch.no_grad():
            replay=torch.cat([deployed(a,b) for a,b in zip(vtokens.split(64),vcolor.split(64))]).cpu().numpy()
        gap=float(np.max(abs(replay-prediction)));assert gap<2e-5
        file=folder/(arm+'.pt')
        torch.save({'state':{k:v.cpu().clone() for k,v in deployed.state_dict().items()},'arm':arm,'bank_size':len(deployed.bank_z),
                    'protocol_sha256':sha(PROTOCOL),'source_checkpoint_sha256':sha(source_file(protocol,seed))},file)
        metrics=profile(deployed,vtokens[:1],vcolor[:1]);parameters=sum(v.numel() for v in deployed.parameters())
        assert parameters==1123092 and parameters<=1129297
        row={'parameters':parameters,'trainable_parameters':sum(v.numel() for v in head.parameters()),'optimizer_steps':STEPS,
             'initial_head_sha256':initial,'final_head_sha256':digest(head.state_dict()),'fit_seconds':seconds,'fit_peak_allocated_mib':peak,
             'excluded_query_person_bank_entries':exclusions,'checkpoint_sha256':sha(file),'checkpoint_bytes':file.stat().st_size,
             'bank_scalars':deployed.bank_z.numel()+deployed.bank_residual.numel(),'bank_bytes':4*(deployed.bank_z.numel()+deployed.bank_residual.numel()),
             'fixed_scale_scalars':1102,'deployment_replay_max_gap':gap,'runtime':metrics,
             'evaluations':score_arrays(prediction,va,risk,domains,folder,arm)}
        record['arms'][arm]=row;write(out/(arm+'_history.json'),{'loss':losses})
        print(json.dumps({'completed':name+'__'+arm,'means':{k:v['full']['mean'] for k,v in row['evaluations'].items()},'seconds':seconds}),flush=True)
        del deployed,head,opt;torch.cuda.empty_cache()
    assert len({record['arms'][a]['initial_head_sha256'] for a in ARMS})==1
    write(out/'result.json',record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'seeds':SEEDS,'arms':ARMS,'fits':27,'optimizer_steps':STEPS,'neural_cap':1129297})
        print('Frozen 27 adapter fits / 9 unchanged cores');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    tr,va=load('train'),load('validation');records=[]
    for protocol,t,domains in banks(tr,va):
        for seed in SEEDS:records.append(run_bank(protocol,seed,t,va,domains))
    write(OUT/'results.json',{'records':records,'bindings':bound,'fits':27,'cores':9,'status':'COMPLETE'})


if __name__=='__main__':main()
