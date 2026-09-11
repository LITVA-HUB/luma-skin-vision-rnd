"""Nine person-excluded core fits and nine matched stable-color correction heads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction import ARMS,inner_folds,choose_predictions,features,StableHead,StableColorModel
from skin_support_curve import SkinRepresentation,patient_roles
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
from skin_local_reference_run import summarize
from luma_skin_vision.color import delta_e00

OUT=ROOT/'docs/benchmarks/skin_crossfit_correction_v1';RUN=ROOT/'experiments/runs/skin_crossfit_correction_v1'
SOURCE=ROOT/'experiments/runs/skin_support_curve_v1';PROTOCOL=ROOT/'docs/research/skin_crossfit_correction_protocol_v1.md'
SEEDS=(17,29,43)


def source_file(seed):return SOURCE/f'n18__baseline__s{seed}'/'final.pt'


def bindings():
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_crossfit_correction.py',ROOT/'tests/test_skin_crossfit_correction.py',
           ROOT/'scripts/skin_support_curve.py',ROOT/'scripts/skin_capture_model.py',ROOT/'scripts/skin_mskcc_pixels.py',
           ROOT/'scripts/skin_mskcc_data.py',ROOT/'scripts/skin_mskcc_train_pixels.py',ROOT/'scripts/skin_pair_train.py',
           ROOT/'scripts/skin_local_reference_run.py',ROOT/'src/luma_skin_vision/color.py',ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz']
    for seed in SEEDS:paths.extend([source_file(seed),ROOT/f'docs/benchmarks/skin_support_curve_v1/n18__baseline__s{seed}/result.json'])
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def predict(model,tokens,mean,std):
    model.eval()
    with torch.no_grad():return np.concatenate([(model(t,None)[0]*std+mean).cpu().numpy() for t in tokens.split(32)])


def fit_core(tr,seed):
    torch.manual_seed(seed);model=SkinRepresentation('baseline').cuda();initial=digest(model.state_dict())
    mean=torch.tensor(tr['target'].mean(0).astype(np.float32),device='cuda')
    std=torch.tensor(tr['target'].std(0).astype(np.float32),device='cuda')
    x=torch.from_numpy(tr['tokens']).cuda();y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    trace=hashlib.sha256();history=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,31):
        batch_indices=np.random.default_rng(seed*1000+epoch).integers(0,len(x),(31,32),dtype=np.int64)
        trace.update(batch_indices.tobytes());losses=[]
        for batch in batch_indices:
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True);p,g,_=model(x[ix],None)
            loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite core loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();history.append(float(np.mean(losses)))
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    record={'initial_sha256':initial,'final_state_sha256':digest(model.state_dict()),'draw_sha256':trace.hexdigest(),
            'updates':930,'fit_seconds':seconds,'fit_peak_allocated_mib':peak,'round_loss':history}
    return model.eval(),mean,std,record


def table_scores(prediction,target,people,sites):return summarize(delta_e00(prediction,target),people,sites)


def score(prediction,ev,risk,folder,name):
    error=delta_e00(prediction,ev['target']);order=np.argsort(risk,kind='stable');coverage=[]
    for c in (1.,.95,.9,.8,.7,.6):
        n=int(np.ceil(c*len(error)));ix=order[:n]
        coverage.append({'coverage':c,'accepted':n,**summarize(error[ix],ev['patient'][ix],ev['site'][ix])})
    file=folder/(name+'.npz');np.savez(file,prediction=prediction,target=ev['target'],patient=ev['patient'],site=ev['site'],error=error,risk=risk,order=order,
                                    curve=np.cumsum(error[order])/np.arange(1,len(error)+1))
    return {'full':summarize(error,ev['patient'],ev['site']),'coverage':coverage,'array_sha256':sha(file)}


def fit_head(arm,seed,x,base,target):
    torch.manual_seed(seed);head=StableHead().cuda();initial=digest(head.state_dict())
    with torch.no_grad():assert torch.equal(head(x),torch.zeros_like(base))
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,300,eta_min=.00001)
    indices=np.random.default_rng(seed).integers(0,len(x),(300,32),dtype=np.int64)
    history=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for batch in indices:
        ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True)
        loss=(base[ix]+head(x[ix])-target[ix]).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite head loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),float('inf'),error_if_nonfinite=True)
        opt.step();scheduler.step();history.append(float(loss.detach()))
    return head.eval(),{'arm':arm,'initial_sha256':initial,'final_state_sha256':digest(head.state_dict()),'updates':300,
                        'draw_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),'fit_seconds':time.perf_counter()-start,
                        'fit_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'loss':history}


def run_seed(data,seed):
    folder=RUN/f's{seed}';out=OUT/f's{seed}'
    if (out/'result.json').exists():
        record=json.loads((out/'result.json').read_bytes())
        for name,expected in record['private_files'].items():assert sha(folder/name)==expected
        print(json.dumps({'verified_existing':seed}),flush=True);return record
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    a,b=patient_roles(data['patient'],data['device'],18,seed);tr,ev=subset(data,a),subset(data,b)
    assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
    folds=inner_folds(tr['patient'],tr['device']);all_x=torch.from_numpy(tr['tokens']).cuda()
    fold_records=[];predictions=[]
    for fold in range(3):
        mask=folds!=fold;inner=subset(tr,mask)
        assert len(set(inner['patient']))==12 and set(inner['patient']).isdisjoint(tr['patient'][~mask])
        model,mean,std,record=fit_core(inner,seed)
        saved={'state':{k:v.cpu().clone() for k,v in model.state_dict().items()},'target_mean':mean.cpu(),'target_std':std.cpu(),'seed':seed,'fold':fold,'protocol_sha256':sha(PROTOCOL)}
        file=folder/f'fold{fold}.pt';torch.save(saved,file);p=predict(model,all_x,mean,std);predictions.append(p)
        record.update({'fold':fold,'train_people':12,'query_people':6,'train_images':len(inner['target']),'query_images':int((~mask).sum()),
                       'train_slr_people':len(set(inner['patient'][inner['device']=='SLR'])),'train_ipod_people':len(set(inner['patient'][inner['device']=='ipod'])),
                       'checkpoint_sha256':sha(file),'checkpoint_bytes':file.stat().st_size})
        fold_records.append(record);del model;torch.cuda.empty_cache()
        print(json.dumps({'core_completed':f's{seed}/fold{fold}','seconds':record['fit_seconds']}),flush=True)
    original=json.loads((ROOT/f'docs/benchmarks/skin_support_curve_v1/n18__baseline__s{seed}/result.json').read_bytes())
    assert sha(source_file(seed))==original['checkpoint_sha256']
    saved=torch.load(source_file(seed),map_location='cpu',weights_only=True)
    core=SkinRepresentation('baseline').cuda().eval().requires_grad_(False);core.load_state_dict(saved['state']);core_hash=digest(core.state_dict())
    mean=saved['target_mean'].cuda();std=saved['target_std'].cuda();ym=mean.cpu().numpy();ys=std.cpu().numpy()
    np.testing.assert_array_equal(ym,tr['target'].mean(0).astype(np.float32));np.testing.assert_array_equal(ys,tr['target'].std(0).astype(np.float32))
    full=predict(core,all_x,mean,std);oof,matched=choose_predictions(np.stack(predictions),folds)
    tables={'in_full':full,'in_matched':matched,'out_person':oof}
    cm=tr['color'].mean(0).astype(np.float32);cs=np.maximum(tr['color'].std(0),1e-6).astype(np.float32)
    target=torch.tensor(((tr['target'].astype(np.float32)-ym)/ys),device='cuda')
    vtokens=torch.from_numpy(ev['tokens']).cuda();vpred=predict(core,vtokens,mean,std)
    historical=np.load(source_file(seed).parent/'evaluation.npz')['prediction'];np.testing.assert_array_equal(vpred,historical)
    vx=torch.from_numpy(features(ev['color'],vpred,cm,cs,ym,ys)).cuda()
    scale=np.maximum(tr['color'].astype(np.float64).std(0),1e-6)
    risk=np.sqrt(np.min(np.mean(((ev['color'][:,None]-tr['color'][None])/scale)**2,axis=2),axis=1))
    np.savez(folder/'tables.npz',**tables,all_inner_predictions=np.stack(predictions),folds=folds,train_mask=a,evaluation_mask=b,
             target=tr['target'],person=tr['patient'],site=tr['site'],color_mean=cm,color_std=cs,target_mean=ym,target_std=ys)
    record={'seed':seed,'train_people':18,'train_images':len(tr['target']),'evaluation_people':6,'evaluation_images':len(ev['target']),
            'folds':fold_records,'source_core_sha256':sha(source_file(seed)),'core_state_sha256':core_hash,'original_core_replay_exact':True,
            'source_exploratory_only':True,'only_original_TRAIN_loaded':True,'risk_calibrated':False,'reserved_endpoint_access':False,
            'table_training_errors':{name:table_scores(p,tr['target'],tr['patient'],tr['site']) for name,p in tables.items()},'arms':{}}
    record['arms']['base']={'parameters':929297,'evaluation':score(vpred,ev,risk,folder,'base')}
    for arm in ARMS:
        x=torch.from_numpy(features(tr['color'],tables[arm],cm,cs,ym,ys)).cuda()
        head,row=fit_head(arm,seed,x,x[:,36:],target)
        with torch.no_grad():pred=vpred+head(vx).cpu().numpy()*ys
        deployed=StableColorModel().cuda().eval();deployed.base.load_state_dict(core.state_dict());deployed.head.load_state_dict(head.state_dict())
        with torch.no_grad():
            for name,value in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:getattr(deployed,name).copy_(torch.tensor(value,device='cuda'))
            vc=torch.from_numpy(ev['color']).cuda()
            replay=torch.cat([deployed(t,c) for t,c in zip(vtokens.split(32),vc.split(32))]).cpu().numpy()
        gap=float(abs(replay-pred).max());assert gap<2e-5
        file=folder/(arm+'.pt');torch.save({'state':{k:v.cpu().clone() for k,v in deployed.state_dict().items()},'seed':seed,'arm':arm,'protocol_sha256':sha(PROTOCOL)},file)
        row.update({'parameters':sum(v.numel() for v in deployed.parameters()),'head_parameters':sum(v.numel() for v in head.parameters()),
                    'checkpoint_sha256':sha(file),'checkpoint_bytes':file.stat().st_size,'deployment_replay_max_gap':gap,
                    'evaluation':score(replay,ev,risk,folder,arm)})
        assert row['parameters']==1117332<=1129297 and row['head_parameters']==188035
        write(out/(arm+'_history.json'),{'loss':row.pop('loss')});record['arms'][arm]=row
        assert digest(core.state_dict())==core_hash
        print(json.dumps({'head_completed':f's{seed}/{arm}','mean':row['evaluation']['full']['mean'],'at80':row['evaluation']['coverage'][3]['mean']}),flush=True)
        del deployed,head;torch.cuda.empty_cache()
    assert len({record['arms'][arm]['initial_sha256'] for arm in ARMS})==1
    assert len({record['arms'][arm]['draw_sha256'] for arm in ARMS})==1
    record['private_files']={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    write(out/'result.json',record);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'seeds':SEEDS,'new_inner_cores':9,'heads':9,'outer_core_replays':3,'only_original_TRAIN':True})
        print('Frozen subject-excluded correction screen');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    data=load('train');records=[run_seed(data,seed) for seed in SEEDS]
    write(OUT/'results.json',{'status':'COMPLETE','records':records,'bindings':bound,'new_core_fits':9,'head_fits':9})


if __name__=='__main__':main()
