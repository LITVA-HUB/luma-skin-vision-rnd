"""Unchanged stable-color correction recipes on camera-held-out source banks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction import ARMS,features,StableColorModel
from skin_crossfit_correction_train import fit_core,fit_head,predict,score,table_scores
from skin_support_curve import SkinRepresentation
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write

OUT=ROOT/'docs/benchmarks/skin_correction_transfer_v1';RUN=ROOT/'experiments/runs/skin_correction_transfer_v1'
SOURCE=ROOT/'experiments/runs/skin_sampling_transfer_v1';PROTOCOL=ROOT/'docs/research/skin_correction_transfer_protocol_v1.md'
SEEDS=(17,29,43)


def four_folds(person,camera):
    result=np.full(len(person),-1,np.int64)
    for j,(device,n) in enumerate([('SLR',8),('ipod',16)]):
        ids=np.unique(person[camera==device])
        if len(ids)==0:continue
        if len(ids)!=n:raise ValueError('Unexpected camera person count')
        if any(len(np.unique(camera[person==p]))!=1 for p in ids):raise ValueError('Ambiguous person camera')
        for fold,group in enumerate(np.split(np.random.default_rng(917031+j).permutation(ids),4)):
            result[np.isin(person,group)]=fold
    if len(result)==0 or np.any(result<0):raise ValueError('Unsupported camera population')
    return result


def route_tables(predictions,fold):
    if predictions.shape!=(4,len(fold),3) or not np.isin(fold,np.arange(4)).all():raise ValueError('Invalid prediction routing')
    row=np.arange(len(fold));return predictions[fold,row].copy(),predictions[(fold+1)%4,row].copy()


def source_file(protocol,seed):return SOURCE/f'{protocol}__image__s{seed}'/'final.pt'


def bindings():
    files=[PROTOCOL,Path(__file__),ROOT/'tests/test_skin_correction_transfer.py',ROOT/'scripts/skin_crossfit_correction.py',
           ROOT/'scripts/skin_crossfit_correction_train.py',ROOT/'scripts/skin_support_curve.py',ROOT/'scripts/skin_capture_model.py',
           ROOT/'scripts/skin_local_reference_transfer.py',ROOT/'scripts/skin_local_reference_run.py',ROOT/'scripts/skin_mskcc_data.py',
           ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'scripts/skin_mskcc_train_pixels.py',ROOT/'scripts/skin_pair_train.py',ROOT/'src/luma_skin_vision/color.py',
           ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',ROOT/'data/processed/skin_mskcc_pixels_v1/validation.npz',
           ROOT/'docs/benchmarks/skin_crossfit_correction_v1/results.json',ROOT/'docs/benchmarks/skin_crossfit_correction_v1/audit.json']
    for protocol in ('mixed','from_SLR','from_ipod'):
        for seed in SEEDS:files.extend([source_file(protocol,seed),ROOT/f'docs/benchmarks/skin_sampling_transfer_v1/{protocol}__image__s{seed}/result.json'])
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def score_domains(prediction,va,risk,domains,folder,name):
    return {d:score(prediction[m],subset(va,m),risk[m],folder,name+'__'+d) for d,m in domains.items()}


def write_arrays(path,**values):
    if path.exists():
        saved=dict(np.load(path));assert set(saved)==set(values)
        for k in saved:np.testing.assert_array_equal(saved[k],values[k])
    else:np.savez(path,**values)


def run_bank(protocol,seed,tr,va,domains):
    name=f'{protocol}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes())
        for file,h in r['private_files'].items():assert sha(folder/file)==h
        print(json.dumps({'verified_existing':name}),flush=True);return r
    folder.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    assert set(tr['patient']).isdisjoint(va['patient']) and set(tr['site']).isdisjoint(va['site'])
    if 'unseen' in domains:assert set(tr['device']).isdisjoint(va['device'][domains['unseen']])
    folds=four_folds(tr['patient'],tr['device']);x=torch.from_numpy(tr['tokens']).cuda();fold_rows=[];predictions=[]
    for fold in range(4):
        included=folds!=fold;inner=subset(tr,included)
        assert set(inner['patient']).isdisjoint(tr['patient'][~included])
        file=folder/f'fold{fold}.pt';predfile=folder/f'fold{fold}_predictions.npz';record_file=out/f'fold{fold}.json'
        if record_file.exists():
            row=json.loads(record_file.read_bytes());assert sha(file)==row['checkpoint_sha256'] and sha(predfile)==row['predictions_sha256']
            p=np.load(predfile)['prediction']
        else:
            if file.exists() or predfile.exists():raise ValueError('Incomplete core artifacts: inspect before recovery')
            model,mean,std,row=fit_core(inner,seed)
            p=predict(model,x,mean,std)
            torch.save({'state':{k:v.cpu().clone() for k,v in model.state_dict().items()},'target_mean':mean.cpu(),'target_std':std.cpu(),
                        'seed':seed,'fold':fold,'protocol_sha256':sha(PROTOCOL)},file)
            np.savez(predfile,prediction=p)
            row.update({'fold':fold,'train_people':len(set(inner['patient'])),'query_people':len(set(tr['patient'][~included])),
                        'train_images':len(inner['target']),'query_images':int((~included).sum()),'train_cameras':np.unique(inner['device']).tolist(),
                        'checkpoint_sha256':sha(file),'predictions_sha256':sha(predfile),'checkpoint_bytes':file.stat().st_size})
            write(record_file,row);del model;torch.cuda.empty_cache()
            print(json.dumps({'core_completed':name+f'/fold{fold}','seconds':row['fit_seconds']}),flush=True)
        fold_rows.append(row);predictions.append(p)
    original=json.loads((ROOT/f'docs/benchmarks/skin_sampling_transfer_v1/{protocol}__image__s{seed}/result.json').read_bytes())
    assert sha(source_file(protocol,seed))==original['checkpoint_sha256']
    saved=torch.load(source_file(protocol,seed),map_location='cpu',weights_only=True)
    core=SkinRepresentation('baseline').cuda().eval().requires_grad_(False);core.load_state_dict(saved['state']);core_hash=digest(core.state_dict())
    mean=saved['target_mean'].cuda();std=saved['target_std'].cuda();ym=mean.cpu().numpy();ys=std.cpu().numpy()
    np.testing.assert_array_equal(ym,tr['target'].mean(0).astype(np.float32));np.testing.assert_array_equal(ys,tr['target'].std(0).astype(np.float32))
    full=predict(core,x,mean,std);oof,matched=route_tables(np.stack(predictions),folds);tables={'in_full':full,'in_matched':matched,'out_person':oof}
    cm=tr['color'].mean(0).astype(np.float32);cs=np.maximum(tr['color'].std(0),1e-6).astype(np.float32)
    target=torch.tensor((tr['target'].astype(np.float32)-ym)/ys,device='cuda');vx=torch.from_numpy(va['tokens']).cuda()
    base=predict(core,vx,mean,std)
    for domain,mask in domains.items():
        historical=np.load(source_file(protocol,seed).parent/(domain+'.npz'))['prediction']
        np.testing.assert_array_equal(base[mask],historical)
    vf=torch.from_numpy(features(va['color'],base,cm,cs,ym,ys)).cuda()
    scale=np.maximum(tr['color'].astype(np.float64).std(0),1e-6)
    risk=np.sqrt(np.min(np.mean(((va['color'][:,None]-tr['color'][None])/scale)**2,axis=2),axis=1))
    write_arrays(folder/'tables.npz',**tables,all_inner_predictions=np.stack(predictions),folds=folds,target=tr['target'],person=tr['patient'],site=tr['site'],
                 color_mean=cm,color_std=cs,target_mean=ym,target_std=ys)
    r={'protocol':protocol,'seed':seed,'train_people':len(set(tr['patient'])),'train_images':len(tr['target']),'train_cameras':np.unique(tr['device']).tolist(),
       'folds':fold_rows,'source_core_sha256':sha(source_file(protocol,seed)),'core_state_sha256':core_hash,'original_core_replays_exact':True,
       'source_exploratory_only':True,'validation_used_for_fit_or_selection':False,'reserved_endpoint_access':False,'risk_calibrated':False,
       'table_training_errors':{a:table_scores(p,tr['target'],tr['patient'],tr['site']) for a,p in tables.items()},'arms':{}}
    r['arms']['base']={'parameters':929297,'evaluations':score_domains(base,va,risk,domains,folder,'base')}
    for arm in ARMS:
        record_file=out/(arm+'.json');file=folder/(arm+'.pt')
        if record_file.exists():
            row=json.loads(record_file.read_bytes());assert sha(file)==row['checkpoint_sha256']
            for d,v in row['evaluations'].items():assert sha(folder/f'{arm}__{d}.npz')==v['array_sha256']
        else:
            if file.exists():raise ValueError('Incomplete head artifacts: inspect before recovery')
            f=torch.from_numpy(features(tr['color'],tables[arm],cm,cs,ym,ys)).cuda()
            head,row=fit_head(arm,seed,f,f[:,36:],target)
            with torch.no_grad():prediction=base+head(vf).cpu().numpy()*ys
            deployed=StableColorModel().cuda().eval();deployed.base.load_state_dict(core.state_dict());deployed.head.load_state_dict(head.state_dict())
            with torch.no_grad():
                for key,value in [('color_mean',cm),('color_std',cs),('target_mean',ym),('target_std',ys)]:getattr(deployed,key).copy_(torch.tensor(value,device='cuda'))
                vc=torch.from_numpy(va['color']).cuda()
                replay=torch.cat([deployed(t,c) for t,c in zip(vx.split(32),vc.split(32))]).cpu().numpy()
            gap=float(abs(replay-prediction).max());assert gap<2e-5
            torch.save({'state':{k:v.cpu().clone() for k,v in deployed.state_dict().items()},'seed':seed,'arm':arm,'protocol_sha256':sha(PROTOCOL)},file)
            row.update({'parameters':sum(p.numel() for p in deployed.parameters()),'head_parameters':188035,'checkpoint_sha256':sha(file),
                        'checkpoint_bytes':file.stat().st_size,'deployment_replay_max_gap':gap,'evaluations':score_domains(replay,va,risk,domains,folder,arm)})
            assert row['parameters']==1117332<=1129297
            write(out/(arm+'_history.json'),{'loss':row.pop('loss')});write(record_file,row)
            print(json.dumps({'head_completed':name+'/'+arm,'means':{d:v['full']['mean'] for d,v in row['evaluations'].items()}}),flush=True)
            del deployed,head;torch.cuda.empty_cache()
        r['arms'][arm]=row;assert digest(core.state_dict())==core_hash
    assert len({r['arms'][a]['initial_sha256'] for a in ARMS})==1 and len({r['arms'][a]['draw_sha256'] for a in ARMS})==1
    r['private_files']={p.name:sha(p) for p in folder.iterdir() if p.is_file()};write(out/'result.json',r);return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit']);args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'seeds':SEEDS,'inner_core_fits':36,'head_fits':27,'original_cores':9})
        print('Frozen unchanged correction camera-transfer follow-up');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    tr,va=load('train'),load('validation');records=[]
    for protocol,t,domains in banks(tr,va):
        for seed in SEEDS:records.append(run_bank(protocol,seed,t,va,domains))
    write(OUT/'results.json',{'status':'COMPLETE','records':records,'bindings':bound,'inner_core_fits':36,'head_fits':27})


if __name__=='__main__':main()
