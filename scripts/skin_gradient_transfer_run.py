"""Frozen source checkpoint derivative and finite-update diagnostic."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_gradient_transfer import partitions,group_means,gradient_statistics,transient_step,component_gradients,STEPS
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_support_curve import SkinRepresentation
from skin_capture_model import MODES
from skin_pair_train import subset,write

OUT=ROOT/'docs/benchmarks/skin_gradient_transfer_v1'
RUN=ROOT/'experiments/runs/skin_gradient_transfer_v1'
SOURCE=ROOT/'experiments/runs/skin_sampling_transfer_v1'
PROTOCOL=ROOT/'docs/research/skin_gradient_transfer_protocol_v1.md'
MODELS=[f'{p}__{a}__s17' for p in ('mixed','from_SLR','from_ipod') for a in ('image','person_color')]


def bindings():
    files=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_gradient_transfer.py',ROOT/'tests/test_skin_gradient_transfer.py',
           ROOT/'scripts/skin_support_curve.py',ROOT/'scripts/skin_capture_model.py',ROOT/'scripts/skin_mskcc_pixels.py',
           ROOT/'scripts/skin_mskcc_data.py',ROOT/'scripts/skin_mskcc_train_pixels.py',ROOT/'scripts/skin_pair_train.py',
           ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/cache.json',
           ROOT/'docs/research/skin_mskcc_pixel_protocol_v1.md',ROOT/'src/luma_skin_vision/color.py']
    files += [SOURCE/name/'final.pt' for name in MODELS]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def setup(name,train):
    protocol=name.split('__')[0]
    tr=train if protocol=='mixed' else subset(train,train['device']==protocol.removeprefix('from_'))
    checkpoint=torch.load(SOURCE/name/'final.pt',map_location='cpu',weights_only=True)
    mean=checkpoint['target_mean'].cuda();std=checkpoint['target_std'].cuda()
    np.testing.assert_array_equal(mean.cpu().numpy(),tr['target'].mean(0).astype(np.float32))
    np.testing.assert_array_equal(std.cpu().numpy(),tr['target'].std(0).astype(np.float32))
    y=((torch.from_numpy(tr['target']).cuda().float()-mean)/std).double()
    x=torch.from_numpy(tr['tokens']).cuda().double()
    model=SkinRepresentation('baseline');model.load_state_dict(checkpoint['state']);model=model.cuda().double().eval()
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    return tr,model,x,y,mode,mean.double(),std.double()


def values(model,x,y,mode,mean,std,target):
    pred=[];mse=[];ce=[]
    with torch.no_grad():
        for start in range(0,len(x),32):
            sl=slice(start,start+32);p,g,_=model(x[sl],None)
            pred.append((p*std+mean).cpu().numpy());mse.append((p-y[sl]).square().mean(1).cpu().numpy())
            ce.append(torch.nn.functional.cross_entropy(g,mode[sl],reduction='none').cpu().numpy())
    p=np.concatenate(pred);return p,np.column_stack([np.concatenate(mse),np.concatenate(ce),delta_e00(p,target)])


def run_model(name,train):
    path=OUT/(name+'.json');folder=RUN/name
    if path.exists():
        rec=json.loads(path.read_bytes())
        for f,h in rec['arrays'].items():assert sha(folder/f)==h
        print(json.dumps({'verified_existing':name}),flush=True);return rec
    folder.mkdir(parents=True,exist_ok=True)
    tr,model,x,y,mode,mean,std=setup(name,train);state_hash=digest(model.state_dict())
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    pred,base=values(model,x,y,mode,mean,std,tr['target'])
    np.savez(folder/'base.npz',prediction=pred,values=base,target=tr['target'],standardized_target=y.cpu().numpy())
    pooled=torch.stack(component_gradients(model,x,y,mode,np.arange(len(x))))
    np.savez(folder/'pooled.npz',gradients=pooled.cpu().numpy())
    cases=[];invariance_gap=0.;restorations=0
    for part,group in partitions(tr['patient']):
        k=group.max()+1;weights=np.bincount(group)/len(group)
        parts=[component_gradients(model,x,y,mode,np.flatnonzero(group==j)) for j in range(k)]
        color=torch.stack([g[0] for g in parts]);aux=torch.stack([g[1] for g in parts]);del parts
        stack=torch.cat([color,aux]);gram=(stack@stack.T).cpu().numpy();del stack
        for j,g in enumerate((color,aux)):
            gap=float((torch.tensor(weights,device=x.device)@g-pooled[j]).abs().max());invariance_gap=max(invariance_gap,gap)
            assert gap<1e-10
        gm=group_means(base,group)
        np.testing.assert_allclose(weights@gm,base.mean(0),atol=1e-12,rtol=1e-12)
        np.savez(folder/(part+'__gram.npz'),gram=gram,group=group,weights=weights,base_group=gm)
        for objective,coefficient in [('color',0.),('joint',.1)]:
            grads=color+coefficient*aux
            gg=gram[:k,:k]+coefficient*(gram[:k,k:]+gram[k:,:k])+coefficient**2*gram[k:,k:]
            statistics=gradient_statistics(gg,weights);steps=[]
            for source in (0,k//2,k-1):
                norm=float(grads[source].norm())
                for length in STEPS:
                    with transient_step(model,grads[source],length):
                        p,v=values(model,x,y,mode,mean,std,tr['target'])
                    assert digest(model.state_dict())==state_hash;restorations+=1
                    filename=f'{part}__{objective}__g{source}__h{length}.npz'
                    first=-length*gg[source]/norm;changes=group_means(v-base,group)
                    actual=changes[:,0]+coefficient*changes[:,1]
                    np.savez(folder/filename,prediction=p,values=v,group_changes=changes,first_order=first)
                    other=np.arange(k)!=source
                    steps.append({'file':filename,'source_index':int(source),'length':length,'gradient_norm':norm,
                        'max_first_order_remainder':float(abs(actual-first).max()),'max_first_order_magnitude':float(abs(first).max()),
                        'other_groups_objective_increased_fraction':float((actual[other]>0).mean()),
                        'objective_change_image_mean':float(weights@actual),'color_mse_change_image_mean':float((v-base)[:,0].mean()),
                        'skin_delta_e00_change_image_mean':float((v-base)[:,2].mean()),
                        'own_objective_change':float(actual[source])})
            cases.append({'partition':part,'objective':objective,'statistics':statistics,'steps':steps})
        print(json.dumps({'model':name,'partition':part,'color_cosine':cases[-2]['statistics']['mean_cosine'],
                          'seconds':time.perf_counter()-start}),flush=True)
        del color,aux,grads
    pg=(pooled@pooled.T).cpu().numpy();mode_norm=float(np.sqrt(pg[1,1]));color_norm=float(np.sqrt(pg[0,0]))
    rec={'model':name,'images':len(x),'people':len(set(tr['patient'])),'parameters':sum(p.numel() for p in model.parameters()),
         'baseline_train_mean_delta_e00':float(base[:,2].mean()),'baseline_mse':float(base[:,0].mean()),'baseline_mode_ce':float(base[:,1].mean()),
         'pooled_color_mode_cosine':float(pg[0,1]/(mode_norm*color_norm)),'weighted_aux_to_color_norm':float(.1*mode_norm/color_norm),
         'pooled_component_gram':pg.tolist(),'pooled_gradient_max_partition_gap':invariance_gap,'exact_restorations':restorations,
         'diagnostic_seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
         'original_checkpoint_sha256':sha(SOURCE/name/'final.pt'),'fp64_state_sha256':state_hash,
         'cases':cases,'arrays':{p.name:sha(p) for p in folder.glob('*.npz')},'reserved_endpoint_access':False,
         'new_model_training':False,'diagnostic_only':True}
    write(path,rec);del model,x,y,pooled;torch.cuda.empty_cache();return rec


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'models':MODELS,'objective_cases':48,'transient_steps':288,'train_only':True})
        print('Frozen six-model TRAIN gradient audit');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    train=load('train');records=[run_model(name,train) for name in MODELS]
    assert bindings()==bound
    write(OUT/'results.json',{'models':records,'source_lock_sha256':sha(lock),'train_only':True,'independent_accuracy_changed':False})
    print('All six TRAIN diagnostics complete; no original checkpoint changed',flush=True)


if __name__=='__main__':main()
