"""18subject-excluded color fits produce honest TRAIN residual labels for risk."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
import time
import numpy as np
import torch
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import ROOT,OUT,RUN,PROTOCOL,SEEDS,patient_folds,plain_predict,density
from skin_mskcc_vote import PatchVotes
from skin_mskcc_train_pixels import predict
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_data import sha
from luma_skin_vision.color import delta_e00


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    train,val=load('train'),load('validation');folds=patient_folds(train['patient'])
    fold=np.array([folds[p] for p in train['patient']]);pred=np.zeros((3,len(fold),3));wit=np.zeros((3,len(fold),8));dist=np.zeros(len(fold))
    rows=[{k:val[k][i].item() for k in ['patient','site']} for i in range(len(val['target']))]
    run=RUN/'oof';run.mkdir(exist_ok=False,parents=True);OUT.mkdir(exist_ok=True,parents=True)
    records=[];covered=np.zeros((3,len(fold)),dtype=int)
    for f in range(6):
        fit=np.flatnonzero(fold!=f);held=np.flatnonzero(fold==f)
        fit_people=set(train['patient'][fit]);held_people=set(train['patient'][held])
        assert fit_people.isdisjoint(held_people) and len(fit_people)==20 and len(held_people)==4
        dist[held]=density(train['color'][fit],train['color'][held])
        for si,seed in enumerate(SEEDS):
            start=time.perf_counter();torch.manual_seed(seed)
            mean=torch.tensor(train['target'][fit].mean(0),dtype=torch.float32,device='cuda')
            std=torch.tensor(train['target'][fit].std(0),dtype=torch.float32,device='cuda')
            x=torch.from_numpy(train['tokens'][fit]).cuda();vx=torch.from_numpy(val['tokens']).cuda()
            y=(torch.tensor(train['target'][fit],dtype=torch.float32,device='cuda')-mean)/std
            model=PatchVotes(std.detach().cpu(),0).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
            best=float('inf');best_epoch=None;path=run/f'fold{f}_seed{seed}.pt'
            for epoch in range(1,81):
                model.train();order=np.random.default_rng(seed*1000+epoch).permutation(len(fit))
                for offset in range(0,len(order),32):
                    ix=torch.tensor(order[offset:offset+32],device='cuda');opt.zero_grad(set_to_none=True)
                    loss=(model(x[ix])-y[ix]).square().mean()
                    if not torch.isfinite(loss):raise ValueError('InvalidOOFfit')
                    loss.backward();opt.step()
                schedule.step();p,_=predict(model,vx,mean,std,'votes_mean')
                score=summarize(delta_e00(p,val['target']),rows)['patient_balanced_mean']
                if score<best:
                    best=score;best_epoch=epoch
                    torch.save({'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                'target_mean':mean.cpu(),'target_std':std.cpu(),'fold':f,'seed':seed,'epoch':epoch},path)
            query={'tokens':train['tokens'][held]};p,w=plain_predict(path,query)
            pred[si,held]=p;wit[si,held]=w;covered[si,held]+=1
            np.savez(run/f'fold{f}_seed{seed}.npz',held_indices=held,prediction=p,witness=w)
            records.append({'fold':f,'seed':seed,'fit_people':20,'heldout_people':4,'fit_images':len(fit),'heldout_images':len(held),
                            'best_epoch':best_epoch,'selection_validation_mean':best,'checkpoint_sha256':sha(path),'seconds':time.perf_counter()-start})
            (OUT/'oof_progress.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
            print(json.dumps(records[-1]),flush=True)
            del model,opt,x,vx,y,mean,std;torch.cuda.empty_cache()
    assert np.all(covered==1) and np.isfinite(pred).all() and np.isfinite(wit).all()
    path=RUN/'oof.npz';np.savez(path,prediction=pred,witness=wit,density=dist,fold=fold,target=train['target'],patient=train['patient'])
    receipt={'protocol_sha256':sha(PROTOCOL),'oof_sha256':sha(path),'fits':records,'coverage':'Each TRAIN image predicted exactly once per seed by subject-excluded fit',
             'calibration_test_opened':False,'script_sha256':sha(__file__), 'core_sha256':sha(ROOT/'scripts/skin_mskcc_selective_core.py')}
    (OUT/'oof.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
