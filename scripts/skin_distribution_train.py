"""Matched real-image conditional native-Lab distribution experiment."""
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
from skin_pair_train import subset,rows,write
from skin_capture_model import MODES
from skin_distribution_model import ARMS,ColorDistribution,density_nll,color_decision

OUT=ROOT/'docs/benchmarks/skin_distribution_v1'
RUN=ROOT/'experiments/runs/skin_distribution_v1'
PROTOCOL=ROOT/'docs/research/skin_distribution_protocol_v1.md'
SEEDS=[17,29,43]


def prediction(model,x,mean,std):
    model.eval();values=[[],[],[],[]]
    with torch.no_grad():
        for part in x.split(64):
            p,g,h,s=model(part)
            for bucket,value in zip(values,((p*std+mean),g.softmax(1),h*std+mean,s*std)):
                bucket.append(value.cpu().numpy())
    return tuple(np.concatenate(v) for v in values)


def endpoints(arm,p,g,h,s):
    if arm in ('mse','mse_mode'):
        risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
        return {'mean':(p,risk)},{}
    if arm=='gaussian':h=p[:,None];s=s.mean(1,keepdims=True);g=np.ones((len(p),1))
    a=color_decision(h,s,g,3);b=color_decision(h,s,g,2)
    return {'mean':(p,a['mean_expected_error']),
        'decision3':(a['prediction'],a['expected_error']),
        'decision2':(b['prediction'],b['expected_error'])},a


def score(p,risk,ev):
    e=delta_e00(p,ev['target']);order=np.argsort(risk,kind='stable')
    coverage=[]
    for c in (1.,.95,.9,.8,.7,.6):
        n=int(np.ceil(len(e)*c));ix=order[:n]
        coverage.append({'requested_coverage':c,'accepted':n,**summarize(e[ix],rows(subset(ev,ix)))})
    return {'full':summarize(e,rows(ev)),'coverage':coverage,
        'predicted_error_mae_or_dispersion_mae':float(np.mean(np.abs(risk-e)))},e,order


def fit(arm,seed,tr,va,protocol,evaluation=None):
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        rec=json.loads((out/'result.json').read_bytes())
        assert sha(folder/'best.pt')==rec['best_sha256'] and sha(folder/'final.pt')==rec['final_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient'])
    if evaluation is not None:
        assert set(tr['device']).isdisjoint(evaluation['device']) and set(va['device']).isdisjoint(evaluation['device'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();vx=torch.from_numpy(va['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    modes=torch.tensor([MODES.index(str(v)) for v in tr['mode']],device='cuda')
    model=ColorDistribution();initial=digest(model.state_dict())
    base_initial=digest({k:v for k,v in model.state_dict().items() if not k.startswith('scale_head.')})
    model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'arm':arm,'seed':seed,'epoch':epoch,'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),
            'protocol_sha256':sha(PROTOCOL)}
    steps=int(np.ceil(len(x)/32));best=float('inf');history=[]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,81):
        model.train();a,b=pair_indices(tr['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);p,g,h,s=model(x[ix])
            if arm in ('mse','mse_mode'):
                loss=(p-y[ix]).square().mean()
                if arm=='mse_mode':loss=loss+.1*torch.nn.functional.cross_entropy(g,modes[ix])
            elif arm=='gaussian':loss=(2/3)*density_nll(y[ix],p[:,None],s.mean(1,keepdim=True),g[:,:1]*0).mean()
            else:loss=(2/3)*density_nll(y[ix],h,s,g).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite density objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std)[0]
        metric=summarize(delta_e00(p,va['target']),rows(va));value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_patient_mean':value,'selection_mean':metric['mean']})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0])
    elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=va if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h,s=prediction(model,ex,mean,std);ep,decision=endpoints(arm,p,g,h,s)
    arrays={'point':p,'gate':g,'hypotheses':h,'scales':s,'target':ev['target'],'patient':ev['patient'],'site':ev['site']}
    scores={}
    for label,(pred,risk) in ep.items():
        scores[label],e,order=score(pred,risk,ev)
        arrays.update({label+'_prediction':pred,label+'_risk':risk,label+'_error':e,
            label+'_curve':np.cumsum(e[order])/np.arange(1,len(e)+1)})
    np.savez(folder/'evaluation.npz',**arrays)
    record={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'scores':scores,'initial_sha256':initial,'base_initial_sha256':base_initial,
        'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'stored_parameters':sum(p.numel() for p in model.parameters()),
        'active_parameters':sum(p.numel() for k,p in model.named_parameters() if not(arm in ('mse','mse_mode') and k.startswith('scale_head.'))),
        'checkpoint_bytes':(folder/'best.pt').stat().st_size,'fit_peak_allocated_mib':peak,'fit_seconds':elapsed,
        'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(x),'selection_images':len(vx),'evaluation_images':len(ex),
        'train_cameras':np.unique(tr['device']).tolist(),'selection_cameras':np.unique(va['device']).tolist(),
        'evaluation_cameras':np.unique(ev['device']).tolist(),'source_exploratory_only':True,
        'risk_calibrated':False,'reserved_endpoint_access':False,
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),
        'model_code_sha256':sha(ROOT/'scripts/skin_distribution_model.py')}
    if decision:
        record['quadrature_decision2_vs3_mean_delta_e00']=float(delta_e00(ep['decision2'][0],ep['decision3'][0]).mean())
        record['fraction_decision3_changes_mean']=float((delta_e00(ep['mean'][0],ep['decision3'][0])>.01).mean())
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'means':{k:v['full']['mean'] for k,v in scores.items()},'seconds':elapsed}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','mixed','transfer']);stage=parser.parse_args().stage
    files=[PROTOCOL,Path(__file__)]+[ROOT/'scripts'/n for n in [
        'skin_distribution_model.py','skin_capture_model.py','skin_mskcc_data.py','skin_mskcc_pixels.py',
        'skin_mskcc_summary_pilot.py','skin_mskcc_train_pixels.py','skin_pair_invariance.py','skin_pair_train.py']]
    files += [ROOT/'tests/test_skin_distribution_model.py',ROOT/'src/luma_skin_vision/color.py']
    files += [ROOT/'data/processed/skin_mskcc_pixels_v1'/f'{role}.npz' for role in ('train','validation')]
    bindings={str(p.relative_to(ROOT)):sha(p) for p in files};lock=OUT/'source_lock.json'
    if stage=='prepare':
        OUT.mkdir(parents=True,exist_ok=True)
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bindings
        else:write(lock,{'bindings':bindings,'arms':ARMS,'seeds':SEEDS,'reserved_endpoint_access':False})
        print('Frozen source and code; no fitting');return
    assert json.loads(lock.read_bytes())['bindings']==bindings
    tr,va=load('train'),load('validation')
    protocols=[('mixed',tr,va,None)] if stage=='mixed' else [
        ('from_'+c,subset(tr,tr['device']==c),subset(va,va['device']==c),subset(va,va['device']!=c)) for c in ('SLR','ipod')]
    for protocol,t,v,e in protocols:
        for seed in SEEDS:
            for arm in ARMS:fit(arm,seed,t,v,protocol,e)


if __name__=='__main__':main()
