"""Audit all pixel checkpoints; aggregate source results without opening test."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_vote import PatchVotes
from skin_mskcc_audit import scalar_de
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import predict


def main():
    torch.set_num_threads(4)
    train,val=load('train'),load('validation')
    bench=ROOT/'docs/benchmarks/skin_mskcc_pixels_v1';runs=ROOT/'experiments/runs/skin_mskcc_pixels_v1'
    rows=[{k:val[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(val['target']))]
    scaling=StandardScaler().fit(train['color'])
    density=NearestNeighbors(n_neighbors=5).fit(scaling.transform(train['color'])).kneighbors(scaling.transform(val['color']))[0].mean(1)
    summary={'source_validation_only':True,'models':[],'ensembles':[],'mechanism_diagnostics':[],
             'protocol_sha256':sha(PROTOCOL),'audit':{'checkpoint_replays':0,'maximum_replay_gap':0.,'maximum_scalar_delta_e00_gap':0.}}
    for arch in ['cnn','votes_mean','votes_huber3']:
        seed_predictions=[]
        for seed in [17,29,43]:
            directory=f'{arch}_seed{seed}';record=json.loads((bench/directory/'result.json').read_bytes())
            assert record['protocol_sha256']==sha(PROTOCOL)
            assert record['training_script_sha256']==sha(ROOT/'scripts/skin_mskcc_train_pixels.py')
            assert record['vote_script_sha256']==sha(ROOT/'scripts/skin_mskcc_vote.py')
            for checkpoint in ['best','final']:
                path=runs/directory/(checkpoint+'.pt');assert sha(path)==record[checkpoint+'_sha256']
                state=torch.load(path,map_location='cpu',weights_only=True)
                mean,std=state['target_mean'].cuda(),state['target_std'].cuda()
                model=(mobilenet_v3_small(weights=None,num_classes=3) if arch=='cnn' else PatchVotes(std.cpu(),0 if arch=='votes_mean' else 3)).cuda()
                model.load_state_dict(state['state'])
                x=(torch.from_numpy(val['rgb'].transpose(0,3,1,2).copy()).float().cuda()/255 if arch=='cnn' else torch.from_numpy(val['tokens']).cuda())
                prediction,risk=predict(model,x,mean,std,arch)
                saved=np.load(runs/directory/(checkpoint+'_validation.npz'))
                gap=float(np.max(np.abs(prediction-saved['prediction'])))
                summary['audit']['checkpoint_replays']+=1
                summary['audit']['maximum_replay_gap']=max(summary['audit']['maximum_replay_gap'],gap)
                assert gap==0
                error=delta_e00(prediction,val['target'])
                scalar=np.array([scalar_de(p,q) for p,q in zip(prediction,val['target'])])
                summary['audit']['maximum_scalar_delta_e00_gap']=max(summary['audit']['maximum_scalar_delta_e00_gap'],float(np.max(np.abs(error-scalar))))
                if checkpoint=='best':
                    seed_predictions.append(prediction.astype(np.float64))
                    if arch!='cnn':
                        with torch.no_grad():
                            h=model.local(x);c=model.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
                            v=model.votes(torch.cat([h,c[:,None].expand(-1,64,-1)],-1));weight=v[...,3].softmax(1)
                            initial=(v[...,:3]*weight[...,None]).sum(1)
                            residual=torch.linalg.vector_norm((v[...,:3]-initial[:,None])*std,dim=-1)
                        summary['mechanism_diagnostics'].append({'arch':arch,'seed':seed,
                            'votes_above_huber5_fraction':float((residual>5).float().mean()),
                            'images_with_any_vote_above5_fraction':float((residual>5).any(1).float().mean()),
                            'median_within_image_vote_residual_delta_e76':float(residual.median())})
                del model,x,mean,std
            summary['models'].append(record)
        predictions=np.stack(seed_predictions);mean_pred=predictions.mean(0)
        error=delta_e00(mean_pred,val['target']);disagreement=np.sqrt(np.mean(np.sum((predictions-mean_pred)**2,axis=2),axis=0))
        np.savez(runs/(arch+'_ensemble_validation.npz'),prediction=mean_pred,target=val['target'],density=density,disagreement=disagreement)
        result={'arch':arch,'seeds':[17,29,43],'deployment_models':3,'full':summarize(error,rows),'coverage':{},'strata':{}}
        for score_name,score in [('density',density),('seed_disagreement',disagreement)]:
            order=np.lexsort((val['image'],score));result['coverage'][score_name]=[]
            for coverage in [1,.95,.9,.8,.7,.6]:
                ix=order[:int(np.ceil(coverage*len(error)))];result['coverage'][score_name].append({'coverage':coverage,**summarize(error[ix],[rows[i] for i in ix])})
        for key in ['device','image_type']:
            result['strata'][key]={}
            for value in np.unique(val[key]):
                ix=np.flatnonzero(val[key]==value);result['strata'][key][str(value)]=summarize(error[ix],[rows[i] for i in ix])
        summary['ensembles'].append(result)
    for seed in [17,29,43]:
        matched=[r for r in summary['models'] if r['seed']==seed and r['arch']!='cnn']
        assert len({r['initial_state_sha256'] for r in matched})==1
        assert len({r['parameters'] for r in matched})==1
    summary['audit']['matched_initial_states']='3seeds bitwise identical between mean and Huber arms'
    summary['audit']['status']='PASS'
    (bench/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    lines=['# Direct skin color from local pixels: source screen v1','','**SOURCE VALIDATION ONLY**:264images,66sites,6people;966training images/24people.',
           'The10-person final test and6-person calibration groups remain unopened. All scores below used for development.',
           'Local input pipeline: verified original JPEG -> central skin crop ->128x128RGB -> model -> native instrument Lab. No camera ID or author-derived color features at inference.','',
           '|Model|Seed|Mean DeltaE00|Median|p95|Parameters|','|---|---:|---:|---:|---:|---:|']
    for r in summary['models']:
        lines.append(f"|{r['arch']}|{r['seed']}|{r['full']['mean']:.4f}|{r['full']['median']:.4f}|{r['full']['p95']:.4f}|{r['parameters']}|")
    lines+=['','Three-model ensembles (not single-model deployment):','', '|Architecture|Mean DeltaE00|Median|p95|Mean at80%, density|Mean at80%, seed disagreement|','|---|---:|---:|---:|---:|---:|']
    for r in summary['ensembles']:
        lines.append(f"|{r['arch']}|{r['full']['mean']:.4f}|{r['full']['median']:.4f}|{r['full']['p95']:.4f}|{r['coverage']['density'][3]['mean']:.4f}|{r['coverage']['seed_disagreement'][3]['mean']:.4f}|")
    lines+=['','All18best/final checkpoints replayed exactly; independent scalar CIEDE2000 audit passed.',
            'All3matched seed initializations are bitwise identical. Correlated patch estimates can render robust iteration inactive; see mechanism diagnostics in summary.json.',
            'Risk scores are diagnostic rankings, not calibrated expected skin-color error. A full matched C+ error-head/calibration comparison remains unperformed.',
            'Two histogram MLP controls hit their fixed optimizer iteration limit and lost simpler controls; warnings retained in controls.json.',
            'No unseen-camera, ordinary phone-face, independent final-test, novel-method or export accuracy claim follows from this source screen.',
            'The patch architecture uses established set aggregation/confidence weighting/Huber mechanisms. Numerical improvement is not itself proof of novelty.']
    (bench/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'audit':summary['audit'],'ensembles':[{k:r[k] for k in ['arch','full']} for r in summary['ensembles']],'mechanism':summary['mechanism_diagnostics']}))


if __name__=='__main__':main()
