"""Independent replay and scalar color audit for the12adaptive source ablations."""
import json
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_vote_v2 import GlobalColorMLP,VoteAblation
from skin_mskcc_train_ablation_v2 import predict,ABLATION_PROTOCOL
from skin_mskcc_audit import scalar_de
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00


def main():
    torch.set_num_threads(4);val=load('validation')
    bench=ROOT/'docs/benchmarks/skin_mskcc_pixel_ablation_v2';runs=ROOT/'experiments/runs/skin_mskcc_pixel_ablation_v2'
    rows=[{k:val[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(val['target']))]
    summary={'source_validation_only':True,'models':[],'ensembles':[],'audit':{'checkpoint_replays':0,'max_replay_gap':0.,'max_scalar_gap':0.}}
    for arch in ['global_mlp','shared_tight','local_mean','local_tight']:
        preds=[]
        for seed in [17,29,43]:
            name=f'{arch}_seed{seed}';r=json.loads((bench/name/'result.json').read_bytes())
            assert r['protocol_sha256']==sha(PROTOCOL) and r['ablation_protocol_sha256']==sha(ABLATION_PROTOCOL)
            assert r['training_script_sha256']==sha(ROOT/'scripts/skin_mskcc_train_ablation_v2.py')
            assert r['vote_script_sha256']==sha(ROOT/'scripts/skin_mskcc_vote_v2.py')
            for checkpoint in ['best','final']:
                path=runs/name/(checkpoint+'.pt');assert sha(path)==r[checkpoint+'_sha256']
                state=torch.load(path,weights_only=True,map_location='cpu')
                model=(GlobalColorMLP() if arch=='global_mlp' else VoteAblation(state['target_std'],local_only=arch.startswith('local_'),steps=0 if arch=='local_mean' else 3)).cuda()
                model.load_state_dict(state['state'])
                x=torch.from_numpy(val['color' if arch=='global_mlp' else 'tokens']).cuda()
                pred,risk=predict(model,x,state['target_mean'].cuda(),state['target_std'].cuda(),arch)
                original=np.load(runs/name/(checkpoint+'_validation.npz'))
                gap=float(np.max(np.abs(pred-original['prediction'])))
                summary['audit']['max_replay_gap']=max(summary['audit']['max_replay_gap'],gap);assert gap==0
                e=delta_e00(pred,val['target']);ind=np.array([scalar_de(p,q) for p,q in zip(pred,val['target'])])
                summary['audit']['max_scalar_gap']=max(summary['audit']['max_scalar_gap'],float(np.max(np.abs(e-ind))))
                summary['audit']['checkpoint_replays']+=1
                if checkpoint=='best':preds.append(pred.astype(np.float64))
            summary['models'].append(r)
        mean=np.mean(preds,axis=0);error=delta_e00(mean,val['target'])
        record={'arch':arch,'models':3,'full':summarize(error,rows),'strata':{}}
        for key in ['device','image_type']:
            record['strata'][key]={}
            for v in np.unique(val[key]):
                ids=np.flatnonzero(val[key]==v);record['strata'][key][str(v)]=summarize(error[ids],[rows[i] for i in ids])
        summary['ensembles'].append(record)
        np.savez(runs/(arch+'_ensemble_validation.npz'),prediction=mean,target=val['target'])
    for seed in [17,29,43]:
        r=[r for r in summary['models'] if r['seed']==seed and r['arch']!='global_mlp']
        assert len({x['initial_state_sha256'] for x in r})==1
    summary['audit']['matched_initializations']='3seeds identical across all vote ablations';summary['audit']['status']='PASS'
    (bench/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    lines=['# Adaptive direct-skin source ablations v2','','These reuse the six-person development set. No calibration/test person was opened.','',
           '|Ablation|Seed|Mean DeltaE00|Median|p95|','|---|---:|---:|---:|---:|']
    for r in summary['models']:lines.append(f"|{r['arch']}|{r['seed']}|{r['full']['mean']:.4f}|{r['full']['median']:.4f}|{r['full']['p95']:.4f}|")
    lines+=['','The capacity-matched global MLP has938755parameters; each vote model924932.',
            'Local-only votes lose useful context. Tenfold tighter robust iterations do not provide a reliable gain.',
            'These negative mechanisms remain evidence, not discarded runs. Standard error-head/calibration and independent test remain next steps.',
            'All24best/final checkpoints replay exactly, with independent scalar DeltaE00 checks.']
    (bench/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'audit':summary['audit'],'ensembles':summary['ensembles']}))


if __name__=='__main__':main()
