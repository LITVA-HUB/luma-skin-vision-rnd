"""Post-screen inference-step ablation; never replaces the predeclared scores."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_spatial_model import SpatialColor
from skin_spatial_train import OUT,RUN,prediction,subset


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    assert len(list(OUT.glob('*/result.json')))==54
    lock=json.loads((OUT/'source_lock.json').read_bytes())
    for file,digest in lock['bindings'].items():assert sha(ROOT/file)==digest
    va=load('validation');records=[];comparisons=0;maxgap=0.
    for path in sorted(OUT.glob('*/result.json')):
        r=json.loads(path.read_bytes())
        if r['arm'] not in ['graph3','graph3_scrambled']:continue
        folder=RUN/path.parent.name;checkpoint=folder/'best.pt';assert sha(checkpoint)==r['best_sha256']
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        model=SpatialColor(r['arm']).cuda().eval();model.load_state_dict(state['state'])
        ev=va if r['protocol']=='mixed' else subset(va,va['device']!=r['protocol'][5:])
        x=torch.from_numpy(ev['tokens']).cuda();mean=state['target_mean'].cuda();std=state['target_std'].cuda()
        for steps in [0,1,3,8]:
            model.steps=steps;p,risk=prediction(model,x,mean,std)
            if steps==3:
                with np.load(folder/'evaluation.npz') as old:np.testing.assert_array_equal(p,old['prediction'])
            e=delta_e00(p,ev['target']);independent=np.array([scalar_de(a,b) for a,b in zip(p,ev['target'],strict=True)])
            comparisons+=len(e);maxgap=max(maxgap,float(np.abs(e-independent).max()))
            records.append({'arm':r['arm'],'protocol':r['protocol'],'seed':r['seed'],'inference_steps':steps,
                'checkpoint_sha256':r['best_sha256'],'mean_delta_e00':float(e.mean()),'p95_delta_e00':float(np.quantile(e,.95)),
                'parameters_retrained':False,'post_screen_diagnostic':True})
        del model,x;torch.cuda.empty_cache()
    assert len(records)==72 and maxgap<1e-10
    result={'scope':'POST-SCREEN source diagnostic; all steps0/1/3/8 kept; no new best-epoch/model selection or reserved endpoints',
            'source_lock_sha256':sha(OUT/'source_lock.json'),'script_sha256':sha(Path(__file__)),
            'independent_scalar_cases':comparisons,'maximum_metric_gap':maxgap,'rows':records}
    target=OUT/'recurrence_probe.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for protocol in ['mixed','from_SLR','from_ipod']:
        for arm in ['graph3','graph3_scrambled']:
            print(json.dumps({'protocol':protocol,'arm':arm,'mean_by_steps':{str(k):float(np.mean([r['mean_delta_e00'] for r in records if r['protocol']==protocol and r['arm']==arm and r['inference_steps']==k])) for k in [0,1,3,8]}}))


if __name__=='__main__':main()
