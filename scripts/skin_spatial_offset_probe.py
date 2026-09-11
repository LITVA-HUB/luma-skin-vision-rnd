"""Opponent control: is graph removal mostly a source-derived constant Lab shift?"""
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
    tr,va=load('train'),load('validation');records=[];gap=0.;cases=0
    for protocol in ['mixed','from_SLR','from_ipod']:
        train=tr if protocol=='mixed' else subset(tr,tr['device']==protocol[5:])
        ev=va if protocol=='mixed' else subset(va,va['device']!=protocol[5:])
        for seed in [17,29,43]:
            path=RUN/f'{protocol}__graph3__s{seed}'/'best.pt'
            record=json.loads((OUT/f'{protocol}__graph3__s{seed}'/'result.json').read_bytes())
            assert sha(path)==record['best_sha256']
            state=torch.load(path,weights_only=True,map_location='cpu');m=SpatialColor('graph3').cuda().eval();m.load_state_dict(state['state'])
            mean,std=state['target_mean'].cuda(),state['target_std'].cuda()
            values={}
            for label,data in [('train',train),('evaluation',ev)]:
                x=torch.from_numpy(data['tokens']).cuda()
                for steps in [0,3]:
                    m.steps=steps;values[label,steps]=prediction(m,x,mean,std)[0]
            # The offset uses only TRAIN predictions; no validation target or prediction.
            differences=values['train',0].astype(float)-values['train',3]
            offset=differences.mean(0)
            corrected=values['evaluation',3]+offset
            row={'protocol':protocol,'seed':seed,'checkpoint_sha256':sha(path),
                 'offset_derived_without_evaluation_inputs_or_any_targets':True,
                 'train_mean_offset_native_lab':offset.tolist(),
                 'train_constant_shift_energy_fraction':float(np.sum(offset**2)/np.mean(np.sum(differences**2,axis=1))),
                 'train_offset_residual_rms_delta_e76':float(np.sqrt(np.mean(np.sum((differences-offset)**2,axis=1)))),
                 'metrics':{}}
            for method,p in [('graph3',values['evaluation',3]),('graph0',values['evaluation',0]),('graph3_constant_offset',corrected)]:
                e=delta_e00(p,ev['target']);ind=np.array([scalar_de(a,b) for a,b in zip(p,ev['target'],strict=True)])
                cases+=len(e);gap=max(gap,float(np.abs(e-ind).max()))
                row['metrics'][method]={'mean':float(e.mean()),'p95':float(np.quantile(e,.95))}
            records.append(row);del m,x;torch.cuda.empty_cache()
    assert gap<1e-10
    result={'scope':'POST-DISCOVERY source opponent control; fixed offset from fitting predictions only; no independent test claim',
            'script_sha256':sha(Path(__file__)),'source_lock_sha256':sha(OUT/'source_lock.json'),
            'scalar_cases':cases,'maximum_metric_gap':gap,'rows':records}
    target=OUT/'offset_probe.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for protocol in ['mixed','from_SLR','from_ipod']:
        rows=[r for r in records if r['protocol']==protocol]
        print(json.dumps({'protocol':protocol,'mean_constant_energy_fraction':float(np.mean([r['train_constant_shift_energy_fraction'] for r in rows])),
                         'mean_error':{k:float(np.mean([r['metrics'][k]['mean'] for r in rows])) for k in ['graph3','graph0','graph3_constant_offset']}}))


if __name__=='__main__':main()
