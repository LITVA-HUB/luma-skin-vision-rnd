"""Frozen-model texture/adapter removal with actual instrument color scoring."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from skin_support_curve import SkinRepresentation,patient_roles,pixel_patches
from skin_support_curve_train import OUT,RUN,bindings,predict
from skin_pair_train import subset,rows,write
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_audit import scalar_de
from luma_skin_vision.color import delta_e00

PROTOCOL=ROOT/'docs/research/skin_support_curve_interventions_v1.md'


def intervene(rgb,kind):
    patches=pixel_patches(rgb);n=len(rgb)
    if kind=='mean':patches=patches.mean((2,3),keepdim=True).expand_as(patches)
    elif kind=='shuffle':
        perm=torch.tensor(np.random.default_rng(51871).permutation(256),device=rgb.device)
        patches=patches.flatten(2)[:,:,perm].reshape(-1,3,16,16)
    else:raise ValueError('Unknown feature intervention')
    return patches.reshape(n,8,8,3,16,16).permute(0,1,4,2,5,3).reshape(n,128,128,3)


def main():
    assert json.loads((OUT/'audit.json').read_bytes())['status']=='PASS'
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    data=load('train');records=sorted(OUT.glob('*/result.json'));assert len(records)==27
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    result=[];cases=0;maximum_gap=0.;provenance=0
    for path in records:
        r=json.loads(path.read_bytes());folder=RUN/path.parent.name;arm=r['arm'];a,b=patient_roles(data['patient'],data['device'],r['train_people'],r['seed'])
        tr,ev=subset(data,a),subset(data,b)
        saved=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');model=SkinRepresentation(arm).cuda();model.load_state_dict(saved['state'])
        mean=saved['target_mean'].cuda();std=saved['target_std'].cuda();original=np.load(folder/'evaluation.npz')['prediction']
        def score(pred,d):
            nonlocal cases,maximum_gap
            e=np.array([scalar_de(x,y) for x,y in zip(pred,d['target'])]);cases+=len(e)
            maximum_gap=max(maximum_gap,float(np.max(abs(e-delta_e00(pred,d['target'])))));assert maximum_gap<1e-10
            return summarize(e,rows(d)),e
        p=predict(model,torch.from_numpy(tr['tokens']).cuda(),torch.from_numpy(tr['rgb']).cuda().float()/255,mean,std)
        train_score,e=score(p,tr);arrays={'train_prediction':p,'train_error':e}
        entry={'name':path.parent.name,'arm':arm,'train_people':r['train_people'],'seed':r['seed'],'train_scores':train_score,'interventions':{}}
        x=torch.from_numpy(ev['tokens']).cuda();rgb=torch.from_numpy(ev['rgb']).cuda().float()/255
        if arm!='baseline':
            model.arm='baseline';p=predict(model,x,rgb,mean,std);model.arm=arm
            s,e=score(p,ev);arrays.update(zero_prediction=p,zero_error=e)
            entry['interventions']['zero']={'scores':s,'prediction_lab_rms_change':float(np.sqrt(np.mean((p-original)**2)))}
        if arm=='pixels':
            model.eval()
            with torch.no_grad():
                residual=model.adapter(pixel_patches(rgb[:2])).cpu().numpy()
            entry['first_two_source_residual_rms']=float(np.sqrt(np.mean(residual**2)))
            for kind in ('shuffle','mean'):
                changed=intervene(rgb,kind)
                before=pixel_patches(rgb);after=pixel_patches(changed)
                if kind=='shuffle':torch.testing.assert_close(before.flatten(2).sort(2).values,after.flatten(2).sort(2).values,rtol=0,atol=0)
                else:torch.testing.assert_close(before.mean((2,3)),after.mean((2,3)),rtol=1e-6,atol=2e-7)
                provenance+=len(before)
                p=predict(model,x,changed,mean,std);s,e=score(p,ev);arrays[kind+'_prediction']=p;arrays[kind+'_error']=e
                entry['interventions'][kind]={'scores':s,'prediction_lab_rms_change':float(np.sqrt(np.mean((p-original)**2)))}
        np.savez(folder/'interventions.npz',**arrays);entry['array_sha256']=sha(folder/'interventions.npz');result.append(entry)
        del model,x,rgb;torch.cuda.empty_cache()
    record={'status':'PASS','scope':'post-fit TRAIN/internal-holdout diagnostics; no new training or selection','models':result,
        'training_prediction_arrays':27,'intervention_prediction_arrays':36,'independent_scalar_color_cases':cases,
        'actual_patch_intervention_provenance_cases':provenance,'maximum_color_gap':maximum_gap,'reserved_endpoint_access':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PROTOCOL,ROOT/'tests/test_skin_support_curve_interventions.py',OUT/'audit.json']}}
    write(OUT/'interventions.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('models','bindings')}))
    for n in (6,12,18):
        for arm in ('baseline','statistics','pixels'):
            rr=[x for x in result if x['train_people']==n and x['arm']==arm]
            print(json.dumps({'n':n,'arm':arm,'train_mean':float(np.mean([x['train_scores']['mean'] for x in rr])),
                'interventions':{k:float(np.mean([x['interventions'][k]['scores']['mean'] for x in rr])) for k in rr[0]['interventions']},
                'lab_rms_changes':{k:float(np.mean([x['interventions'][k]['prediction_lab_rms_change'] for x in rr])) for k in rr[0]['interventions']}}))


if __name__=='__main__':main()
