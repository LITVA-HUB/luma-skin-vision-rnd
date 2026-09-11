"""All frozen local-image color comparators, including negative source arms."""
import joblib
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from skin_mskcc_data import ROOT
from skin_mskcc_vote import PatchVotes
from skin_mskcc_vote_v2 import GlobalColorMLP,VoteAblation
from skin_mskcc_train_pixels import predict

ARCHES=['cnn','votes_mean','votes_huber3','global_mlp','shared_tight','local_mean','local_tight']


def checkpoint(arch,seed):
    phase='skin_mskcc_pixels_v1' if arch in ARCHES[:3] else 'skin_mskcc_pixel_ablation_v2'
    return ROOT/f'experiments/runs/{phase}/{arch}_seed{seed}/best.pt'


def paths():
    return ([checkpoint(a,s) for a in ARCHES for s in [17,29,43]]+
            sorted((ROOT/'experiments/runs/skin_mskcc_pixel_controls_v1').glob('*.joblib')))


def all_predictions(data):
    output={}
    for p in sorted((ROOT/'experiments/runs/skin_mskcc_pixel_controls_v1').glob('*.joblib')):
        feature=p.stem.rsplit('_',1)[0];output['control_'+p.stem]=joblib.load(p).predict(data[feature])
    for arch in ARCHES:
        for seed in [17,29,43]:
            state=torch.load(checkpoint(arch,seed),weights_only=True,map_location='cpu')
            if arch=='cnn':model=mobilenet_v3_small(weights=None,num_classes=3)
            elif arch in ['votes_mean','votes_huber3']:model=PatchVotes(state['target_std'],0 if arch=='votes_mean' else 3)
            elif arch=='global_mlp':model=GlobalColorMLP()
            else:model=VoteAblation(state['target_std'],local_only=arch.startswith('local_'),steps=0 if arch=='local_mean' else 3)
            model=model.cuda();model.load_state_dict(state['state'])
            x=(torch.from_numpy(data['rgb'].transpose(0,3,1,2).copy()).float().cuda()/255 if arch=='cnn'
               else torch.from_numpy(data['color' if arch=='global_mlp' else 'tokens']).cuda())
            p,_=predict(model,x,state['target_mean'].cuda(),state['target_std'].cuda(),arch)
            output[f'{arch}_s{seed}']=p.astype(np.float64)
            del model,x
        output[arch+'_ensemble']=np.mean([output[f'{arch}_s{s}'] for s in [17,29,43]],axis=0)
    for seed in [17,29,43]:output[f'fusion_s{seed}']=(output[f'cnn_s{seed}']+output[f'votes_mean_s{seed}'])/2
    output['fusion_ensemble']=np.mean([output[f'fusion_s{s}'] for s in [17,29,43]],axis=0)
    if len(output)!=41 or not all(np.isfinite(p).all() for p in output.values()):raise ValueError('Invalid/incomplete comparator registry')
    return output
