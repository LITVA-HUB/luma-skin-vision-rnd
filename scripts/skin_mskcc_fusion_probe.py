"""Exploratory fixed50/50fusion of successful CNN and patch models; no fitting."""
import json
import numpy as np
from skin_mskcc_data import ROOT
from skin_mskcc_pixels import load
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00


def main():
    val=load('validation');runs=ROOT/'experiments/runs/skin_mskcc_pixels_v1'
    rows=[{k:val[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(val['target']))]
    records=[];fusions=[]
    for seed in [17,29,43]:
        a=np.load(runs/f'cnn_seed{seed}/best_validation.npz')['prediction'].astype(float)
        b=np.load(runs/f'votes_mean_seed{seed}/best_validation.npz')['prediction'].astype(float)
        p=(a+b)/2;fusions.append(p);e=delta_e00(p,val['target'])
        records.append({'seed':seed,'full':summarize(e,rows),'deployment_models':2,'parameters':1520931+924932})
        np.savez(runs/f'fusion_seed{seed}_validation.npz',prediction=p,target=val['target'])
    p=np.mean(fusions,axis=0);e=delta_e00(p,val['target'])
    rng=np.random.default_rng(20260911);people=np.unique(val['patient']);diff={}
    for base in ['cnn','votes_mean']:
        q=np.load(runs/(base+'_ensemble_validation.npz'))['prediction'];d=e-delta_e00(q,val['target'])
        group=np.array([d[val['patient']==person].mean() for person in people])
        draws=group[rng.integers(0,len(people),size=(2000,len(people)))].mean(1)
        diff[base]={'mean_difference':float(group.mean()),'descriptive_source_bootstrap95':np.quantile(draws,[.025,.975]).tolist()}
    result={'phase':'ADAPTIVE SOURCE EXPLORATION;fixed50/50weights;no learned mixture selection',
            'preregistered_independent_test':False,'single_seed_pairs':records,
            'six_model_ensemble':{'full':summarize(e,rows),'parameters':3*(1520931+924932),'paired_difference':diff},
            'scope':'Six validation people reused for architecture and epoch selection; intervals do not account for selection and are not confirmatory.'}
    out=ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/fusion_probe.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    np.savez(runs/'fusion_ensemble_validation.npz',prediction=p,target=val['target'])
    print(json.dumps(result))


if __name__=='__main__':main()
