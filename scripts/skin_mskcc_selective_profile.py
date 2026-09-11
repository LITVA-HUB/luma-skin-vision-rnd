"""Read-only profile of the frozen three-model color system and CPU risk heads.

This does not change evaluation code or export a deployment implementation.
"""
import json,time
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,SEEDS,verify_lock,deployment_designs
from skin_mskcc_vote import PatchVotes


def main():
    torch.set_num_threads(4)
    lock=verify_lock(OUT/'final_lock.json',sha(OUT/'final_lock.json'),'final')
    train=load('train');one={k:v[:1] for k,v in train.items()}
    designs=deployment_designs(one,train)['ensemble']
    models=[];files=[]
    for seed in SEEDS:
        path=ROOT/f'experiments/runs/skin_mskcc_pixels_v1/votes_mean_seed{seed}/best.pt'
        state=torch.load(path,weights_only=True,map_location='cpu')
        model=PatchVotes(state['target_std'],0).cuda().eval();model.load_state_dict(state['state'])
        models.append((model,state['target_mean'].cuda(),state['target_std'].cuda()));files.append(path)
    x=torch.from_numpy(one['tokens']).cuda();torch.cuda.empty_cache()
    def color():
        return torch.stack([model(x)*std+mean for model,mean,std in models]).double().mean(0)
    with torch.no_grad():
        actual=color().cpu().numpy()
        np.testing.assert_allclose(actual,designs['prediction'],atol=1e-10,rtol=0)
        for _ in range(40):color()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();gpu=[];wall=[]
        for _ in range(200):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();start.record();color();end.record();end.synchronize()
            wall.append((time.perf_counter()-t)*1000);gpu.append(start.elapsed_time(end))
    record={'scope':'Frozen three-seed color estimator, batch1 prepared64x18 descriptors; no witness extraction, density, risk head, JPEG preprocessing or localization included in GPU timing',
            'device':torch.cuda.get_device_name(),'color_parameters':sum(p.numel() for m,_,_ in models for p in m.parameters()),
            'color_checkpoint_bytes':sum(p.stat().st_size for p in files),
            'gpu_median_ms':float(np.median(gpu)),'gpu_p95_ms':float(np.quantile(gpu,.95)),
            'synchronized_wall_median_ms':float(np.median(wall)),
            'inference_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'source_prediction_replay_max_difference':float(abs(actual-designs['prediction']).max()),
            'cpu_risk_heads':{},'export':'NOT DONE; no new deployment accuracy claim'}
    for arm in ['C_plus','Proposed']:
        key='ensemble__'+arm;path=ROOT/lock['selected'][key]['path'];head=joblib.load(path)
        calpath=ROOT/lock['calibrators'][key]['path'];cal=joblib.load(calpath);times=[]
        with threadpool_limits(4):
            for i in range(120):
                t=time.perf_counter();cal.predict(np.maximum(head.predict(designs[arm]),0))
                if i>=20:times.append((time.perf_counter()-t)*1000)
        record['cpu_risk_heads'][arm]={'selected':lock['selected'][key]['candidate'],
            'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'head_and_calibrator_bytes':path.stat().st_size+calpath.stat().st_size,
            'scope':'Prepared feature vector to predicted error, CPU, batch1; not total pipeline'}
    (OUT/'profile.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__=='__main__':main()
