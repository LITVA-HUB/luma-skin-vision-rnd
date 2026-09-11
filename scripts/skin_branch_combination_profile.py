"""Batch-one prepared-token latency of all fixed two-model combinations."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_capture_model import CaptureColor
from skin_train_branch_model import TrainingBranchColor
from skin_branch_combination import PAIRS,OUT


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    train=load('train');records=[]
    for name,pair in PAIRS.items():
        models=[];scales=[];digests=[];sizes=0;active=0
        for benchmark,arm in pair:
            path=ROOT/'experiments/runs'/benchmark/f'mixed__{arm}__s17'/'best.pt'
            r=json.loads((ROOT/'docs/benchmarks'/benchmark/path.parent.name/'result.json').read_bytes())
            assert sha(path)==r['best_sha256'];s=torch.load(path,weights_only=True,map_location='cpu')
            m=CaptureColor(arm.split('_')[0]) if benchmark=='skin_capture_v1' else TrainingBranchColor(arm)
            m=m.cuda().eval();m.load_state_dict(s['state']);models.append(m)
            scales.append((s['target_mean'].cuda(),s['target_std'].cuda()));sizes+=path.stat().st_size
            active+=r.get('active_parameters',r['stored_parameters']);digests.append(sha(path))
        x=torch.from_numpy(train['tokens'][:1]).cuda()
        def predict():
            a=models[0](x)[0]*scales[0][1]+scales[0][0]
            b=models[1](x)[0]*scales[1][1]+scales[1][0]
            return .5*(a+b)
        with torch.inference_mode():
            for _ in range(50):p=predict()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(200):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();p=predict();end.record();end.synchronize();times.append(start.elapsed_time(end))
            peak=torch.cuda.max_memory_allocated()/2**20
        assert torch.isfinite(p).all()
        records.append({'method':name,'active_parameters':active,'stored_parameters':sum(q.numel() for m in models for q in m.parameters()),
                        'checkpoint_bytes_total':sizes,'median_gpu_ms':float(np.median(times)),'p95_gpu_ms':float(np.quantile(times,.95)),
                        'peak_allocated_mib':peak,'member_checkpoint_sha256':digests})
        del models,m,scales,x,p;torch.cuda.empty_cache()
    result={'scope':'Batch1 CUDA event; prepared1x64x18tokens to averaged nativeLab; noJPEG/features/CPU/risk/localization',
            'warmup':50,'repeats':200,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'script_sha256':sha(Path(__file__)),'optimized_or_exported':False,'rows':records}
    (OUT/'profile.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
