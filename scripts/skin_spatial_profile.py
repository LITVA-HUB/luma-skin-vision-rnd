"""Prepared-token GPU model latency only; no claim of complete photo latency."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_spatial_model import SpatialColor,ARMS
from skin_spatial_train import OUT,RUN


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    train=load('train');rows=[]
    for arm in ARMS:
        path=RUN/f'mixed__{arm}__s17'/'best.pt'
        result=json.loads((OUT/f'mixed__{arm}__s17'/'result.json').read_bytes());assert sha(path)==result['best_sha256']
        saved=torch.load(path,weights_only=True,map_location='cpu');model=SpatialColor(arm).cuda().eval();model.load_state_dict(saved['state'])
        x=torch.from_numpy(train['tokens'][:1]).cuda();mean=saved['target_mean'].cuda();std=saved['target_std'].cuda()
        with torch.inference_mode():
            for _ in range(50):p=model(x)[0]*std+mean
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(200):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();p=model(x)[0]*std+mean;end.record();end.synchronize();times.append(start.elapsed_time(end))
            peak=torch.cuda.max_memory_allocated()/2**20
        assert torch.isfinite(p).all()
        rows.append({'arm':arm,'seed':17,'batch':1,'input_shape':[1,64,18],'median_gpu_ms':float(np.median(times)),
                     'p95_gpu_ms':float(np.quantile(times,.95)),'peak_allocated_mib':peak,
                     'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':sha(path),
                     'stored_parameters':sum(p.numel() for p in model.parameters()),'active_parameters':model.active_parameters()})
        del model,x,mean,std,p;torch.cuda.empty_cache()
    report={'scope':'CUDA event batch1 prepared-token nativeLab output; excludes JPEG/resize/features/CPU/risk/localization',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'warmup_iterations':50,'measured_iterations':200,'exported':False,'optimized':False,
            'script_sha256':sha(Path(__file__)),'rows':rows}
    (OUT/'profile.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':main()
