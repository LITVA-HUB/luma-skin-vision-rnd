"""Profile seed17 single models separately from verified original-JPEG input work."""
import json
import time
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from skin_mskcc_data import ROOT,RAW,manifest,sha
from skin_mskcc_pixels import load,decode
from skin_mskcc_vote import PatchVotes


def main():
    torch.set_num_threads(4);data=load('train');results=[]
    for arch in ['cnn','votes_mean']:
        path=ROOT/f'experiments/runs/skin_mskcc_pixels_v1/{arch}_seed17/best.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=(mobilenet_v3_small(weights=None,num_classes=3) if arch=='cnn' else PatchVotes(state['target_std'],0)).cuda().eval()
        model.load_state_dict(state['state']);mean,std=state['target_mean'].cuda(),state['target_std'].cuda()
        x=(torch.from_numpy(data['rgb'][:1].transpose(0,3,1,2).copy()).cuda().float()/255 if arch=='cnn' else torch.from_numpy(data['tokens'][:1]).cuda())
        with torch.no_grad():
            for _ in range(40):out=model(x)*std+mean
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();gpu=[];wall=[]
            for _ in range(200):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                before=time.perf_counter();start.record();out=model(x)*std+mean;end.record();end.synchronize()
                wall.append((time.perf_counter()-before)*1000);gpu.append(start.elapsed_time(end))
        results.append({'arch':arch,'seed':17,'device':torch.cuda.get_device_name(),'batch':1,
                        'model_input':'128RGBtensor' if arch=='cnn' else '64x18precomputed patch descriptors',
                        'gpu_median_ms':float(np.median(gpu)),'gpu_p95_ms':float(np.quantile(gpu,.95)),
                        'synchronized_wall_median_ms':float(np.median(wall)),
                        'inference_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':sha(path)})
        del model,x,mean,std,out;torch.cuda.empty_cache()
    expected={r['image']:r for r in json.loads((RAW/'image_receipts.json').read_bytes())}
    rows=manifest()['rows'];chosen=[]
    for camera in ['SLR','ipod']:
        chosen += [r for r in rows if r['role']=='train' and r['device']==camera][:5]
    times=[]
    for row in chosen:
        before=time.perf_counter();decode(row,expected);times.append((time.perf_counter()-before)*1000)
    result={'model_profiles':results,'source_original_jpeg_preparation':{'n':len(chosen),'median_ms':float(np.median(times)),
            'p95_ms':float(np.quantile(times,.95)),'includes':'file read+SHA256+JPEG decode+EXIF transpose+central crop+128resize+all color/histogram/patch descriptors',
            'cache_state':'local files may be OS-cached;not cold-storage benchmark','excludes':'skin/face localization;this dataset already supplies close skin crops'},
            'deployment_export':'NOT DONE;source screen only','calibration_or_test_opened':False}
    (ROOT/'docs/benchmarks/skin_mskcc_pixels_v1/profile.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
