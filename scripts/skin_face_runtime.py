"""Measured CPU inference latency after training/testing; includes a JPEG-to-mask path."""
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet
from skin_lapa_prepare import file_sha


def describe(samples):
    values = np.asarray(samples)*1000
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('invalid timing observations')
    return dict(repetitions=len(values),median_ms=float(np.median(values)),
                p95_ms=float(np.quantile(values,.95)),minimum_ms=float(values.min()))


def tensor_from_rgb(rgb,layout):
    x = torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1)[None].float()/255
    return x.contiguous(memory_format=layout)


def main():
    root = DATA_ROOT/'facial_skin_v1'
    pipeline = json.loads((root/'pipeline.json').read_text())
    if pipeline['stage'] != 'complete':
        raise RuntimeError('wait for the queued held evaluation and diagnostics to complete')
    if (root/'runtime.json').exists():
        raise RuntimeError('completed runtime measurements exist; preserve them')
    result = json.loads((root/'test_results.json').read_text())
    if file_sha(root/'best.pt') != result['model_sha256']:
        raise ValueError('selected model checksum changed')
    data = DATA_ROOT/'lapa'/'prepared_192'
    profile = json.loads((data/'profile.json').read_text())
    if file_sha(data/'val_rgb.npy') != profile['splits']['val']['rgb_sha256']:
        raise ValueError('validation input checksum changed')
    val = np.load(data/'val_rgb.npy',mmap_mode='r')
    # Fixed evenly spaced validation inputs, selected without any timing observation.
    ids = np.linspace(0,len(val)-1,16,dtype=int)
    private = DATA_ROOT/'private_user_faces'
    manifest = json.loads((private/'manifest.json').read_text(encoding='utf-8'))
    originals = [Path(item['path']).read_bytes() for item in manifest['records']]
    if any(sha_bytes(raw) != item['sha256'] for raw,item in zip(originals,manifest['records'],strict=True)):
        raise ValueError('supplied original checksum changed')
    protocol = dict(model_sha256=result['model_sha256'],validation_indices=ids.tolist(),
                     cpu_threads=[1,2,4],layouts=['contiguous','channels_last'],
                     model_only_repetitions=48,jpeg_repetitions_per_photo=3,
                     warmup_calls=8,logit_parity_absolute_tolerance=.001,
                     task='one192x192 facial-skin mask; JPEG bytes already in RAM; no disk/network/skin-color head timing',
                     jpeg_inputs=[dict(sha256=item['sha256'],width=item['width'],height=item['height']) for item in manifest['records']],
                     source_sha256=sha_bytes(Path(__file__).read_bytes()),
                     machine=platform.processor(),torch_version=str(torch.__version__),
                     data_profile_sha256=file_sha(data/'profile.json'))
    write_json(root/'runtime_protocol.json',protocol)
    state = torch.load(root/'best.pt',map_location='cpu',weights_only=True)['state_dict']
    torch.set_num_threads(1)
    base = SkinUNet().eval()
    base.load_state_dict(state)
    with torch.inference_mode():
        reference = [base(tensor_from_rgb(val[i],torch.contiguous_format)).numpy().copy() for i in ids]
    records = []
    with torch.inference_mode():
        for threads in protocol['cpu_threads']:
            torch.set_num_threads(threads)
            for label,layout in [('contiguous',torch.contiguous_format),('channels_last',torch.channels_last)]:
                model = SkinUNet().eval().to(memory_format=layout)
                model.load_state_dict(state)
                inputs = [tensor_from_rgb(val[i],layout) for i in ids]
                max_error,changed = 0.,0
                for x,expected in zip(inputs,reference,strict=True):
                    output = model(x).numpy()
                    max_error = max(max_error,float(np.max(np.abs(output-expected))))
                    changed += int(np.sum((output >= 0) != (expected >= 0)))
                if max_error > protocol['logit_parity_absolute_tolerance']:
                    raise ValueError('CPU layout/thread numerical drift exceeded the prospective bound')
                for i in range(protocol['warmup_calls']):
                    model(inputs[i % len(inputs)])
                model_times,jpeg_times = [],[]
                for i in range(protocol['model_only_repetitions']):
                    begin = time.perf_counter()
                    _ = model(inputs[i % len(inputs)])
                    model_times.append(time.perf_counter()-begin)
                for _ in range(protocol['jpeg_repetitions_per_photo']):
                    for raw in originals:
                        begin = time.perf_counter()
                        with Image.open(io.BytesIO(raw)) as im:
                            rgb = im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR)
                        mask = (model(tensor_from_rgb(rgb,layout))[0,0] >= 0).numpy()
                        if mask.shape != (192,192):
                            raise ValueError('incorrect mask size')
                        jpeg_times.append(time.perf_counter()-begin)
                row = dict(threads=threads,layout=label,model_only=describe(model_times),
                           jpeg_to_mask=describe(jpeg_times),max_logit_difference=max_error,
                           differing_mask_pixels=changed,comparison_pixels=int(np.prod(reference[0].shape)*len(reference)))
                records.append(row)
                print('SEG CPU TIMING',json.dumps(row),flush=True)
    write_json(root/'runtime.json',dict(records=records,model_sha256=result['model_sha256'],
                   protocol_sha256=file_sha(root/'runtime_protocol.json'),
                   measurements='actual calls on this host; small sample, no cross-device latency claim',
                   numeric_float32_parameter_bytes=result['parameters']*4))


if __name__ == '__main__':
    main()
