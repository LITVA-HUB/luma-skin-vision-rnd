"""Mask-induced apparent-color error; never an instrument skin-color benchmark."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, binary_mask
from skin_lapa_prepare import file_sha
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from luma_skin_vision.color import delta_e00, srgb_to_lab

OUT = DATA_ROOT/'facial_skin_v1'


def appearance(rgb,mask):
    rgb,mask = np.asarray(rgb),np.asarray(mask)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or mask.shape != rgb.shape[:2] or mask.dtype != bool:
        raise ValueError('uint8 RGB image and matching binary mask required')
    pixels = rgb[mask].astype(float)/255
    if len(pixels) < 16:
        return None
    return dict(pixels=len(pixels),median_lab=srgb_to_lab(np.median(pixels,axis=0)).tolist(),
                mean_lab=srgb_to_lab(pixels.mean(axis=0)).tolist())


def color_error(reference,prediction):
    if reference is None or prediction is None:
        return None
    return dict(median_delta_e00=float(delta_e00(reference['median_lab'],prediction['median_lab'])),
                mean_delta_e00=float(delta_e00(reference['mean_lab'],prediction['mean_lab'])))


def freeze():
    if (OUT/'appearance_protocol.json').exists():
        raise RuntimeError('appearance protocol exists; preserve it')
    protocol = dict(task='extract visible color from an image by semantic skin mask',
                    reference='author LaPa skin+nose mask on the same192x192 RGB image',
                    primary='CIEDE2000 between channel-wise median encoded RGB colors converted to CIELAB',
                    secondary='CIEDE2000 between mean encoded RGB colors converted to CIELAB',
                    color_convention='assumed sRGB, IEC transfer, D65/CIE1931 2deg; no ICC/camera/illuminant correction',
                    minimum_mask_pixels=16,missing_mask_policy='missing color, report coverage; compare errors also on common coverage',
                    subset='same128 lowest SHA256(LumaSegTest128|stem) as the already fixed held segmentation comparison',
                    selectors='none: fixed final training selection, CNN logit0, frozen UCI threshold',
                    physical_skin_truth=False,neither_native_Lab_nor_phone_tone_accuracy=True,
                    held_results_present_at_freeze=(OUT/'test_results.json').exists(),
                    held_protocol_present_at_freeze=(OUT/'test_protocol.json').exists(),
                    protocol_sha256=file_sha(OUT/'protocol.json'),
                    sources={str(p.relative_to(ROOT)):file_sha(p) for p in [Path(__file__),ROOT/'src/luma_skin_vision/color.py',ROOT/'scripts/skin_face_segment.py']})
    write_json(OUT/'appearance_protocol.json',protocol)
    print('APPEARANCE PROTOCOL FROZEN',protocol['held_results_present_at_freeze'],protocol['held_protocol_present_at_freeze'],flush=True)


def run():
    if json.loads((OUT/'pipeline.json').read_text())['stage'] != 'complete':
        raise RuntimeError('wait for held evaluation and diagnostic pipeline completion')
    if (OUT/'appearance_results.json').exists():
        raise RuntimeError('appearance results exist; preserve them')
    protocol = json.loads((OUT/'appearance_protocol.json').read_text())
    for name,digest in protocol['sources'].items():
        if file_sha(ROOT/name) != digest:
            raise ValueError('appearance source changed after its protocol freeze')
    selected = json.loads((OUT/'selection.json').read_text())
    if selected['model_sha256'] != file_sha(OUT/'best.pt'):
        raise ValueError('selected model checksum changed')
    rows = [r for r in json.loads((DATA_ROOT/'lapa/usable_triplets.json').read_text()) if r['split']=='test']
    subset = sorted(rows,key=lambda r:hashlib.sha256(('LumaSegTest128|'+r['stem']).encode()).digest())[:128]
    held_protocol = json.loads((OUT/'test_protocol.json').read_text())
    assert [r['stem'] for r in subset] == held_protocol['baseline_subset']
    torch.set_num_threads(1)
    model = SkinUNet().eval()
    model.load_state_dict(torch.load(OUT/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    base_root = DATA_ROOT/'skin_pixel_v1'
    base_result = json.loads((base_root/'results.json').read_text())
    raw = (base_root/'selected_local_model.pkl').read_bytes()
    if sha_bytes(raw) != base_result['model_sha256']:
        raise ValueError('own UCI model checksum changed')
    baseline = pickle.loads(raw)
    threshold = json.loads((base_root/'selection.json').read_text())['winner']['threshold']
    records = []
    with torch.inference_mode(),threadpool_limits(limits=1):
        for i,item in enumerate(subset):
            c=item['components']
            for kind in ['images','labels']:
                if file_sha(c[kind]['path']) != c[kind]['sha256']:
                    raise ValueError('original image/label checksum changed')
            with Image.open(c['images']['path']) as im,Image.open(c['labels']['path']) as lab:
                rgb=np.asarray(im.convert('RGB').resize((192,192),Image.Resampling.BILINEAR)).copy()
                label=np.asarray(lab.resize((192,192),Image.Resampling.NEAREST))
            reference=appearance(rgb,binary_mask(label).astype(bool))
            cnn=model(torch.from_numpy(rgb).permute(2,0,1)[None].float()/255)[0,0].numpy() >= 0
            uci=baseline.predict_proba(rgb[...,::-1].reshape(-1,3).astype(float)/255)[:,1].reshape(192,192) >= threshold
            predictions={name:appearance(rgb,mask) for name,mask in [('cnn',cnn),('uci',uci)]}
            records.append(dict(stem=item['stem'],reference=reference,predictions=predictions,
                                errors={name:color_error(reference,value) for name,value in predictions.items()}))
            if (i+1)%32==0:
                print('APPEARANCE',i+1,'of',len(subset),flush=True)
    summary={}
    common=[r for r in records if all(r['errors'][name] is not None for name in ['cnn','uci'])]
    for name in ['cnn','uci']:
        valid=[r['errors'][name] for r in records if r['errors'][name] is not None]
        paired=[r['errors'][name] for r in common]
        summary[name]=dict(valid_colors=len(valid),total_images=len(records),common_coverage_images=len(paired),
                           all_available={key:float(np.mean([r[key] for r in valid])) if valid else None for key in ['median_delta_e00','mean_delta_e00']},
                           common_coverage={key:float(np.mean([r[key] for r in paired])) if paired else None for key in ['median_delta_e00','mean_delta_e00']})
    result=dict(summary=summary,records=records,model_sha256=selected['model_sha256'],
                 appearance_protocol_sha256=file_sha(OUT/'appearance_protocol.json'),
                 physical_truth=False,limitation='image-derived appearance targets only; independent of any instrument skin-color accuracy claim')
    write_json(OUT/'appearance_results.json',result)
    print('APPEARANCE RESULT',json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    if args.run:
        run()
