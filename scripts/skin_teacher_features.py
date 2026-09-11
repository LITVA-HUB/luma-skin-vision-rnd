"""Immutable source-only DINOv2 skin features; no online loading or fitting."""
import os
os.environ.setdefault('XFORMERS_DISABLED', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels

OUT = ROOT/'docs/benchmarks/skin_teacher_readout_v1'
CACHE = ROOT/'data/processed/skin_teacher_readout_v1'
PROTOCOL = ROOT/'docs/research/skin_teacher_readout_protocol_v1.md'
TEACHER = ROOT/'artifacts/teachers/dinov2-7764ea0f912e'
WEIGHT_SHA = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def preprocess(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (128, 128, 3):
        raise ValueError('Expected uint8 source skin RGB at 128x128')
    x = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float()/255
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return (x-mean)/std


def teacher():
    acquisition = json.loads((TEACHER/'acquisition.json').read_bytes())
    for name, record in acquisition['files'].items():
        if sha(TEACHER/'source'/name) != record['sha256']:
            raise ValueError('Teacher source changed')
    weights = TEACHER/'dinov2_vits14_pretrain.pth'
    if sha(weights) != WEIGHT_SHA:
        raise ValueError('Teacher weight changed')
    sys.path.insert(0, str(TEACHER/'source'))
    from dinov2.hub.backbones import dinov2_vits14
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model.eval().requires_grad_(False).cuda()


def extract(model, rgb):
    parts = []
    with torch.no_grad():
        for start in range(0, len(rgb), 32):
            output = model.forward_features(preprocess(rgb[start:start+32]).cuda())
            cls = output['x_norm_clstoken']; patch = output['x_norm_patchtokens']
            assert cls.shape[1:] == (384,) and patch.shape[1:] == (256, 384)
            value = torch.cat([cls, patch.mean(1)], dim=1).cpu().numpy()
            assert np.isfinite(value).all()
            parts.append(value)
    return np.concatenate(parts)


def load(role):
    if role not in {'train', 'validation'}:
        raise ValueError('Source-only teacher cache')
    receipt = json.loads((OUT/'feature_receipt.json').read_bytes())
    for name, digest in receipt['bindings'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Feature source binding changed: '+name)
    path = CACHE/(role+'.npz')
    if sha(path) != receipt['roles'][role]['sha256']:
        raise ValueError('Feature cache changed')
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['extract', 'verify'])
    args = parser.parse_args()
    model = teacher(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if args.stage == 'extract':
        CACHE.mkdir(parents=True, exist_ok=False); OUT.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), PROTOCOL, ROOT/'tests/test_skin_teacher_features.py',
                 ROOT/'scripts/skin_mskcc_pixels.py', TEACHER/'acquisition.json',
                 TEACHER/'source_manifest.json', TEACHER/'dinov2_vits14_pretrain.pth']
        files += [ROOT/'data/processed/skin_mskcc_pixels_v1'/(r+'.npz') for r in ['train', 'validation']]
        receipt = {'bindings': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'roles': {},
                   'teacher_parameters': sum(p.numel() for p in model.parameters()),
                   'teacher_weight_bytes': (TEACHER/'dinov2_vits14_pretrain.pth').stat().st_size,
                   'scope': 'SOURCE FEATURES ONLY; no fine-tuning or reserved endpoints',
                   'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'batch': 32}
        for role in ['train', 'validation']:
            data = load_pixels(role); value = extract(model, data['rgb'])
            path = CACHE/(role+'.npz')
            np.savez(path, teacher=value, image=data['image'])
            receipt['roles'][role] = {'n': len(value), 'features': value.shape[1],
                                    'sha256': sha(path), 'bytes': path.stat().st_size}
            print(json.dumps({'role': role, **receipt['roles'][role]}), flush=True)
        receipt['seconds'] = time.perf_counter()-started
        receipt['peak_extraction_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        (OUT/'feature_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
        print(json.dumps({'seconds': receipt['seconds'], 'peak_mib': receipt['peak_extraction_allocated_mib']}))
    else:
        n = 0
        for role in ['train', 'validation']:
            data = load_pixels(role); saved = load(role)
            np.testing.assert_array_equal(saved['image'], data['image'])
            value = extract(model, data['rgb'])
            np.testing.assert_array_equal(value, saved['teacher']); n += len(value)
        record = {'status': 'PASS', 'exact_feature_vectors': n, 'features': 768,
                  'same_device_and_batch_replay': True, 'reserved_endpoints_used': False,
                  'feature_receipt_sha256': sha(OUT/'feature_receipt.json'),
                  'seconds': time.perf_counter()-started}
        (OUT/'feature_audit.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
