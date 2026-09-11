"""Frozen spatially corresponding skin teacher tokens; source caches only."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels
from skin_teacher_features import teacher, preprocess, TEACHER

OUT = ROOT/'docs/benchmarks/skin_local_teacher_v1'
CACHE = ROOT/'data/processed/skin_local_teacher_v1'
PROTOCOL = ROOT/'docs/research/skin_local_teacher_protocol_v1.md'


def pool_tokens(tokens):
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 384):
        raise ValueError('Expected original 16x16 teacher token grid')
    return tokens.reshape(-1, 8, 2, 8, 2, 384).mean((2, 4)).reshape(-1, 64, 384)


def image_permutation(identifier):
    seed = int.from_bytes(hashlib.sha256(('LumaLocalTeacher1|'+str(identifier)).encode()).digest()[:8], 'little')
    return np.random.default_rng(seed).permutation(64).astype(np.uint8)


def extract(model, rgb, verify_pooling=False):
    chunks = []; gap = 0.; cells = 0
    with torch.no_grad():
        for start in range(0, len(rgb), 32):
            tokens = model.forward_features(preprocess(rgb[start:start+32]).cuda())['x_norm_patchtokens']
            value = pool_tokens(tokens).cpu().numpy()
            if verify_pooling and start == 0:
                original = tokens[:16].cpu().numpy().reshape(-1, 16, 16, 384).astype(np.float64)
                manual = np.stack([np.stack([g[2*r:2*r+2, 2*c:2*c+2].mean((0, 1)) for r in range(8) for c in range(8)]) for g in original])
                gap = float(np.max(np.abs(manual-value[:len(manual)]))); cells += len(manual)*64
                # Four-term FP32 sum, exact power-of-two division: gamma_3*sum(abs)/4.
                # A fixed absolute epsilon incorrectly rejects larger valid descriptors.
                u = np.finfo(np.float32).eps/2
                absolute_sum = np.abs(original).reshape(-1, 8, 2, 8, 2, 384).sum((2, 4)).reshape(-1, 64, 384)
                bound = (3*u/(1-3*u))*absolute_sum/4
                assert np.all(np.abs(manual-value[:len(manual)]) <= bound+1e-12)
            assert value.shape[1:] == (64, 384) and np.isfinite(value).all(); chunks.append(value)
    return np.concatenate(chunks), gap, cells


def load(role):
    if role not in {'train', 'validation'}: raise ValueError('Source-only local teacher')
    r = json.loads((OUT/'feature_receipt.json').read_bytes())
    for name, digest in r['bindings'].items():
        if sha(ROOT/name) != digest: raise ValueError('Local teacher source changed')
    path = CACHE/(role+'.npz')
    if sha(path) != r['roles'][role]['sha256']: raise ValueError('Local teacher cache changed')
    data = load_pixels(role)
    with np.load(path) as saved:
        np.testing.assert_array_equal(data['image'], saved['image'])
        data['teacher'] = saved['teacher']; data['permutation'] = saved['permutation']
    return data


def pack(data, arm):
    if arm == 'plain': return data['tokens'].copy()
    values = data['teacher']
    if arm == 'shuffled':
        values = values[np.arange(len(values))[:, None], data['permutation']]
    return np.concatenate([data['tokens'], values], axis=-1).astype(np.float32)


def normalization(data):
    x = data['teacher'].astype(np.float64)
    mean = x.mean((0, 1)); std = x.std((0, 1)); std = np.where(std < 1e-8, 1., std)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['extract', 'verify']); args = parser.parse_args()
    model = teacher(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if args.stage == 'extract':
        CACHE.mkdir(parents=True, exist_ok=False); OUT.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), PROTOCOL, ROOT/'tests/test_skin_local_teacher.py',
                 ROOT/'scripts/skin_teacher_features.py', ROOT/'scripts/skin_mskcc_pixels.py',
                 ROOT/'docs/benchmarks/skin_teacher_readout_v1/feature_receipt.json',
                 TEACHER/'dinov2_vits14_pretrain.pth']
        files += [ROOT/'data/processed/skin_mskcc_pixels_v1'/(role+'.npz') for role in ['train', 'validation']]
        receipt = {'bindings': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'roles': {},
                   'teacher_parameters': sum(p.numel() for p in model.parameters()), 'reserved_endpoints_used': False}
        for role in ['train', 'validation']:
            data = load_pixels(role); value, _, _ = extract(model, data['rgb'])
            path = CACHE/(role+'.npz')
            permutations = np.stack([image_permutation(i) for i in data['image']])
            np.savez(path, teacher=value, permutation=permutations, image=data['image'])
            receipt['roles'][role] = {'n': len(value), 'shape': list(value.shape), 'bytes': path.stat().st_size, 'sha256': sha(path)}
            print(json.dumps({'role': role, **receipt['roles'][role]}), flush=True)
        receipt['seconds'] = time.perf_counter()-started
        receipt['peak_extraction_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        (OUT/'feature_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    else:
        n = 0; max_gap = 0.; cells = 0
        for role in ['train', 'validation']:
            data = load(role); value, gap, count = extract(model, data['rgb'], verify_pooling=True)
            np.testing.assert_array_equal(value, data['teacher'])
            for identity, permutation in zip(data['image'], data['permutation'], strict=True):
                np.testing.assert_array_equal(permutation, image_permutation(identity))
                np.testing.assert_array_equal(np.sort(permutation), np.arange(64))
            n += len(value); max_gap = max(max_gap, gap); cells += count
        result = {'status': 'PASS', 'exact_image_token_sets': n, 'tokens_per_image': 64,
                  'manual_fp64_pooled_cells': cells, 'maximum_manual_pooling_gap': max_gap,
                  'per_image_permutations_checked': n, 'reserved_endpoints_used': False,
                  'feature_receipt_sha256': sha(OUT/'feature_receipt.json')}
        (OUT/'feature_audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8'); print(json.dumps(result))


if __name__ == '__main__': main()
