"""Synthetic CUDA gate for HR; never loads the real TRAIN or held arrays."""
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from chromaseed_architecture_scale import VARIANTS, capacity
from chromaseed_architecture_scale import fit as original_fit
from chromaseed_head_range import Predictor, predict_torch
from chromaseed_head_range_fit import fit
from chromaseed_refine_train import setup
from skin_local_search_train import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/runs/chromaseed_head_range_v1'


def payload_digest(model):
    return {k: dict(shape=list(v.shape), dtype=str(v.dtype), sha256=hashlib.sha256(v.tobytes()).hexdigest())
            for k, v in model.items()}


def main():
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_chromaseed_architecture_scale import fixture

    old = json.loads((ROOT / 'docs/benchmarks/chromaseed_head_range_cpu_probe/verification.json').read_text())
    for name, digest in {**old['sources'], **old['unchanged_as_sources']}.items():
        assert sha(ROOT / name) == digest, name
    seal = ROOT / 'docs/benchmarks/chromaseed_architecture_scale_v1/verification.json'
    assert sha(seal) == 'ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9'
    assert not (RUN / 'source_lock.json').exists(), 'Production already frozen'
    assert not (RUN / 'preflight.json').exists(), 'Preserve completed preflight'
    setup('cuda')
    protocol = dict(synthetic=True, source_sha256=sha(__file__), adapter_sources=old['sources'],
                    as_verification_sha256=sha(seal), variants=list(VARIANTS),
                    unit_steps=16, unit_checkpoints=[8, 16], changed_head_steps=64,
                    changed_heads=['wide', 'linear'], exact_unit_payloads=True,
                    numpy_cuda_native_lab_atol=0.002, numpy_cuda_rtol=1e-6,
                    peak_allocated_limit_bytes=6_500_000_000, tested_slots=[0, 5],
                    query_rows=[0, 7, 14], gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    path = RUN / 'preflight_protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        write_json(path, protocol)
    x, tokens, y, warm = fixture()
    weights = np.linspace(.25, 2., len(y))
    records = []
    for variant in VARIANTS:
        expected, old_info = original_fit(x, tokens, y, weights, warm, variant, 16, (8, 16), 'cuda', 'cuda_graph')
        actual, unit_info = fit(x, tokens, y, weights, warm, variant, 'unit', 16, (8, 16), 'cuda', 'cuda_graph')
        for step in (8, 16):
            for reference, model in zip(expected[step], actual[step], strict=True):
                assert reference.keys() == model.keys()
                for key in reference:
                    np.testing.assert_array_equal(reference[key], model[key])
        assert unit_info['sampling_sha256'] == old_info['sampling_sha256']
        records.append(dict(variant=variant, mode='unit', original_payloads_exact=12,
                            slot_payload_hashes=[payload_digest(m) for m in actual[16]],
                            peak_bytes=unit_info['cuda_peak_allocated_bytes']))
        print('HR PREFLIGHT UNIT EXACT', variant, flush=True)
        del expected, actual
        gc.collect()
        for mode in ['wide', 'linear']:
            models, info = fit(x, tokens, y, weights, warm, variant, mode, 64, (64,), 'cuda', 'cuda_graph')
            assert info['cuda_peak_allocated_bytes'] < protocol['peak_allocated_limit_bytes']
            assert info['deployed_parameters'] == capacity(variant)
            difference = 0.
            for slot in protocol['tested_slots']:
                model = models[64][slot]
                assert str(model['head_mode']) == mode
                q = protocol['query_rows']
                reference = Predictor(model)(x[q], tokens[q], all_passes=True)
                actual = predict_torch(model, x[q], tokens[q], all_passes=True)
                np.testing.assert_allclose(actual, reference, atol=.002, rtol=1e-6)
                difference = max(difference, float(np.abs(actual-reference).max()))
            records.append(dict(variant=variant, mode=mode, info=info, maximum_lab_difference=difference,
                                slot_payload_hashes=[payload_digest(m) for m in models[64]]))
            print('HR PREFLIGHT', variant, mode, round(info['full_bank_seconds'], 3),
                  'seconds; peak', info['cuda_peak_allocated_bytes'], 'max Lab drift', difference, flush=True)
            del models, actual, reference
            gc.collect()
    result = dict(passed=True, synthetic_only=True, real_training_data_accessed=False,
                  preflight_protocol_sha256=sha(path), source_sha256=sha(__file__), records=records,
                  exact_unit_payloads=sum(r.get('original_payloads_exact', 0) for r in records),
                  maximum_lab_difference=max(r.get('maximum_lab_difference', 0.) for r in records))
    write_json(RUN / 'preflight.json', result)
    print('HR CUDA PREFLIGHT PASSED', sha(RUN / 'preflight.json'), flush=True)


if __name__ == '__main__':
    main()
