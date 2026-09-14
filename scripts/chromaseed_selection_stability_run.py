"""Registered diagnostic: no model fitting, no outer prediction/result access."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from chromaseed_selection_stability import bootstrap_counts, diagnose, person_losses
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
REPEATS = 20_000
BOOT_SEEDS = (2026091301, 2026091302, 2026091303)
MODEL_SEEDS = (17, 29, 43)


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lock_inputs(run, parent, previous, cache):
    if cache.name != 'train.npz' or sha(cache) != CACHE_HASH:
        raise ValueError('only original TRAIN permitted')
    inherited = js(parent / 'source_lock.json')
    for rel, expected in inherited['sources'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f'frozen source changed: {rel}')
    paths = [parent / 'source_lock.json', parent / 'selections.json', previous / 'source_lock.json', previous / 'selections.json',
             ROOT / 'docs/benchmarks/chromaseed_weak_ridge_v1/verification.json']
    for directory in sorted((parent / 'inner').glob('*/fold*')):
        receipt = js(directory / 'receipt.json')
        if receipt['source_lock_sha256'] != sha(parent / 'source_lock.json') or receipt['files']['oof.npz'] != sha(directory / 'oof.npz'):
            raise ValueError('OOF source binding mismatch')
        paths.extend([directory / 'receipt.json', directory / 'oof.npz'])
    own = ['scripts/chromaseed_selection_stability.py', 'scripts/chromaseed_selection_stability_run.py',
           'tests/test_chromaseed_selection_stability.py', 'docs/research/chromaseed_selection_stability_v1_protocol.md']
    sources = dict(inherited['sources'])
    sources.update({rel: sha(ROOT / rel) for rel in own})
    lock = dict(sources=sources, cache_sha256=CACHE_HASH,
                input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                repeats=REPEATS, bootstrap_seeds=list(BOOT_SEEDS), numpy=np.__version__, threads=1,
                loaded_cache_keys=['target', 'patient', 'device'], purpose='fixed-OOF conditional sensitivity; no new model')
    path = run / 'source_lock.json'
    if path.exists() and js(path) != lock:
        raise ValueError('diagnostic lock changed')
    if not path.exists():
        write_json(path, lock)
    return sha(path)


def name_for(config, seed):
    family = 'norm_mse' if config['steps'] == 0 else config['family']
    return f"pred__{family}_s{seed}_w{config['width_index']}_a{config['alpha_index']}_t{config['steps']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=ROOT / 'experiments/runs/chromaseed_weak_ridge_v1')
    parser.add_argument('--previous', type=Path, default=ROOT / 'experiments/runs/chromaseed_perceptual_v1')
    args = parser.parse_args()
    run, parent, previous = args.run.resolve(), args.parent.resolve(), args.previous.resolve()
    started = time.perf_counter()
    lock = lock_inputs(run, parent, previous, args.cache)
    if (run / 'results.json').exists():
        raise ValueError('completed results already exist; use a fresh directory')
    write_json(run / 'progress.json', dict(status='diagnosing', pid=os.getpid(), updated_unix=time.time()))
    with np.load(args.cache, allow_pickle=False) as data:
        target, person, device = (data[k] for k in ('target', 'patient', 'device'))
    selected, old = js(parent / 'selections.json'), js(previous / 'selections.json')
    assert selected['source_lock_sha256'] == sha(parent / 'source_lock.json')
    assert old['source_lock_sha256'] == sha(previous / 'source_lock.json')
    records, arrays, traces = [], {}, {}
    checks = dict(roles=0, oof_archives=0, candidate_scores=0, original_choices=0, deletion_selections=0, bootstrap_selections=0)
    max_score_drift = 0.
    for ri, (role, (fit, _)) in enumerate(roles(person, device).items()):
        fit_rows = np.flatnonzero(fit)
        folds = folds_for(person[fit], device[fit])
        banks, row_chunks = [], []
        for fold in range(3):
            with np.load(parent / 'inner' / role / f'fold{fold}' / 'oof.npz', allow_pickle=False) as z:
                bank = {k: z[k] for k in z.files}
            rows = bank.pop('row_indices')
            np.testing.assert_array_equal(rows, fit_rows[folds == fold])
            assert not set(person[rows]) & set(person[fit_rows[folds != fold]])
            banks.append(bank)
            row_chunks.append(rows)
            checks['oof_archives'] += 1
        rows = np.concatenate(row_chunks)
        people = np.unique(person[rows])
        cameras = []
        for p in people:
            camera = np.unique(device[rows][person[rows] == p])
            assert len(camera) == 1
            cameras.append(camera[0])
        counts = bootstrap_counts(np.array(cameras), REPEATS, BOOT_SEEDS[ri])
        arrays[f'{role}__counts'] = counts
        # Ordinal axes only: no IDs, row indices, colors or camera-to-person mapping exported.
        for family, choice in selected['roles'][role].items():
            configs = [{k: c[k] for k in ('family', 'width_index', 'alpha_index', 'steps', 'width_factor', 'alpha')} for c in choice['candidates']]
            losses = []
            for c, expected in zip(configs, choice['candidates'], strict=True):
                pred = np.stack([np.concatenate([b[name_for(c, s)] for b in banks]) for s in MODEL_SEEDS])
                value = person_losses(pred, target[rows], person[rows])
                drift = abs(float(value.mean()) - expected['person_mean'])
                assert drift < 1e-10
                max_score_drift = max(max_score_drift, drift)
                losses.append(value)
                checks['candidate_scores'] += 1
            losses = np.array(losses)
            pc = old['roles'][role][family]['selected']
            parent_index = next(i for i, c in enumerate(configs) if all(c[k] == pc[k] for k in ('steps', 'alpha', 'width_factor')))
            result, trace = diagnose(losses, configs, counts, parent_index)
            oc = configs[result['original_index']]
            assert all(oc[k] == choice['selected'][k] for k in ('steps', 'alpha', 'width_factor'))
            checks['original_choices'] += 1
            checks['deletion_selections'] += len(people)
            checks['bootstrap_selections'] += REPEATS
            result.update(role=role, family=family, camera_people={str(c): cameras.count(c) for c in np.unique(cameras)})
            records.append(result)
            arrays[f'{role}__{family}__losses'] = losses
            traces.update({f'{role}__{family}__{k}': v for k, v in trace.items()})
        checks['roles'] += 1
        print(f'DIAGNOSED {role}: {len(people)} people, five families, {REPEATS} shared camera-stratified draws', flush=True)
    assert checks == dict(roles=3, oof_archives=9, candidate_scores=891, original_choices=15, deletion_selections=210, bootstrap_selections=300000)
    np.savez_compressed(run / 'scores.npz', **arrays)
    np.savez_compressed(run / 'traces.npz', **traces)
    write_json(run / 'results.json', dict(source_lock_sha256=lock, records=records, checks=checks, max_score_drift=max_score_drift,
                                        files={n: sha(run / n) for n in ('scores.npz', 'traces.npz')},
                                        elapsed_seconds=time.perf_counter() - started,
                                        evidence='Conditional score sensitivity on reused TRAIN; no new fitting or outer evaluation'))
    write_json(run / 'progress.json', dict(status='diagnostic_complete_audit_pending', pid=os.getpid(), updated_unix=time.time()))


if __name__ == '__main__':
    main()
