"""Independent scalar/loop reconstruction of the fixed-OOF diagnostic."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    started = time.perf_counter()
    lock, result = js(run / 'source_lock.json'), js(run / 'results.json')
    assert args.cache.name == 'train.npz' and sha(args.cache) == CACHE_HASH == lock['cache_sha256']
    assert result['source_lock_sha256'] == sha(run / 'source_lock.json')
    for group in ('sources', 'input_sha256'):
        for rel, h in lock[group].items():
            assert sha(ROOT / rel) == h, rel
    for rel, h in result['files'].items():
        assert sha(run / rel) == h, rel
    with np.load(args.cache, allow_pickle=False) as z:
        target, person, camera = (z[k] for k in ('target', 'patient', 'device'))
    with np.load(run / 'scores.npz', allow_pickle=False) as z:
        scores = {k: z[k] for k in z.files}
    with np.load(run / 'traces.npz', allow_pickle=False) as z:
        traces = {k: z[k] for k in z.files}
    wroot = ROOT / 'experiments/runs/chromaseed_weak_ridge_v1'
    wselect = js(wroot / 'selections.json')
    pselect = js(ROOT / 'experiments/runs/chromaseed_perceptual_v1/selections.json')
    checks = dict(source_bindings=len(lock['sources']), input_bindings=len(lock['input_sha256']),
                  oof_archives=0, candidate_person_values=0, candidate_scores=0, original_choices=0,
                  bootstrap_draws=0, bootstrap_selections=0, deletion_selections=0, paired_distributions=0)
    maxima = dict(person_score_drift=0., paired_gap_drift=0.)
    for ri, (role, (fit, _)) in enumerate(roles(person, camera).items()):
        fit_rows = np.flatnonzero(fit)
        folds = folds_for(person[fit], camera[fit])
        banks, row_chunks = [], []
        for fold in range(3):
            with np.load(wroot / 'inner' / role / f'fold{fold}' / 'oof.npz', allow_pickle=False) as z:
                bank = {k: z[k] for k in z.files}
            query = bank.pop('row_indices')
            np.testing.assert_array_equal(query, fit_rows[folds == fold])
            assert not set(person[query]) & set(person[fit_rows[folds != fold]])
            banks.append(bank)
            row_chunks.append(query)
            checks['oof_archives'] += 1
        rows = np.concatenate(row_chunks)
        people = np.unique(person[rows])
        pcamera = np.array([np.unique(camera[rows][person[rows] == p]).item() for p in people])
        rng = np.random.default_rng(lock['bootstrap_seeds'][ri])
        expected_counts = np.zeros((lock['repeats'], len(people)), dtype=np.int64)
        for label in np.unique(pcamera):
            members = np.flatnonzero(pcamera == label)
            draw = rng.integers(len(members), size=(lock['repeats'], len(members)))
            for i, chosen in enumerate(draw):
                expected_counts[i, members] = np.bincount(chosen, minlength=len(members))
        counts = scores[f'{role}__counts']
        np.testing.assert_array_equal(counts, expected_counts)
        checks['bootstrap_draws'] += len(counts)
        for record in (c for c in result['records'] if c['role'] == role):
            family = record['family']
            configs = record['candidates']
            losses = np.zeros((len(configs), len(people)))
            for ci, config in enumerate(configs):
                f = 'norm_mse' if config['steps'] == 0 else family
                values = []
                for seed in (17, 29, 43):
                    name = f"pred__{f}_s{seed}_w{config['width_index']}_a{config['alpha_index']}_t{config['steps']}"
                    pred = np.concatenate([b[name] for b in banks])
                    errors = delta_e00(pred, target[rows])
                    values.append([sum(errors[person[rows] == p]) / sum(person[rows] == p) for p in people])
                for pi in range(len(people)):
                    losses[ci, pi] = sum(v[pi] for v in values) / 3.
            actual = scores[f'{role}__{family}__losses']
            drift = float(np.max(np.abs(losses - actual)))
            maxima['person_score_drift'] = max(maxima['person_score_drift'], drift)
            assert drift < 1e-10
            checks['candidate_person_values'] += losses.size
            full = [sum(row) / len(people) for row in losses]
            cfgkey = lambda i: (configs[i]['steps'], configs[i]['alpha'], configs[i]['width_factor'])  # noqa: E731
            ranking = sorted(range(len(configs)), key=lambda i: (full[i], *cfgkey(i)))
            original, runner, parent = record['original_index'], record['runner_up_index'], record['parent_index']
            assert ranking[:2] == [original, runner]
            assert abs(record['full_person_mean'] - full[original]) < 1e-10
            for config, computed, stored in zip(configs, full, wselect['roles'][role][family]['candidates'], strict=True):
                assert abs(computed - config['full_person_mean']) < 1e-10
                assert abs(computed - stored['person_mean']) < 1e-10
            assert all(configs[parent][k] == pselect['roles'][role][family]['selected'][k] for k in ('alpha', 'width_factor', 'steps'))
            assert all(configs[original][k] == wselect['roles'][role][family]['selected'][k] for k in ('alpha', 'width_factor', 'steps'))
            checks['candidate_scores'] += len(configs)
            checks['original_choices'] += 1
            order = sorted(range(len(configs)), key=cfgkey)
            tiepos = np.argsort(order)
            prefix = f'{role}__{family}__'
            # Each output row is recalculated by explicit weighted summation, not primary matrix multiplication.
            bw, br, gp, gr = [], [], [], []
            for sample in counts:
                score = (losses * sample[None, :]).sum(axis=1) / len(people)
                winner = min(order, key=lambda i: score[i])
                rank = 1 + sum((score[i] < score[original]) or (score[i] == score[original] and tiepos[i] < tiepos[original]) for i in range(len(configs)))
                bw.append(winner)
                br.append(rank)
                gp.append(score[original] - score[parent])
                gr.append(score[original] - score[runner])
            dw, dr = [], []
            for excluded in range(len(people)):
                score = np.delete(losses, excluded, axis=1).mean(axis=1)
                dw.append(min(order, key=lambda i: score[i]))
                dr.append(1 + sum((score[i] < score[original]) or (score[i] == score[original] and tiepos[i] < tiepos[original]) for i in range(len(configs))))
            for field, computed in [('bootstrap_winners', bw), ('bootstrap_original_ranks', br),
                                    ('deletion_winners', dw), ('deletion_original_ranks', dr)]:
                np.testing.assert_array_equal(traces[prefix + field], computed, err_msg=f'{role}/{family}/{field}')
            for field, computed in [('gap_vs_parent', gp), ('gap_vs_runner_up', gr)]:
                drift = float(np.max(np.abs(traces[prefix + field] - computed)))
                assert drift < 1e-10
                maxima['paired_gap_drift'] = max(maxima['paired_gap_drift'], drift)
            assert record['deletion_switches'] == sum(i != original for i in dw)
            assert record['bootstrap_original_frequency'] == sum(i == original for i in bw) / len(bw)
            assert record['bootstrap_unique_winners'] == len(set(bw))
            assert record['bootstrap_weak_alpha_frequency'] == sum(configs[i]['alpha'] < .1 for i in bw) / len(bw)
            assert record['bootstrap_positive_steps_frequency'] == sum(configs[i]['steps'] > 0 for i in bw) / len(bw)
            for i, config in enumerate(configs):
                assert config['deletion_wins'] == dw.count(i) and config['bootstrap_wins'] == bw.count(i)
            for key, val in [('paired_vs_parent', gp), ('paired_vs_runner_up', gr)]:
                comparator = parent if key == 'paired_vs_parent' else runner
                assert abs(record[key]['mean'] - (full[original] - full[comparator])) < 1e-10
                np.testing.assert_allclose(record[key]['conditional_percentiles_2_5_97_5'], np.quantile(val, [.025, .975]), rtol=0, atol=1e-10)
                assert record[key]['fraction_below_zero'] == sum(v < 0. for v in val) / len(val)
            for key, val in [('bootstrap_original_rank', br), ('deletion_original_rank', dr),
                             ('bootstrap_full_oof_regret', [full[i] - full[original] for i in bw]),
                             ('deletion_full_oof_regret', [full[i] - full[original] for i in dw])]:
                expected = dict(mean=float(np.mean(val)), minimum=min(val), median=float(np.median(val)), p95=float(np.quantile(val, .95)), maximum=max(val))
                for k, v in expected.items():
                    assert abs(record[key][k] - v) < 1e-10
            checks['bootstrap_selections'] += len(counts)
            checks['deletion_selections'] += len(people)
            checks['paired_distributions'] += 2
        print(f'AUDITED {role}: all candidate/person scores and 100000 resample choices', flush=True)
    assert checks['candidate_scores'] == 891 and checks['bootstrap_selections'] == 300000 and checks['deletion_selections'] == 210
    write_json(out / 'audit.json', dict(passed=True, source_lock_sha256=sha(run / 'source_lock.json'),
                                      results_sha256=sha(run / 'results.json'), audit_source_sha256=sha(Path(__file__)),
                                      dependencies={p: sha(ROOT / p) for p in ('scripts/skin_local_search_train.py', 'src/luma_skin_vision/color.py')},
                                      checks=checks, maxima=maxima, elapsed_seconds=time.perf_counter() - started,
                                      limits='Independent aggregation/resampling; shared verified CIEDE2000 formula and fixed role helpers. Not population validation.'))


if __name__ == '__main__':
    main()
