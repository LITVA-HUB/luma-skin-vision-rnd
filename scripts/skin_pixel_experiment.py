"""Fixed UCI skin-pixel classifier screen; not a skin-color accuracy benchmark."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits


def metrics(y, probability, threshold):
    y, p = np.asarray(y), np.asarray(probability)
    if y.shape != p.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('invalid binary targets')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not 0 <= threshold <= 1:
        raise ValueError('invalid probabilities or threshold')
    pred = p >= threshold
    tp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
    fp, tn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
    recall, specificity = tp/max(1, tp+fn), tn/max(1, tn+fp)
    return dict(rows=len(y), threshold=float(threshold), tp=tp, fn=fn, fp=fp, tn=tn,
                accuracy=(tp+tn)/len(y), balanced_accuracy=(recall+specificity)/2,
                skin_recall=recall, nonskin_recall=specificity, skin_precision=tp/max(1, tp+fp),
                f1=2*tp/max(1, 2*tp+fp+fn), auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None)


def choose(candidates):
    return max(candidates, key=lambda r: (r['validation']['balanced_accuracy'], r['id']))


def main():
    source = DATA_ROOT / 'uci_skin_segmentation'
    out = DATA_ROOT / 'skin_pixel_v1'
    if (out / 'results.json').exists():
        raise RuntimeError('completed experiment exists; preserve it')
    profile = json.loads((source / 'profile.json').read_text())
    raw = (source / 'grouped_pixels.npz').read_bytes()
    assert sha_bytes(raw) == profile['arrays_sha256']
    with np.load(source / 'grouped_pixels.npz', allow_pickle=False) as a:
        x = a['bgr'].astype(np.float64)/255
        y, part = a['skin'], a['partition']
    train, val, test = part == 0, part == 1, part == 2
    configs = [dict(id='logistic_rgb', kind='logistic', C=1.0)] + [
        dict(id=f'hist_l{leaves}_n{iterations}', kind='histogram', max_leaf_nodes=leaves,
             max_iter=iterations, learning_rate=.1, l2_regularization=1.0, min_samples_leaf=40)
        for leaves in (7, 15) for iterations in (100, 200)]
    protocol = dict(dataset_sha256=profile['arrays_sha256'], partition_salt=profile['salt'],
                    configs=configs, thresholds=[i/10 for i in range(1, 10)], random_seed=20260914,
                    selection='maximum validation balanced accuracy; deterministic id tie break',
                    data_unit='exact BGR groups; not held-out people',
                    source_sha256=sha_bytes(Path(__file__).read_bytes()))
    write_json(out / 'protocol.json', protocol)
    trained, candidates = {}, []
    weight = compute_sample_weight('balanced', y[train])
    with threadpool_limits(limits=1):
        for config in configs:
            if config['kind'] == 'logistic':
                model = LogisticRegression(C=config['C'], max_iter=1000, random_state=20260914)
            else:
                model = HistGradientBoostingClassifier(**{k:v for k,v in config.items() if k not in ['id', 'kind']},
                                                       early_stopping=False, random_state=20260914)
            started = time.perf_counter()
            model.fit(x[train], y[train], sample_weight=weight)
            seconds = time.perf_counter()-started
            p = model.predict_proba(x[val])[:, 1]
            threshold_rows = [dict(id=config['id']+f'_t{threshold:.1f}', model_id=config['id'],
                                   threshold=threshold, validation=metrics(y[val], p, threshold))
                              for threshold in protocol['thresholds']]
            best = choose(threshold_rows)
            best.update(config=config, fit_seconds=seconds)
            candidates.append(best)
            trained[config['id']] = model
            print('PIXEL VALIDATION',config['id'],best['threshold'],best['validation']['balanced_accuracy'],flush=True)
        winner = choose(candidates)
        selection = dict(winner=winner, candidates=candidates,
                         protocol_sha256=sha_bytes((out / 'protocol.json').read_bytes()),
                         test_accessed=False)
        write_json(out / 'selection.json', selection)
        selection_sha = sha_bytes((out / 'selection.json').read_bytes())
        # Test probabilities are first computed only after the selection file is frozen.
        tested = []
        for row in (candidates[0], winner):
            model = trained[row['model_id']]
            p = model.predict_proba(x[test])[:, 1]
            tested.append(dict(id=row['model_id'], test=metrics(y[test], p, row['threshold'])))
        model = trained[winner['model_id']]
        model_file = out / 'selected_local_model.pkl'
        model_file.write_bytes(pickle.dumps(model, protocol=5))
        # Only reload our own just-created, hashed artifact; no external pickle/weights.
        restored = pickle.loads(model_file.read_bytes())
        original = model.predict_proba(x[test])[:, 1]
        np.testing.assert_array_equal(restored.predict_proba(x[test])[:, 1], original)
        np.savez_compressed(out / 'held_color_predictions.npz', bgr=np.rint(x[test]*255).astype(np.uint8),
                            target_skin=y[test], probability=original)
        result = dict(selection_sha256=selection_sha, tested=tested,
                      model_sha256=sha_bytes(model_file.read_bytes()), model_bytes=model_file.stat().st_size,
                      actual_new_photographs=0, measurement='binary pixel classification, not native skin Lab',
                      limitations=['source people/images unavailable', 'exact-color grouping avoids direct duplicate leakage only',
                                   'natural-image background may share skin colors', 'not an ordinary-selfie accuracy claim'])
        write_json(out / 'results.json', result)
        print('PIXEL RESULT',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
