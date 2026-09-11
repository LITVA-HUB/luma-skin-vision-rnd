"""Frozen source-only skin Lab controls, no calibration/test endpoint access."""
import json
import math
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, PROTOCOL, MANIFEST, sha, load_source

OUT = ROOT / 'docs/benchmarks/skin_mskcc_summary_v1'
RUN = ROOT / 'experiments/runs/skin_mskcc_summary_v1'
COVERAGES = [1, .95, .9, .8, .7, .6]


def summarize(errors, rows):
    e = np.asarray(errors)
    people = sorted({r['patient'] for r in rows})
    by_person = [np.mean([v for v, r in zip(e, rows) if r['patient'] == p]) for p in people]
    return {'n': len(e), 'people': len(people), 'sites': len({r['site'] for r in rows}),
            'mean': float(e.mean()), 'median': float(np.median(e)),
            'p90': float(np.quantile(e, .9)), 'p95': float(np.quantile(e, .95)),
            'above_5_fraction': float(np.mean(e > 5)), 'above_10_fraction': float(np.mean(e > 10)),
            'patient_balanced_mean': float(np.mean(by_person))}


def candidates():
    yield 'constant', DummyRegressor(strategy='mean')
    for name, degree in [('affine', 1), ('poly2', 2)]:
        for alpha in [.01, 1., 100.]:
            yield f'{name}_a{alpha:g}', make_pipeline(StandardScaler(),
                PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
    for alpha in [.1, 1.]:
        yield f'mlp64x32_a{alpha:g}', TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32),
                activation='tanh', solver='lbfgs', alpha=alpha, max_iter=2000, random_state=17)),
            transformer=StandardScaler())
    yield 'knn5', make_pipeline(StandardScaler(), KNeighborsRegressor(5, weights='distance'))


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    RUN.mkdir(exist_ok=False, parents=True)
    train, x, y, rep = load_source('train')
    val, vx, vy, _ = load_source('validation')
    scaler = StandardScaler().fit(x)
    risk = NearestNeighbors(n_neighbors=5).fit(scaler.transform(x)).kneighbors(scaler.transform(vx))[0].mean(1)
    order = np.lexsort((np.array([r['image'] for r in val]), risk))
    # Repeat measurements are counted once per skin site, never once per repeated photo.
    first = {}; max_repeat_difference = 0.
    for i, row in enumerate(train):
        if row['site'] in first:
            max_repeat_difference = max(max_repeat_difference, float(np.max(np.abs(rep[i] - rep[first[row['site']]]))))
        else:
            first[row['site']] = i
    if max_repeat_difference > 0:
        raise ValueError('Different instrument triplets assigned to repeated site')
    unique_rep = rep[list(first.values())]
    repeat_errors = np.concatenate([delta_e00(unique_rep[:, a], unique_rep[:, b]) for a,b in [(0,1),(0,2),(1,2)]])
    results = {'scope': 'SOURCE VALIDATION, never final test; author image summaries, not local pixel pipeline',
               'protocol_sha256': sha(PROTOCOL), 'manifest_sha256': sha(MANIFEST),
               'script_sha256': sha(Path(__file__)),
               'data_script_sha256': sha(ROOT/'scripts/skin_mskcc_data.py'),
               'color_script_sha256': sha(ROOT/'src/luma_skin_vision/color.py'),
               'train_images': len(train), 'validation_images': len(val),
               'repeatability': {'source_sites': len(unique_rep), 'pairs': len(repeat_errors),
                                'mean_delta_e00': float(repeat_errors.mean()),
                                'median_delta_e00': float(np.median(repeat_errors)),
                                'p95_delta_e00': float(np.quantile(repeat_errors,.95))}, 'models': []}
    for name, model in candidates():
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter('always')
            model.fit(x, y)
        pred = model.predict(vx)
        if not np.isfinite(pred).all():
            raise ValueError(f'{name} has invalid predictions')
        de = delta_e00(pred, vy)
        path = RUN / (name + '.joblib')
        joblib.dump(model, path)
        replay = joblib.load(path).predict(vx)
        if not np.array_equal(replay, pred):
            raise ValueError('Model replay differs')
        np.savez(RUN / (name + '_validation.npz'), prediction=pred, target=vy, delta_e00=de, risk=risk,
                 image=np.array([r['image'] for r in val]), patient=np.array([r['patient'] for r in val]))
        record = {'name': name, 'fit_seconds': time.perf_counter()-start,
                  'model_bytes': path.stat().st_size, 'model_sha256': sha(path),
                  'warnings': [str(w.message) for w in caught], 'full': summarize(de, val),
                  'mean_delta_e76': float(np.linalg.norm(pred-vy,axis=1).mean()),
                  'coverage': [], 'strata': {}, 'exact_replay': True}
        for coverage in COVERAGES:
            chosen = order[:math.ceil(coverage*len(val))]
            record['coverage'].append({'coverage': coverage, **summarize(de[chosen], [val[i] for i in chosen])})
        for key in ['device', 'image_type', 'anatomic_site']:
            record['strata'][key] = {}
            for value in sorted({r[key] for r in val}):
                ids = [i for i,r in enumerate(val) if r[key] == value]
                record['strata'][key][value] = summarize(de[ids], [val[i] for i in ids])
        results['models'].append(record)
        (OUT / 'results.partial.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'method': name, 'validation': record['full'], 'warnings': record['warnings']}),flush=True)
    best = min(results['models'],key=lambda r:r['full']['patient_balanced_mean'])
    results['source_selected'] = best['name']
    results['calibration_or_test_endpoints_opened'] = False
    (OUT / 'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    lines = ['# MSKCC direct skin-color source pilot v1', '',
             '**MEASURED SOURCE VALIDATION ONLY.** 966 training images / 24 people; 264 validation images / 6 different people.',
             'Ten test participants / 400 images and six calibration participants / 208 images remain numerically unopened.', '',
             'Input: author-supplied three image-median Lab features. Target: mean of three real SkinColorCatch Lab readings at that skin site.',
             'DeltaE00 is in the native instrument convention; no illuminant angle or invented skin reference is used.', '',
             '|Locally fit control|Mean DeltaE00|Median|p95|Mean at 80% diagnostic coverage|',
             '|---|---:|---:|---:|---:|']
    for r in results['models']:
        lines.append(f"|{r['name']}|{r['full']['mean']:.4f}|{r['full']['median']:.4f}|{r['full']['p95']:.4f}|{r['coverage'][3]['mean']:.4f}|")
    lines += ['', f"Source selection by patient-balanced mean: **{best['name']}**.", '',
              'The selection score has been used to choose the model and is not an independent final performance estimate.',
              'Coverage score is fixed 5-neighbor distance, not calibrated expected color error. Rejection may fail.', '',
              f"TRAIN-only repeatability: {len(unique_rep)} distinct sites / {len(repeat_errors)} pairwise measurement differences; median DeltaE00 {np.median(repeat_errors):.4f}, p95 {np.quantile(repeat_errors,.95):.4f}.",
              'This reflects within-site repeated measurements, not a certified physical accuracy floor.', '',
              'No novel proposed architecture, actual pixel pipeline, unseen-camera result, phone-selfie accuracy or cosmetic matching accuracy has been demonstrated here.',
              'All ten controls, predictions, warning logs and exact model replay checks are retained. Original image acquisition is separate.', '',
              'Sources: [original MSKCC release](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/),',
              '[manufacturer instrument convention](https://store.delfintech.com/products/skincolorcatch).',
              'Attribution: Memorial Sloan Kettering Cancer Center; original CC-BY data.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'source_selected': best['name'], 'repeatability':results['repeatability']}),flush=True)


if __name__ == '__main__':
    main()
