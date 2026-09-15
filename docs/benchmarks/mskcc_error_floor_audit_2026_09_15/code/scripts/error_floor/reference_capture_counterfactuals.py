"""Descriptive TRAIN exclusions, never a model evaluation or noise subtraction."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from luma_skin_vision.color import delta_e00


def stats(v):
    v = np.asarray(v, float).ravel()
    return {"n": len(v), **{k: float(f(v)) for k, f in {
        "mean": np.mean, "median": np.median, "p90": lambda x: np.quantile(x, .9),
        "p95": lambda x: np.quantile(x, .95), "max": np.max}.items()}} if len(v) else {"n": 0}


def main(root):
    ref = json.loads((root / 'instrument.json').read_text())
    cap = json.loads((root / 'capture/capture.json').read_text())
    if cap['scope'] != 'COMPLETE_TRAIN_CAPTURE_DIAGNOSTIC_NO_TRAINING':
        raise ValueError('Complete 966-image capture analysis required')
    rows = json.loads((root / 'audit_train_rows.private.json').read_text())
    features = np.load(root / 'capture/capture_features.private.npz', allow_pickle=False)
    assert np.array_equal(features['image'], [r['image'] for r in rows])
    unique = {r['site']: r for r in rows}
    sites = sorted(unique)
    reps = np.array([unique[s]['repetitions'] for s in sites])
    pairs = list(itertools.combinations(range(3), 2))
    differences = np.stack([reps[:, j]-reps[:, i] for i, j in pairs], 1)
    pair_de = np.stack([delta_e00(reps[:, i], reps[:, j]) for i, j in pairs], 1)
    site_ref = pair_de.mean(1)
    site_captures = {s['site_id']: s for s in cap['per_site']}
    quantity = 'original_central_median_delta_e00'
    site_image = np.array([site_captures[f'S{i+1:03d}'][quantity]['mean']
                          if site_captures[f'S{i+1:03d}']['pairs'] else np.nan
                          for i in range(len(sites))])
    reference_threshold = float(np.quantile(site_ref, .8))
    capture_threshold = float(np.nanquantile(site_image, .8))
    diagnostic = dict(zip(features['diagnostic_names'], features['diagnostic'].T))
    # Union is bounded above by sum; this proxy is explicitly not a learned gate.
    clip = (diagnostic['original_high_clip_pixel_fraction'] +
            diagnostic['original_low_clip_pixel_fraction'])
    clip_threshold = float(np.quantile(clip, .8))
    row_site_index = np.array([sites.index(r['site']) for r in rows])
    masks = {
        'all_TRAIN': np.ones(len(rows), bool),
        'lower_80pct_site_reference_variability': site_ref[row_site_index] <= reference_threshold,
        'lower_80pct_site_capture_variability': site_image[row_site_index] <= capture_threshold,
        'lower_80pct_image_clipping_proxy': clip <= clip_threshold,
    }
    masks['reference_and_capture_stable'] = (masks['lower_80pct_site_reference_variability'] &
                                             masks['lower_80pct_site_capture_variability'])
    masks['reference_capture_and_clipping_stable'] = (masks['reference_and_capture_stable'] &
                                                     masks['lower_80pct_image_clipping_proxy'])
    subsets = []
    for name, mask in masks.items():
        site_indices = np.unique(row_site_index[mask])
        eligible = site_indices[np.isfinite(site_image[site_indices])]
        subsets.append({'name': name, 'status': 'LEAKY_DIAGNOSTIC_ONLY',
                        'images': int(mask.sum()), 'coverage': float(mask.mean()),
                        'people': len({rows[i]['patient'] for i in np.flatnonzero(mask)}),
                        'sites': len(site_indices),
                        'instrument_pairs_on_retained_sites': stats(pair_de[site_indices]),
                        'site_equal_weight_mean_observed_median_variability': stats(site_image[eligible]),
                        'C_error': None,
                        'C_error_status': 'BLOCKED_MISSING_SAVED_OOF_PREDICTIONS'})
    coord_hist = []
    for c, name in enumerate('Lab'):
        values, counts = np.unique(differences[:, :, c], return_counts=True)
        coord_hist.extend({'coordinate': name, 'signed_difference_later_minus_earlier': float(v),
                           'count': int(n), 'fraction': float(n / 744)} for v, n in zip(values, counts))
    with (root / 'instrument_signed_coordinate_histogram.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coord_hist[0])); writer.writeheader(); writer.writerows(coord_hist)
    out = {
        'status': 'REFERENCE_AND_CAPTURE_DIAGNOSTICS_ONLY_C_OOF_MISSING',
        'mask_definition_frozen_without_C_errors': True,
        'thresholds': {'site_reference_pair_mean_p80': reference_threshold,
                       'site_capture_pair_median_color_mean_p80': capture_threshold,
                       'image_original_high_plus_low_clip_p80': clip_threshold},
        'tie_policy': 'Include all values <= percentile; report actual image coverage',
        'single_image_site_policy': 'Capture variability unknown, excluded only from capture-stable masks',
        'subset_reference_summary_policy': 'Use complete triplet at each retained site once; not image-weighted',
        'subset_capture_summary_policy': 'Use original full-site capture-pair mean; do not recompute after excluding images',
        'warning': 'These masks use TRAIN reference data and repeated images and are not deployable gates. They do not remove latent noise or identify attainable model accuracy. No main metric is altered.',
        'subsets': subsets,
        'assessment_order': [{'pair': f'{i+1}->{j+1}',
                              'signed_mean_difference_Lab': (reps[:, j]-reps[:, i]).mean(0).tolist(),
                              'delta_e00': stats(delta_e00(reps[:, i], reps[:, j]))}
                             for i, j in pairs],
        'no_model_loaded_or_fitted': True}
    (root / 'reference_capture_counterfactuals.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(root / 'diagnostic_subset_masks.anonymized.npz',
                        row_index=np.arange(len(rows)), **masks)
    print(json.dumps({'thresholds': out['thresholds'],
                      'coverage': {s['name']: s['coverage'] for s in subsets}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    main(parser.parse_args().audit)
