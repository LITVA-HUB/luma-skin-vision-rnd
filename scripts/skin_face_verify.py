"""Read-only Seg1 provenance and stored-metric arithmetic verification."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
RUN = DATA / 'facial_skin_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f'Numeric mismatch: {a} != {b}')


def confusion(row):
    tp, fp, fn = (row[k] for k in ['tp', 'fp', 'fn'])
    close(row['iou'], tp / (tp + fp + fn))
    close(row['dice'], 2 * tp / (2 * tp + fp + fn))
    # Frozen segment.py defines an empty precision/recall denominator as 1.
    # IoU still penalizes missing predictions, and color coverage is separate.
    close(row['precision'], tp / (tp + fp) if tp + fp else 1.)
    close(row['recall'], tp / (tp + fn) if tp + fn else 1.)


def main():
    protocol, selected, history = read('protocol.json'), read('selection.json'), read('history.json')
    appearance_protocol = read('appearance_protocol.json')
    for bound in [protocol, appearance_protocol]:
        for name, digest in bound['sources'].items():
            assert sha(ROOT / name) == digest, name
    assert selected['protocol_sha256'] == sha(RUN / 'protocol.json')
    assert selected['history_sha256'] == sha(RUN / 'history.json')
    assert selected['model_sha256'] == sha(RUN / 'best.pt')
    assert selected['model_bytes'] == (RUN / 'best.pt').stat().st_size
    assert len(history) == protocol['epochs'] == 12
    assert [h['epoch'] for h in history] == list(range(1, 13))
    winner = max(history, key=lambda h: h['validation']['mean_image_iou'])
    assert winner['epoch'] == selected['selected_epoch'] == 12
    assert winner['validation'] == selected['validation']
    for h in history:
        assert math.isfinite(h['training_loss'])
        confusion(h['validation'])
        a = np.load(RUN / f"validation_epoch_{h['epoch']:02d}_iou.npy", allow_pickle=False)
        assert len(a) == 1692 and np.isfinite(a).all()
        close(a.mean(), h['validation']['mean_image_iou'])
    tested, test_protocol = read('test_results.json'), read('test_protocol.json')
    assert tested['selection_sha256'] == test_protocol['selection_sha256'] == sha(RUN / 'selection.json')
    assert tested['protocol_sha256'] == sha(RUN / 'test_protocol.json')
    assert tested['model_sha256'] == selected['model_sha256']
    assert test_protocol['code_sha256'] == sha(ROOT / 'scripts/skin_face_evaluate.py')
    assert test_protocol['triplets_sha256'] == sha(DATA / 'lapa/usable_triplets.json')
    assert tested['ground_truth_color_accuracy'] is None
    for row in [tested['test'], *tested['paired_128'].values()]:
        confusion(row)
        assert sum(row[k] for k in ['tp', 'fp', 'fn', 'tn']) == row['images'] * 192**2
    with np.load(RUN / 'test_image_metrics.npz', allow_pickle=False) as a:
        assert len(a['stems']) == len(set(a['stems'])) == len(a['iou']) == 2000
        assert a['stems'][a['comparison_indices']].tolist() == test_protocol['baseline_subset']
        np.testing.assert_array_equal(a['iou'][a['comparison_indices']], a['comparison_iou'])
        close(a['iou'].mean(), tested['test']['mean_image_iou'])
        close(np.median(a['iou']), tested['test']['median_image_iou'])
        close(np.percentile(a['iou'], 10), tested['test']['image_iou_p10'])
        close(a['comparison_iou'].mean(), tested['paired_128']['cnn']['mean_image_iou'])
    baseline = read('test_baseline_per_image.json')
    assert [r['stem'] for r in baseline] == test_protocol['baseline_subset']
    for row in baseline:
        confusion(row)
    for key in ['tp', 'fp', 'fn', 'tn']:
        assert sum(r[key] for r in baseline) == tested['paired_128']['uci_color_only'][key]
    close(np.mean([r['iou'] for r in baseline]), tested['paired_128']['uci_color_only']['mean_image_iou'])
    apparent = read('appearance_results.json')
    assert apparent['appearance_protocol_sha256'] == sha(RUN / 'appearance_protocol.json')
    assert not apparent['physical_truth']
    assert not appearance_protocol['held_results_present_at_freeze']
    assert not appearance_protocol['held_protocol_present_at_freeze']
    assert (RUN / 'appearance_protocol.json').stat().st_mtime < (RUN / 'test_protocol.json').stat().st_mtime
    assert (RUN / 'selection.json').stat().st_mtime <= (RUN / 'test_protocol.json').stat().st_mtime
    assert [r['stem'] for r in apparent['records']] == test_protocol['baseline_subset']
    common = [r for r in apparent['records'] if all(r['errors'][k] is not None for k in ['cnn', 'uci'])]
    for model, summary in apparent['summary'].items():
        valid = [r for r in apparent['records'] if r['errors'][model] is not None]
        assert summary['valid_colors'] == len(valid)
        assert summary['common_coverage_images'] == len(common)
        for category, rows in [('all_available', valid), ('common_coverage', common)]:
            for key, value in summary[category].items():
                close(value, np.mean([r['errors'][model][key] for r in rows]))
    timing, timing_protocol = read('runtime.json'), read('runtime_protocol.json')
    assert timing_protocol['source_sha256'] == sha(ROOT / 'scripts/skin_face_runtime.py')
    assert timing['protocol_sha256'] == sha(RUN / 'runtime_protocol.json')
    assert len(timing['records']) == 6
    for row in timing['records']:
        assert row['max_logit_difference'] <= timing_protocol['logit_parity_absolute_tolerance']
        assert row['differing_mask_pixels'] == 0
        for key, count in [('model_only', 48), ('jpeg_to_mask', 15)]:
            record = row[key]
            assert record['repetitions'] == count
            assert 0 < record['minimum_ms'] <= record['median_ms'] <= record['p95_ms']
    pipeline = read('pipeline.json')
    assert pipeline['stage'] == 'complete'
    assert pipeline['pipeline_source_sha256'] == sha(ROOT / 'scripts/skin_face_pipeline.py')
    assert pipeline['test_results_sha256'] == sha(RUN / 'test_results.json')
    assert pipeline['diagnostic_sha256'] == sha(RUN / 'diagnostic.json')
    filenames = ['protocol.json', 'selection.json', 'history.json', 'best.pt', 'test_protocol.json',
                 'test_results.json', 'test_image_metrics.npz', 'test_baseline_per_image.json',
                 'appearance_protocol.json', 'appearance_results.json', 'runtime_protocol.json',
                 'runtime.json', 'pipeline.json', 'diagnostic.json']
    verification = dict(status='verified', method='source/artifact hashes and stored-metric arithmetic; no independent inference rerun',
                        source_sha256=sha(__file__), artifacts={n: sha(RUN / n) for n in filenames},
                        epochs=12, test_images=2000, paired_images=128, common_color_coverage=126,
                        test_now_exposed=True, ordinary_phone_instrument_accuracy=None)
    target = RUN / 'final_verification.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8')) == verification
        print('SEG1 READ-ONLY VERIFICATION PASSED', sha(target), flush=True)
    else:
        target.write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
        print('SEG1 VERIFICATION PASSED', sha(target), flush=True)


if __name__ == '__main__':
    main()
