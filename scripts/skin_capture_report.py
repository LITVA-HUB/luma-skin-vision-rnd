"""Read-only summary of all architecture/objective/camera factorial cells."""
import json
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_capture_train import OUT,SEEDS,OBJECTIVES,ARCHES


def main():
    rows=[json.loads(p.read_bytes()) for p in OUT.glob('*/result.json')];assert len(rows)==54
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    groups=[]
    for protocol in ['mixed','from_SLR','from_ipod']:
        for objective in OBJECTIVES:
            for arch in ARCHES:
                rs=sorted([r for r in rows if (r['protocol'],r['objective'],r['arch'])==(protocol,objective,arch)],key=lambda r:r['seed'])
                assert [r['seed'] for r in rs]==SEEDS
                groups.append({'protocol':protocol,'objective':objective,'arch':arch,
                    'seed_means':[r['full']['mean'] for r in rs],'mean_over_seed_means':float(np.mean([r['full']['mean'] for r in rs])),
                    'mean_over_seed_p95':float(np.mean([r['full']['p95'] for r in rs])),
                    'mean_mode_accuracy':float(np.mean([r['mode_accuracy'] for r in rs])),
                    'mean_capture_disagreement':float(np.mean([r['paired_repeatability']['mean_delta_e00_between_captures'] for r in rs]))})
    def get(p,o,a):return next(g for g in groups if (g['protocol'],g['objective'],g['arch'])==(p,o,a))
    contrasts=[]
    for protocol in ['mixed','from_SLR','from_ipod']:
        for objective in OBJECTIVES:
            m,u,p=[get(protocol,objective,a) for a in ['mixture','uniform','plain']]
            contrasts.append({'protocol':protocol,'objective':objective,
                'mixture_minus_matched_uniform':m['mean_over_seed_means']-u['mean_over_seed_means'],
                'mixture_minus_plain':m['mean_over_seed_means']-p['mean_over_seed_means'],
                'paired_seed_differences':(np.array(m['seed_means'])-u['seed_means']).tolist()})
    summary={'scope':'Source-only exploratory factorial, not a new independent test','groups':groups,'contrasts':contrasts,
        'fits':54,'parameters_per_model':rows[0]['stored_parameters'],'source_lock_sha256':sha(OUT/'source_lock.json'),
        'reserved_test_or_calibration_loaded':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    lines=['# Latent capture conditioning and perceptual objective',
      '', 'All54fits completed on REAL original MSKCC images and instrument-native Lab labels. Results are REPRODUCED LOCALLY, but use previously studied source TRAIN/VALIDATION people. The independent400-image TEST and208-image CALIBRATION caches/labels/predictions were not loaded. These scores are not new final-test accuracy or universal facial-phone validation.',
      '', '## Exact factorial and provenance',
      '', 'Three architectures x two objectives x three seeds x three camera protocols. Every model has929,297stored parameters, identical per-seed initialization and80epochs. Plain and uniform average four color hypotheses. Mixture weights them using a gate inferred from the image. Uniform and mixture receive identical0.1-weighted capture-mode supervision; plain does not. Thus uniform is the strict equal-capacity/supervision comparator, while plain tests whether the added supervision itself helps. The gate uses no provided camera or capture metadata at inference.',
      '', 'The auxiliary gate has2,052parameters and is unnecessary for plain/uniform color inference. Mixture uses it. Keeping unused parameters in a checkpoint is not extra functional capacity for plain/uniform. Four linear hypothesis outputs averaged uniformly are expressively reducible to one output head; the conditional combination is the mechanism being tested.',
      '', 'MSE is standardized native-Lab squared error. DE2 is actual CIEDE2000 squared/25, with native double-precision instrument targets. Full evaluation uses independently checked NumPy/scalar CIEDE2000. No angular-to-color conversion, artificial skin labels, new pretrained weights or large inference model is used.',
      '', '## All source results',
      '', 'Values are means over three separate seed scores, NOT an ensemble result. p95 is likewise averaged over seed p95 values.',
      '', '| Protocol | Architecture | Objective | Mean DeltaE00: seed17 /29 /43 | Mean over seeds | Mean p95 | Mode accuracy |',
      '|---|---|---|---|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['protocol']} | {g['arch']} | {g['objective']} | {' / '.join(f'{v:.4f}' for v in g['seed_means'])} | {g['mean_over_seed_means']:.4f} | {g['mean_over_seed_p95']:.4f} | {g['mean_mode_accuracy']:.1%} |")
    lines += ['', 'Mixed:24TRAIN people/966images;6selection/evaluation people/264images. FromSLR:8TRAIN people/323images and3same-camera selection people/132images; opposite-camera3people/132images evaluated after checkpoint selection. FromiPod:16TRAIN people/643images and3same-camera selection people/132images, opposite3people/132images evaluated after selection. No opposite-camera observations or target scales enter a fit. All54configurations were fixed before these experiments; no camera-selected challenger subset. Historical source exposure still limits independence.',
      '', '## Matched mechanism contrasts',
      '', 'Negative favors mixture. These are descriptive paired contrasts on small previously exposed source cohorts, not claims of statistical significance.',
      '', '| Protocol | Objective | Mixture minus uniform | Mixture minus plain | Seedwise mixture minus uniform |',
      '|---|---|---:|---:|---|']
    for c in contrasts:
        lines.append(f"| {c['protocol']} | {c['objective']} | {c['mixture_minus_matched_uniform']:+.4f} | {c['mixture_minus_plain']:+.4f} | {' / '.join(f'{v:+.4f}' for v in c['paired_seed_differences'])} |")
    lines += ['', 'Compare against both controls and both directions. A gain over the auxiliary-task control alone is insufficient if the ordinary plain estimator is stronger. A gain for one camera direction does not establish universal camera independence. Mode accuracy is a diagnostic of the auxiliary task, not skin-color accuracy or calibrated reliability.',
      '', '## Verification and limitations',
      '', f"{audit['fits']}fits, {audit['exact_prediction_array_replays']}exact checkpoint prediction-array replays, {audit['gate_and_hypothesis_replays']}gate/hypothesis replays, {audit['independent_scalar_delta_e00_comparisons']}independent scalarDeltaE00 comparisons; maximum metric gap{audit['maximum_metric_gap']:.3g}. All target scales and same-camera epoch-selection boundaries checked; initial states match. Numerical loss independently agrees on10,000random color pairs within2.14e-14; all34rounded Sharma fixtures pass5e-5 tolerance, with finite-difference/neutral-gradient tests. Formula checks are not synthetic skin-accuracy evidence.",
      '', 'The exact CIEDE2000 formula is piecewise and has hue discontinuities; using an autograd implementation does not remove these. Its direct optimization is not guaranteed to improve generalization or the tail. All gradient norms were checked finite without clipping. Stored model files and measured training allocation are recorded per fit; no new batch1/ONNX/TensorRT or full-pipeline latency claim is made.',
      '', 'These models and all prior negative arms remain archived. The previously measured independent skin benchmark stays authoritative and unchanged. No new source result is promoted to final test, cosmetics accuracy, clinical utility or a patent-novelty claim. [Frozen protocol](../../research/skin_capture_protocol_v1.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps({'groups':groups,'contrasts':contrasts}))


if __name__=='__main__':main()
