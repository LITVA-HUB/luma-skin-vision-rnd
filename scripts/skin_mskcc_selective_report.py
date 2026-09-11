"""Render frozen test aggregates; no fitting, ranking selection or image output."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_mskcc_data import ROOT,sha
from skin_mskcc_selective_core import OUT,verify_lock


def main():
    lock=OUT/'final_lock.json';verify_lock(lock,sha(lock),'final')
    r=json.loads((OUT/'test_results.json').read_bytes());audit=json.loads((OUT/'test_audit.json').read_bytes())
    assert audit['test_results_sha256']==sha(OUT/'test_results.json')
    p=r['risk_models']['ensemble__Proposed'];c=r['risk_models']['ensemble__C_plus'];colors=r['color_comparators']
    profile=json.loads((OUT/'profile.json').read_bytes());reference=json.loads((OUT/'test_data.json').read_bytes())
    lines=['# Independent instrument-referenced skin-color test',
      '', 'All results below are REPRODUCED LOCALLY on real original MSKCC skin photographs and native Lab instrument measurements. Lower DeltaE00 is better. This is not illuminant angular error.',
      '', '## Population and integrity',
      '', '24 TRAIN people / 966 images; 6 source-VALIDATION people / 264 images; 6 CALIBRATION people / 208 images; 10 independent TEST people / 400 images / 105 sites. Person roles are disjoint. Both camera families were seen during development: this is a known-camera, new-person test.',
      '', 'Original source: [MSKCC release](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/), DOI 10.34970/962049, attribution Memorial Sloan Kettering Cancer Center. Original release/per-image metadata says CC-BY; no unsupported version number is assigned. 1,838 paired JPEGs, 2,128,062,766 bytes acquired and individually verified. Dataset rights do not substitute for a clinical/product validation.',
      '', 'Target: mean of actually published complete native Lab instrument readings (SkinColorCatch D65/10-degree convention). It is a site reference, not a dense per-pixel color map. The [pre-test amendment](../../research/skin_mskcc_reference_amendment_v1.md) records three calibration image rows with only one reading. No image was excluded or measurement imputed. Test images by number of complete readings: '+json.dumps(reference['instrument_reading_counts'])+'.',
      '', 'The pre-test Git checkpoint is `checkpoint/skin-selective-pretest-2026-09-11`. Final lock SHA256: `'+sha(lock)+'`. All color/risk models, selections and calibration rules were fixed before TEST decoding. The exposed test is now an evaluation archive; it must not become a tuning set for a new claimed independent result.',
      '', '## Main findings',
      '', f"The primary patch ensemble has mean **{p['full']['mean']:.4f}**, median **{p['full']['median']:.4f}**, p95 **{p['full']['p95']:.4f} DeltaE00** at full coverage. C+ and Proposed have exactly identical color predictions; they differ only in error ranking.",
      '', f"At exact 80% accepted coverage, Proposed mean **{p['coverage'][3]['mean']:.4f}** versus C+ **{c['coverage'][3]['mean']:.4f}**; difference **{r['paired']['ensemble']['proposed_minus_C_plus_risk80']:.4f}**. The patient-cluster 95% bootstrap interval is **{r['paired']['ensemble']['proposed_minus_C_plus_risk80_ci95']}**, crossing zero. This is a small observed gain, not convincing evidence of a novel mechanism.",
      '', f"The stronger simple density ranking on the same color predictions reaches **{colors['votes_mean_ensemble']['coverage'][3]['mean']:.4f}** at 80%. A frozen ordinary CNN/patch six-model fusion achieves full mean **{colors['fusion_ensemble']['full']['mean']:.4f}** and density-ranked 80% mean **{colors['fusion_ensemble']['coverage'][3]['mean']:.4f}**. Proposed does NOT beat the strongest observed full-system comparator. The fusion has 7,337,589 color parameters, so it is not the 2.77M-parameter primary method.",
      '', '## Fixed-coverage color risk',
      '', '| Accepted | C+ mean | Proposed mean | Proposed median | Proposed p95 | Proposed error >10 |',
      '|---:|---:|---:|---:|---:|---:|']
    for a,b in zip(c['coverage'],p['coverage']):
        lines.append(f"| {b['coverage']:.0%} | {a['mean']:.4f} | {b['mean']:.4f} | {b['median']:.4f} | {b['p95']:.4f} | {b['above_10_fraction']:.2%} |")
    t=p['calibration_threshold_outcomes']['0.8']
    lines += ['', f"Exact-coverage ranking is not a deployed threshold guarantee. The threshold set at 80% CALIBRATION coverage accepts **{t['coverage']:.2%}** of TEST images with mean **{t['mean']:.4f}**. A calibrated predicted-error <=2 threshold accepts zero test images; <=5 accepts {p['predicted_error_threshold_outcomes']['5']['coverage']:.2%}, and {p['predicted_error_threshold_outcomes']['5']['above_5_fraction']:.2%} of those still exceed actual DeltaE00 5. Expected error is not a per-image upper bound. Test predicted-error MAE is {p['predicted_error_mae']:.4f}.",
      '', '![Risk versus coverage](risk_coverage.png)',
      '', '## All frozen color comparators',
      '', 'Sorted by observed test mean for descriptive reporting only; no post-test fitting or selection is performed. Their 80% columns all use the same TRAIN-density ranking.',
      '', '| Locally reproduced method | Mean | Median | p95 | Mean at 80% |',
      '|---|---:|---:|---:|---:|']
    for name,a in sorted(colors.items(),key=lambda item:item[1]['full']['mean']):
        f=a['full'];lines.append(f"| {name} | {f['mean']:.4f} | {f['median']:.4f} | {f['p95']:.4f} | {a['coverage'][3]['mean']:.4f} |")
    lines += ['', '## Risk-head replications and mechanism',
      '', 'Six subject-held-out folds produced 18 color fits. Heads used only those OOF residuals. Each arm received the same three candidate head families/budgets; 24 fits total. Primary C+ selected standard MLP, Proposed selected HGB using source validation only. Consequently the primary result compares matched search procedures, not an isolated same-head-family feature ablation. Inputs are 40 versus 49 features. The additional nine encode patch disagreement and cross-seed disagreement. Ensemble confidence and algorithm combinations are existing ideas, not established novelty.',
      '', '| Version | C+ mean at 80% | Proposed mean at 80% | Difference |',
      '|---|---:|---:|---:|']
    for version in ['single17','single29','single43','ensemble']:
        a=r['risk_models'][version+'__C_plus']['coverage'][3]['mean'];b=r['risk_models'][version+'__Proposed']['coverage'][3]['mean']
        lines.append(f'| {version} | {a:.4f} | {b:.4f} | {b-a:.4f} |')
    lines += ['', 'All four patient-cluster intervals cross zero. Huber recurrence, tighter residual thresholds and removing scene context previously failed to provide reliable source gains; these negative ablations are preserved in the source reports.',
      '', '## Camera and capture strata',
      '', '| Primary model stratum | Images | Mean | p95 |', '|---|---:|---:|---:|']
    for kind,values in p['strata'].items():
        for name,a in values.items():lines.append(f"| {kind}: {name} | {a['n']} | {a['mean']:.4f} | {a['p95']:.4f} |")
    lines += ['', 'These device differences mix camera, person and capture effects; they are not a paired camera causal experiment. MSKCC used Canon SLR and iPod Touch with dedicated capture hardware/software and initial white balance. These are not ordinary uncalibrated iPhone/Android facial selfies. No strict unseen-camera skin result is established.',
      '', '## Compute and reproducibility',
      '', f"Primary color model: **{profile['color_parameters']:,} parameters**, **{profile['color_checkpoint_bytes']:,} checkpoint bytes**, RTX 4060 batch-1 GPU median **{profile['gpu_median_ms']:.4f} ms**, p95 **{profile['gpu_p95_ms']:.4f} ms**, PyTorch peak allocated **{profile['inference_peak_allocated_mib']:.2f} MiB**. Timing starts from prepared patch descriptors and excludes disagreement extraction, density, CPU head, JPEG and localization. The CPU Proposed head+calibrator separately takes median {profile['cpu_risk_heads']['Proposed']['median_ms']:.4f} ms on prepared features. These component timings are not end-to-end latency.",
      '', 'Single patch model has 924,932 parameters, measured training peak allocated 108.407 MiB in the original recipe. Ensemble members train sequentially. Earlier original-JPEG preparation alone measured median 91.79 ms on 10 TRAIN images (read/hash/decode/crop/resize/all descriptors; possible OS cache). GPU allocator measurements are not whole-process VRAM. No new skin ONNX/TensorRT deployment is claimed.',
      '', f"Independent audit: {audit['test_prediction_arrays']} prediction arrays, {audit['independent_coverage_cases']} coverage cases, {audit['head_and_calibrator_replays']} exact head/calibrator replays; scalar CIEDE2000 metric maximum gap {audit['max_scalar_metric_gap']:.3g}. All 18 OOF checkpoints and excluded-person scalers also replayed. Full repository suite: 262 passed, 14 historical ONNX warnings, 32.57 seconds.",
      '', '## Meaning for Luma and next decision',
      '', 'IMPLEMENTED: a compact local photograph-to-native-Lab estimator and expected-color-error/rejection subsystem. MEASURED: actual instrument-referenced skin-color error on 400 held-out-person images. NOT PROVED: accurate facial skin measurement from ordinary unseen phones, physical camera independence, instrument-level trueness, reliable cosmetic shade decisions or a defensible new neural-method advantage.',
      '', 'The working product aspiration remains median DeltaE00 <=2 and p95 <=5 at >=80% accepted coverage on supported independent facial-phone data. It is an engineering target, not a universal cosmetics standard. The present primary method has median3.8834/p958.0966 at80%, so it fails that target.',
      '', 'NEXT: retain the ordinary fusion as a frozen comparator and investigate source-only scene/subject shift and within-site capture consistency under the direct skin-color endpoint. Do not re-open this test as a tuning loop. A new confirmatory phone/facial claim needs a distinct appropriately licensed instrument-referenced cohort. Model novelty remains unverified; all angular and synthetic negatives remain archived.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    series={'C+':c,'Proposed':p,'Same color + density':colors['votes_mean_ensemble'],'CNN/patch fusion + density':colors['fusion_ensemble']}
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['method','accepted_count','coverage','mean_delta_e00'])
        for name,a in series.items():
            for n,value in enumerate(a['full_curve_mean'],1):writer.writerow([name,n,n/400,value])
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    for name,a in series.items():
        x=np.arange(1,401)/400;y=np.array(a['full_curve_mean']);mask=x>=.6
        ax.plot(x[mask]*100,y[mask],label=name,linewidth=2)
    ax.set(xlabel='Accepted test images (%)',ylabel='Mean skin-color error (DeltaE00)',title='Real instrument skin reference: 400 images / 10 held-out people')
    ax.grid(alpha=.25);ax.legend();fig.savefig(OUT/'risk_coverage.png',dpi=160);plt.close(fig)
    print('Wrote report, 1600 aggregate curve rows and figure; no participant identifiers.')


if __name__=='__main__':main()
