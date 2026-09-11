"""Summarize all fixed oracle controls without implying image accuracy."""
import json
from skin_issa_data import OUT, sha


def main():
    result=json.loads((OUT/'material_results.json').read_text(encoding='utf-8-sig'))
    audit=json.loads((OUT/'independent_audit.json').read_text(encoding='utf-8-sig'))
    overlap=json.loads((OUT/'overlap_sensitivity.json').read_text(encoding='utf-8-sig'))
    assert audit['status']=='PASS'
    assert audit['bindings']['docs/benchmarks/skin_issa_v1/material_results.json']==sha(OUT/'material_results.json')
    lines=['# ISSA measured skin material controls', '',
        '**Input is the full measured spectrum. These are oracle compression controls,',
        'not photograph/RGB skin-color accuracy. No proposed innovation is established.**', '',
        'Original ISSA v4 (CC BY4.0), 15,256 records. Local file contains 2,107 subject',
        'labels, six fewer than the article-described population. Three labels have',
        'conflicting gender codes and one conflicting age code. Missing age/gender',
        'cells are preserved. Disjoint codes are not independent identity verification.', '',
        'Fitting: 8,680 records / 1,232 labels. Validation: 1,822 / 262. Roles were',
        'frozen before participant spectra/color values were inspected. Source origins',
        '9-11 and 1,917 known-source test records remain reserved and numerically unread.',
        'See [metadata](metadata.json), [split lock](split_lock.json), and',
        '[pre-validation fit lock](material_lock.json).', '',
        'All 31 common measured bands (400-700nm) are used for spectral error. Native',
        'DeltaE00 is evaluated only on 1,156 validation records whose original support',
        'is exactly 400-700nm, using the supplied Lab and the workbook-specific white.',
        'Other records use 360-740nm in their original XYZ calculation. Dropping their',
        'extra measured bands would itself change color. These are neither full-visible',
        'integrations nor MSKCC D65/10-degree targets. No missing wavelengths are filled.', '',
        'Each subject has equal total weight when fitting a basis. All configurations',
        'were fixed before validation and are reported; no selected winner is promoted.', '',
        '| Representation | Latent width | Mean DeltaE00 | Median | p95 | Worst source mean | Mean spectral relative L2 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        e=r['common_support_native_delta_e00']
        lines.append(f"| {r['method']} | {r['width']} | {e['mean']:.4f} | {e['median']:.4f} | {e['p95']:.4f} | {r['worst_origin_mean_delta_e00']:.4f} | {100*r['spectral_relative_l2']['mean']:.3f}% |")
    lines += ['', '## Interpretation', '',
        'The tested two-dimensional bases lose substantial color variation (mean about',
        '2.35-2.53, p95 about 5.52-5.71 DeltaE00), even with the full spectrum supplied.',
        'This does not rule out nonlinear two-variable optical models. Three linear components',
        'reach mean 0.4495 / p95 1.0564. Eight optical-density components reach mean',
        '0.0236 / p95 0.0611, but this extra fidelity does not prove those coefficients',
        'can be inferred from an ordinary photo. Reflectance is better than log/density',
        'at width three; the opposite ordering at width eight is not a universal',
        'advantage of a physically named transform.', '',
        'The 8-component decoder has 279 stored coefficients; it is not a complete',
        'camera model. All fitted basis states occupy 24,634 bytes together. CPU',
        'fitting of the three bases took 0.0156 seconds on this run. GPU training,',
        'inference VRAM and camera-model latency were not measured by this screen.', '',
        '## Verification', '',
        f"{audit['independent_scalar_delta_e00_cases']:,} independent scalar color-error cases; maximum metric gap {audit['maximum_metric_gap']:.3g}.",
        f"Independent weighted SVD versus covariance eigensolver reconstruction gap {result['independent_weighted_svd_max_spectrum_gap']:.3g}.",
        f"Validation spectra exactly matching TRAIN at all 31 bands: {audit['validation_spectra_exactly_matching_training']}.",
        f"TRAIN spectrum fingerprints shared by different subject labels: {audit['train_fingerprints_shared_by_multiple_subject_labels']}.",
        'These exact-duplicate checks do not prove absence of near duplicates.', '',
        '## Exact-spectrum overlap sensitivity (post hoc)', '',
        'Seventeen validation spectra match TRAIN exactly despite different subject',
        'labels. Removing every validation label connected to TRAIN through exact',
        'spectra (including transitive links) excludes two labels / 17 records, all',
        'in the 360-740nm group. Consequently none of the 1,156 primary color',
        'records is removed and every DeltaE00 value above is unchanged. This is',
        'a descriptive sensitivity check, not proof that all remaining identities',
        'are independent. The frozen split and original results remain preserved.', '',
        '| Representation | Width | Mean spectral relative L2 without linked labels |',
        '|---|---:|---:|']
    for r in overlap['records']:
        lines.append(f"| {r['method']} | {r['width']} | {100*r['spectral_relative_l2']['mean']:.3f}% |")
    lines += ['',
        'All TRAIN/VALIDATION native XYZ/Lab values replay numerically. The source',
        'contains an unrelated hue-formula reference outside the data table; hue is',
        'not used. The original workbook was not modified.', '',
        '## Next decision', '',
        'A material basis preserves color in this exploratory check, with the',
        'identity limitations above. Next test',
        'whether it reduces actual image-to-instrument Lab error under a correctly',
        'matched observer/support convention, against ordinary matched bottlenecks.',
        'Alternatively use its feasible color set to reject ambiguous inputs rather',
        'than forcing one plausible skin color. Physical skin manifolds and spectral',
        'PCA are existing ideas; the optical skin-color locus is also addressed by',
        '[2026 prior art](https://pmc.ncbi.nlm.nih.gov/articles/PMC13307969/).', '',
        'Preserved independent MSKCC result remains primary mean 4.4570 / selective',
        '80% mean 4.1591; stronger ordinary fusion 4.3005 / 4.1447. These ISSA oracle',
        'numbers must never replace that result in a camera-accuracy claim.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print('All 16 controls reported with oracle-input and reference-convention limits.')


if __name__=='__main__':main()
