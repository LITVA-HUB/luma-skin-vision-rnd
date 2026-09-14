"""Report all fixed-OOF diagnostic groups without reading outer predictions."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skin_local_search_train import sha, write_json

LABELS = {'norm_mse': 'Normalized ridge', 'constant_de2': 'Shared perceptual',
          'local_de2': 'Local perceptual', 'local_irls': 'Fixed-metric correction',
          'midpoint_irls': 'Midpoint correction'}
ROLE_NAMES = {'mixed': 'Mixed cameras', 'slr_to_ipod': 'SLR fit side', 'ipod_to_slr': 'iPod fit side'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    result = json.loads((run / 'results.json').read_text())
    audit = json.loads((out / 'audit.json').read_text())
    assert audit['passed'] and audit['results_sha256'] == sha(run / 'results.json')
    assert result['source_lock_sha256'] == audit['source_lock_sha256'] == sha(run / 'source_lock.json')
    rows = result['records']
    summary_rows = []
    for r in rows:
        summary_rows.append({k: r[k] for k in (
            'role', 'family', 'n_people', 'n_candidates', 'deletion_switches', 'bootstrap_original_frequency',
            'bootstrap_weak_alpha_frequency', 'bootstrap_positive_steps_frequency', 'bootstrap_original_rank',
            'bootstrap_full_oof_regret', 'paired_vs_parent', 'paired_vs_runner_up')})
    static = [r for r in rows if not r['family'].endswith('irls')]
    iterative = [r for r in rows if r['family'].endswith('irls')]
    static_deletions = sum(r['deletion_switches'] for r in static)
    iterative_deletions = sum(r['deletion_switches'] for r in iterative)
    runner_ranges_cross_zero = sum(r['paired_vs_runner_up']['conditional_percentiles_2_5_97_5'][0] <= 0. <= r['paired_vs_runner_up']['conditional_percentiles_2_5_97_5'][1] for r in rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), gridspec_kw={'width_ratios': [1.2, 1]}, sharey=True)
    y = np.arange(len(rows))
    colors = ['#147D74' if r['family'].endswith('irls') else '#335A85' for r in rows]
    axes[0].barh(y, [100 * r['bootstrap_original_frequency'] for r in rows], color=colors, height=.63)
    axes[0].scatter([100 * (1 - r['deletion_switches'] / r['n_people']) for r in rows], y, s=23, c='#BD6B25', zorder=3, label='Score deletion: original retained')
    axes[0].set_yticks(y, [ROLE_NAMES[r['role']] + ' | ' + LABELS[r['family']] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel('Original configuration retained (%)')
    axes[0].set_title('Bars: 20,000 person resamples per role')
    axes[0].legend(loc='lower right', fontsize=8, frameon=False)
    for i, r in enumerate(rows):
        p = r['paired_vs_parent']
        low, high = p['conditional_percentiles_2_5_97_5']
        axes[1].plot([low, high], [i, i], c=colors[i], linewidth=2)
        axes[1].scatter([p['mean']], [i], c=colors[i], s=24)
    axes[1].axvline(0, c='#555555', linewidth=.8)
    axes[1].set_xlabel('W choice minus previous P choice (inner DeltaE00)')
    axes[1].set_title('Paired 2.5–97.5% conditional sensitivity range')
    for ax in axes:
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for boundary in (4.5, 9.5):
            ax.axhline(boundary, color='#999999', linewidth=.6)
    fig.suptitle('ChromaSeed-S: correction settings are more sensitive to fit-side people', fontsize=15, x=.04, ha='left')
    fig.text(.04, .015, 'Fixed OOF models and folds; reused TRAIN. Roles overlap. Frequencies and ranges do not measure new-phone accuracy.', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'sensitivity.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    lines = [
        '# ChromaSeed-S: people and model-selection sensitivity', '',
        '**Diagnostic progress; no new accuracy gain or model promoted.** Fixed OOF predictions show that iterative correction settings depend more strongly on fit-side people than static readouts. Yet weak ridge remains strongly preferred on the SLR fit side even when people are resampled; a single influential person does not account for that direction of selection. The cause of the earlier outer regressions is still not established.', '',
        'The completed [W study](../chromaseed_weak_ridge_v1/report.md) motivated this diagnostic after 13 of 15 matching-family outer comparisons worsened. S does not read those outer predictions, fit new models or choose replacement settings. It studies only the previously saved inner score table.', '',
        '## What was measured', '',
        f"Nine inner OOF archives; all {result['checks']['candidate_scores']} family-candidate scores reproduce W (maximum difference {result['max_score_drift']:.3g}). Fit-side groups contain 18, 8 and 16 people, with overlap across roles. Each candidate is scored by averaging its image errors within people, then across the three model seeds. These are separate model runs, not a prediction ensemble.", '',
        'Remove each person from selection scores once: 210 family decisions. Then make 20,000 camera-stratified person resamples per role: 60,000 shared draws yield 300,000 family decisions. No models are retrained. The main diagnostic took ' + f"{result['elapsed_seconds']:.3f}s excluding interpreter import; this is diagnostic processing time, not training or inference speed.", '',
        '## All registered groups', '',
        '| Fit-side role | Family | Single-person deletions changing choice | Original choice retained in bootstrap | Weak alpha (<0.1) selected | Original rank, 95th percentile | Full-OOF regret, 95th percentile |',
        '|---|---|---:|---:|---:|---:|---:|'
    ]
    for r in rows:
        lines.append(f"| {ROLE_NAMES[r['role']]} | {LABELS[r['family']]} | {r['deletion_switches']}/{r['n_people']} | {100*r['bootstrap_original_frequency']:.2f}% | {100*r['bootstrap_weak_alpha_frequency']:.2f}% | {r['bootstrap_original_rank']['p95']:.0f} | {r['bootstrap_full_oof_regret']['p95']:.4f} |")
    lines.extend([
        '', 'Full-OOF regret is the chosen alternative’s score on the same original OOF table minus the original minimum. It measures the size of a configuration switch; it is not a fresh performance estimate. Exact switches can occur among nearly equivalent settings. Repeated stopped trajectories retain the registered fewer-step tie break.', '',
        f'Static methods change in {static_deletions}/{sum(r["n_people"] for r in static)} score-deletion cases; iterative families change in {iterative_deletions}/{sum(r["n_people"] for r in iterative)}. These counts are descriptive: methods share the same people and are not independent trials. The original correction configuration survives only ' + f'{100*min(r["bootstrap_original_frequency"] for r in iterative):.2f}–{100*max(r["bootstrap_original_frequency"] for r in iterative):.2f}% of bootstrap draws. All {runner_ranges_cross_zero}/15 runner-up paired ranges include zero.', '',
        '![Selection sensitivity](sensitivity.png)', '',
        '## Fixed paired comparisons to previous P selections', '',
        'Both configurations stay fixed for each comparison. Lower is better for W. The intervals below are empirical percentile ranges conditional on these fitted predictions, these folds and camera proportions; they are **not population confidence intervals or selection-corrected tests**.', '',
        '| Fit-side role | Family | W − P inner person mean | Conditional 2.5–97.5% range | Fraction below zero |',
        '|---|---|---:|---:|---:|'
    ])
    for r in rows:
        p = r['paired_vs_parent']
        low, high = p['conditional_percentiles_2_5_97_5']
        lines.append(f"| {ROLE_NAMES[r['role']]} | {LABELS[r['family']]} | {p['mean']:.4f} | [{low:.4f}, {high:.4f}] | {100*p['fraction_below_zero']:.2f}% |")
    lines.extend([
        '', '## Decision', '',
        'Do not promote a correction count or a new conservative selection rule from this retrospective diagnostic. Shared/local static SLR choices survive every single-person deletion and select weak alpha in approximately 99.99–100% of conditional resamples. Thus restoring the old penalty through a stability rule alone is not supported as a generalization fix. Conversely, iterative configuration switches often concern very small inner score gaps. Neither finding proves what caused the cross-camera loss.', '',
        'The next useful investigation is a separately registered input/target support diagnostic: quantify camera-group coverage and person-held-out camera predictability in the available color features, with matched color-range controls. That can help distinguish hypotheses about camera processing from changes in people/color distribution. It cannot isolate camera causality in this dataset and is planned, not launched. Preserve the existing stronger models while investigating.', '',
        '## Verification and limits', '',
        f"Eight synthetic numerical tests pass. An independent audit reconstructed all {audit['checks']['candidate_person_values']} candidate/person errors from OOF, all 891 mean scores, 15 original choices, 210 deletion choices, all 60,000 stratified draws and 300,000 bootstrap choices/ranks. It recomputed all 30 paired ranges. Maximum person-score drift {audit['maxima']['person_score_drift']:.3g}; maximum paired-gap drift {audit['maxima']['paired_gap_drift']:.3g}. The audit uses separate aggregation/selection loops but shares the previously verified CIEDE2000 formula and frozen role helper.", '',
        'Only the original TRAIN cache was opened, loading target, patient and device arrays. No colors, images, tokens, old validation/calibration/test or outer predictions were loaded. Exported local numerical artifacts use ordinal person axes; no direct patient identifiers. All W/P and earlier source locks are preserved. No model fitting, new data download, deployment or publication occurred.', '',
        'The fits share training people, so resampling fixed OOF losses omits training and fold-generation uncertainty. Historical reuse and overlapping roles remain. These results establish neither ordinary-phone facial quality, scientific novelty, fairness across skin tones, nor Skolkovo eligibility. The broad compact/fast/high-quality goal remains active.', '',
        'For the established risk of overfitting selection criteria, see [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html). For limits on universally unbiased cross-validation variance estimation, see [Bengio and Grandvalet (2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html). Neither paper identifies the cause in our dataset.', '',
        '[Protocol](../../research/chromaseed_selection_stability_v1_protocol.md) · [Reproduce](reproduce.md) · [Audit](audit.json) · [Verification](verification.json) · [Summary](summary.json) · [Next decision](../../research/chromaseed_selection_stability_next_decision.md)', ''
    ])
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    shutil.copy2(run / 'source_lock.json', out / 'source_lock.json')
    write_json(out / 'summary.json', dict(source_lock_sha256=sha(run / 'source_lock.json'), results_sha256=sha(run / 'results.json'),
                                        audit_sha256=sha(out / 'audit.json'), report_source_sha256=sha(Path(__file__)),
                                        rows=summary_rows, static_deletion_switches=static_deletions,
                                        iterative_deletion_switches=iterative_deletions, runner_up_ranges_include_zero=runner_ranges_cross_zero))
    print('Report and diagnostic figure written; no outer archive accessed.')


if __name__ == '__main__':
    main()
