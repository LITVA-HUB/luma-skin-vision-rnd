"""Render measured C_REBASE_V1 diagnostics; no fitting or data exclusion."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/c_rebase_v1'
a = json.loads((OUT / 'run/audit/c_rebase_v1_error_audit.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
groups = a['capture_variability_quartiles']['groups']
x = np.arange(4)
axes[0].plot(x, [g['delta_e00']['median'] for g in groups], 'o-', color='#2563eb', label='Median model error')
axes[0].plot(x, [g['delta_e00']['p95'] for g in groups], 'o-', color='#b45309', label='p95 model error')
axes[0].set_xticks(x, [g['quantile'] + '\nn=' + str(g['delta_e00']['n']) for g in groups])
axes[0].set_ylim(0, 16)
axes[0].set_ylabel('Instrument-reference CIEDE2000')
axes[0].set_title('Capture variability quartiles\nLocked before OOF; unique sites')
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(axis='y', alpha=.2)
axes[1].bar(x, [g['global_p95_tail_count'] for g in groups], color=['#cbd5e1', '#cbd5e1', '#94a3b8', '#b45309'])
for i, g in enumerate(groups):
    axes[1].text(i, g['global_p95_tail_count'] + .6, str(g['global_p95_tail_count']), ha='center')
axes[1].set_xticks(x, [g['quantile'] for g in groups])
axes[1].set_ylim(0, 44)
axes[1].set_ylabel('Images in the global worst 5% (49 images)')
axes[1].set_title('37 of 49 tail errors are in Q4\nAssociation, not causal attribution')
d = a['squared_coordinate_error_decomposition']
within = 100 * d['within_site_fraction_sum_coordinate_mse']
bias = 100 - within
axes[2].barh(['Total coordinate MSE'], [within], color='#2563eb', label='Within-site prediction variation')
axes[2].barh(['Total coordinate MSE'], [bias], left=[within], color='#94a3b8', label='Site-mean prediction bias')
axes[2].text(within/2, 0, f'{within:.1f}%', ha='center', va='center', color='white', weight='bold')
axes[2].text(within+bias/2, 0, f'{bias:.1f}%', ha='center', va='center', weight='bold')
axes[2].set_xlim(0, 100)
axes[2].set_yticks([])
axes[2].set_xlabel('Share of summed squared Lab-coordinate error (%)')
axes[2].set_title('Exact algebraic decomposition\nThese are NOT causal data/model shares')
axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.2), frameon=False, fontsize=9)
fig.suptitle('C_REBASE_V1 | 966 MSKCC TRAIN images / 24 held-out people | diagnostic audit', fontsize=14)
fig.savefig(OUT / 'error_floor_diagnostics.png', dpi=160)
fig.savefig(OUT / 'error_floor_diagnostics.pdf')
print(OUT / 'error_floor_diagnostics.png')
