"""Summarize measured three-seed development outcomes, never ensemble predictions."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'docs/benchmarks/cc_v5/three_seed_screen/summary.json'
    out = root/'docs/benchmarks/cc_v5/final_summary'
    if out.exists():
        raise ValueError('Existing summary; do not overwrite')
    data = json.loads(source.read_text(encoding='utf-8'))
    if data['seeds'] != [17, 29, 43] or len(data['results']) != 36:
        raise ValueError('Incomplete primary experiment')
    names = {'point': 'Point only', 'posterior_random': 'Fixed posterior',
             'action_random': 'Generic action critic', 'transport_random': 'Physical transport',
             'transport_policy': 'Transport + selected actions',
             'transport_gradient': 'Transport + selected actions + derivatives'}
    aggregate = []
    for arm, name in names.items():
        b = [r for r in data['results'] if r['arm'] == arm and r['checkpoint'] == 'best']
        f = [r for r in data['results'] if r['arm'] == arm and r['checkpoint'] == 'last']
        means = [r['reproduction']['mean'] for r in b]
        row = {'arm': arm, 'label': name, 'best_mean': float(np.mean(means)),
               'best_seed_std': float(np.std(means, ddof=1)),
               'final_mean': float(np.mean([r['reproduction']['mean'] for r in f])),
               'fixed_coverage': None, 'coverage': None, 'curve': None}
        if arm != 'point':
            row['fixed_coverage'] = {str(c): float(np.mean([r['selective']['fixed'][str(c)]['mean'] for r in b]))
                                     for c in (100, 95, 90, 80, 70, 60)}
            row['coverage'] = (np.arange(1, 120)/119).tolist()
            row['curve'] = np.mean([r['selective']['curve'] for r in b], axis=0).tolist()
        aggregate.append(row)
    out.mkdir(parents=True)
    write_json(out/'aggregate.json', {'source_sha256': sha256(source), 'script_sha256': sha256(__file__),
               'scope': 'Mean of per-seed metrics on reused119 validation images; not an ensemble or independent CI',
               'records': aggregate})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in aggregate[1:]:
        ax.plot(100*np.array(r['coverage']), r['curve'], label=r['label'], lw=2)
    ax.set(xlim=(60, 100), xlabel='Accepted coverage (%)', ylabel='Mean reproduction error (degrees)',
           title='V5 development risk–coverage: mean of 3 training seeds')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.text(.12, .015, '119 reused validation images • raw risk ranking • no calibration or independent phone test', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out/'risk_coverage.png', dpi=160)
    fig.savefig(out/'risk_coverage.svg')
    plt.close(fig)
    write_json(out/'manifest.json', {'sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(aggregate))


if __name__ == '__main__':
    main()
