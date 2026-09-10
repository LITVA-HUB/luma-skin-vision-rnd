# V4 source-screen report task

## Changes

- Added `scripts/cc_v4_report.py`, a reproducible source-validation-only auditor and publisher.
- Verified all three run manifests, hashes, matched code/data/roles/budgets, epoch histories, and independently selected searched/direct checkpoints.
- Recomputed recovery and reproduction errors from only the 119 validation GT rows using float32 GT followed by NumPy float64 `atan2`; all saved arrays and summaries agreed within 1e-6 degree.
- Reconstructed the exact 25/51/103 candidate sets, stage 1/2/4 oracles, selected regret, true-error changes, full risk-coverage curves, and fixed 100/95/90/80/70/60 coverage summaries.
- Published `docs/benchmarks/cc_v4_report.md` and the hashed `docs/benchmarks/cc_v4/source_screen/` archive. The archive contains receipts, histories, prediction arrays, protocol/script snapshots, three benchmark rounds per mode, and no weights or images.
- Added atomic regeneration for the generator's own output. Every recursive directory removal and rename has an explicit resolved direct-child containment check, rejects links/junctions, and restores the prior report if staged installation fails.

## Verification

```text
.venv/Scripts/python.exe -m py_compile scripts/cc_v4_report.py
passed

.venv/Scripts/python.exe scripts/cc_v4_report.py --self-check
{"status": "passed", "checks": ["atan2 exact null", "action-grid oracle", "stable risk coverage"]}

.venv/Scripts/python.exe scripts/cc_v4_report.py --replace
passed; posterior/action/transport benchmark status MEASURED

generated archive manifest
46 entries checked, 0 hash mismatches, 0 weights, 0 data images
```

The generated PNG was visually inspected for legibility; its risk panel includes the complete 0–100% coverage domain. `git diff --check` passed for the report-owned files.

## Result and limitations

Transport loses at the frozen step-2 endpoint: 2.388888 degrees versus 2.180884 posterior and 2.316903 action. Its raw 80% risk-ranked mean is 1.999839 degrees versus 1.894555 posterior. The proposed transport mechanism therefore establishes no gain in this screen, and posterior is the simpler next-gate choice.

This is one-seed evidence on a repeatedly used 119-image development validation set. The same-seed histories diverged at epoch 1, before field loss was active, because deterministic CUDA execution was not enforced; mode gaps are not stable causal estimates. A paired starting checkpoint and multiple seeds are needed next. Risk is uncalibrated, timing varies materially across repeated whole-benchmark rounds, direct point-only latency is unmeasured, and no independent test/camera, INTEL-TAU, facial, skin-color, CIELAB/DeltaE, C+ calibration, universality, or novelty claim follows.
