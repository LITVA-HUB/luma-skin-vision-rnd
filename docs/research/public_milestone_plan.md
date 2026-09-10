# Public real-data milestone

Started 2026-09-10. User-authorized pivot; no proprietary facial collection available. Historical synthetic milestone is frozen at tag `milestone/synthetic-only-2026-09-10`, commit `2685bf0`. Its negative result is retained.

1. Verify original licenses and fresh prior art independently; preserve provenance.
2. Acquire only SimpleCube++ (2,113,441,199 bytes, CC BY 4.0) and small metadata archives. Verify publisher checksum. Do not acquire full Cube++.
3. Implement linear 16-bit image handling, black subtraction, target exclusion, recovery/reproduction angular metrics and official train/test separation. Test numerical contracts before implementation.
4. Run Gray World, Max RGB, Shades of Gray and first-order Gray Edge. Lock learned protocol before test evaluation. Benchmark original-compatible split, with explicit scene-dependence limitation.
5. Reproduce a compact randomly initialized convolutional baseline and matched C+ with standard residual-error head. Use disjoint training/validation/risk-calibration subsets from official train only. Predict reproduction angular error, not invented surface ΔE00.
6. Select A/B/C based on prior art, legal data and validation evidence. Matched compute/budget for special mechanism. Report all seeds and failures.
7. Evaluate test error, fixed coverage 100/95/90/80/70/60%, full curve, tails and ablations; distinguish descriptive test ranking from deployment thresholds selected on calibration.
8. Camera-held-out experiment only with verified camera mapping and permissible real ground truth. Same-sensor Canon models are a limited camera shift, not evidence for arbitrary sensor generalization. Investigate a genuinely different sensor with lawful terms.
9. Export/optimize only after positive evidence. Measure GPU training/inference memory, parameters, file size and batch-one latency regardless of accuracy outcome.
10. Independent review, tests, evidence manifest, updated Skolkovo/IP/research documentation and explicit next decision.

No external publication, paid compute or unlicensed weights. Data images remain local and excluded from Git. Official data splits do not automatically guarantee scene independence; document detected duplicates/grouping. Calibration guarantees do not imply conditional selective coverage guarantees.
