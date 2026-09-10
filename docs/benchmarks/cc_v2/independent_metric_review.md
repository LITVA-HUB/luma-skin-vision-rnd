# Independent frozen-v2 metric and evidence review

**PASS WITH SCOPE CAVEATS — 2026-09-10. No P1/P2 scientific or metric defect found in the audited outputs.**

The independent auditor used NumPy, standard-library hashing and immutable Git evidence. It imported no experiment metric/model routines and performed no fitting, tuning or model inference. The reproducible calculation is preserved in [audit_cc_v2_metrics.py](../../../scripts/audit_cc_v2_metrics.py), with [rerun instructions](audit_readme.md); detailed checks, per-run legacy differences and SHA-256 receipts are in [independent_metric_review.json](independent_metric_review.json).

All **25 frozen configurations** are present: 17 CNN, six legacy, two statistics. All **594 domain/method records** were checked, including full recovery/reproduction summaries; fixed coverage 100/95/90/80/70/60%; score ordering with SHA-256 image-ID ties; complete selective curves and discrete AURC; source-calibrated threshold membership/count/error; and validity flags. Predictions and GT match their bound arrays, and IDs, capture groups and cameras match manifests exactly. Fresh data contain 384 distinct IDs, 128 per camera; source regression contains 462 images and Sony regression 30. All cached rows are valid.

| Independent check | Maximum absolute discrepancy |
| --- | ---: |
| Recovery/reproduction summaries | 3.553e-15 degrees |
| Fixed-coverage summaries | 7.106e-15 degrees |
| Selective risk curves, AURC, source-threshold masks | 0 |
| New-head source quantile thresholds | 0 |
| Aggregate means and sample standard deviations | 0 |
| Legacy predictions, GT, angular errors, curves and risk metrics | 0 |
| Legacy context confidence scores | 2.002133e-6 |
| Legacy source confidence thresholds | 1.984911e-7 |

The legacy comparison covers **60 original-head domain records** across all six frozen runs and both regression domains. Preserved config, heads and evaluation files agree with immutable commit `7637d6d` after CRLF/LF normalization, and checkpoints match its evidence manifest. Tiny context-only confidence differences are consistent with the reviewed float32 normalization versus JSON-reloaded float64 center/scale transition. They are below the explicit 1e-5 confidence tolerance; every ranking, acceptance count and reported angular-risk metric remains identical. No historical v1 result was replaced.

All 17 source snapshots, live source bytes, checkpoint/config/prediction chains, selector artifacts, four locked scripts, amendments and fresh-data hashes match. The source hash is `598a44f190d624db58fd4d2a60368f59bf41b1f3e8620c8c49c55d6e1e680398`. Final-lock SHA-256 is `3bb2219e4ccf2b4db43842e3e6666de701988a5af9bdf92f28faa4b3b4e29184`.

The final lock records 18:58:45.715505 UTC, after the latest selector file at 18:56:46.450153 and before the first evaluation file at 19:00:15.770579. Source train/val/risk/cal IDs and capture groups are disjoint. Recorded CNN checkpoint epochs minimize source-validation error. The auditor recomputed 285 CNN/statistics candidate OOF risk80/AURC pairs, verified held-out date folds and calibration data, and reproduced the final across-seed source-only head selection. Seeds 17/29/43 and representative seed 17 remain those in the lock. The earlier small-SoG choice and later source-only capacity amendment remain preserved; this is iterative source research, not a preregistered confirmatory study.

The scope remains bounded:

- Local timestamps and hash chains support the recorded order; they cannot establish absence of unrecorded earlier computation.
- Legacy-upgrade fit-time neural outputs/errors were not persisted. Their source IDs, selection rule and artifact bindings are checked; their OOF numerical errors were not independently regenerated because that would require new inference.
- No measured row exercises invalid-input refusal. Actual valid flags were independently checked; invalid branches rely on the earlier synthetic engineering review.
- Official source test retains the disclosed capture-date overlap; source and Sony30 are regression evidence. Fresh384 is a custom camera-balanced mirror subset with unverified true scene clusters and original-mirror identity. Repeated-reference sensitivity is useful but does not establish scene independence.
- This audit verifies the per-run metrics and aggregate arithmetic. It does not certify bootstrap population assumptions, full INTEL-TAU performance, skin/physical DeltaE, novelty or commercial rights.
