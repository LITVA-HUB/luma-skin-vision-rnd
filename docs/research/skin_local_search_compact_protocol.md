# Secondary frozen-prefix compression study

Status: FROZEN before compact prefix fits and before every outer evaluation, 2026-09-13.

This secondary ablation was conceived after the primary local-search v1 inner results and
64-atom final weights were available, but before any primary or compact outer evaluation.
It is therefore adaptive exploratory work, not a preregistered or independent confirmation.
The user explicitly resumed local RTX 4060 research. No legacy validation, calibration or
test role is permitted.

## Question and fixed source

Can the already ordered 64-atom random and error-guided RBF models retain their inner-person
accuracy with 8, 16 or 32 atoms and a proportionally smaller payload? The study reads only the
hash-bound original `skin_mskcc_pixels_v1/train.npz` and completed
`experiments/runs/skin_local_search_v1` artifacts. It reuses the same mixed, SLR-to-iPod and
iPod-to-SLR roles, the same three person-held-out inner folds and seeds 17, 29 and 43.

For each protocol and RBF family, alpha is exactly the value already selected by the primary
64-atom v1 inner search. There is no new alpha grid. Each inner prefix uses the first K saved
centers and widths from the matching v1 protocol/family/selected-alpha/seed/fold model. The
center bank, width candidates and error-guided or random order are never regenerated. Only the
weighted analytic ridge coefficients are refit on that fold's current fitting rows.

The final K=8/16/32 models likewise take prefixes from the matching v1 selected-alpha final
64-atom model and refit coefficients on the complete outer-fit role. K=64 is the unchanged,
hash-verified v1 model and serves as the reference. Source scalers and target normalization are
preserved from each v1 model. Fit weights remain equal-person, equal-site-within-person and
equal-image-within-site. The bias is unpenalized and all RBF coefficients use the fixed v1 alpha.

## Selection and evaluation boundary

For every protocol/family/K, concatenate person-held-out predictions from the same three inner
folds and compute native-Lab CIEDE2000. Average person-balanced means across the three model
seeds; seeds are optimization repetitions, not independent people. Save the complete descriptive
curve for K in {8,16,32,64}. Also freeze the lowest-inner-error K, breaking exact ties toward
the smaller K. This K choice is secondary and adaptive because the primary inner results were
already known. K must never be selected from an outer result.

All K choices, final prefix weights, source hashes and the unchanged K=64 references must be
persisted before any outer prediction. Fit and evaluate are separate CLI stages. Evaluation
requires an explicit acknowledgement flag and scores all saved K values so losses are retained;
the inner-selected K is marked. The original outer roles are already historically reused and
overlap across protocols, so later values remain exploratory and are not new phone-face evidence.

## Measurements and accounting

Inner and later outer summaries use person-balanced mean as primary, plus image mean, site-then-
person mean, median, p90, fractions above DeltaE00 5 and 10, and camera strata where applicable.
Row predictions and weights stay in gitignored experiment artifacts; tracked reporting contains
aggregates only.

Report numeric scalars, numeric bytes and serialized NPZ bytes for every prefix. Centers and
scalers count even though they are frozen/data-derived. Measure the actual analytic prefix-refit
wall time separately. A prefix depends on the prior 64-atom search, so each receipt records the
matching v1 source-search time plus prefix-refit time. Aggregate inherited search cost counts each
unique v1 source model once; summing the per-prefix charged values would deliberately count a
shared source search multiple times. These are search/refit timings, not neural training time,
end-to-end capture latency or evidence of lower energy use.

This experiment tests compression of a known RBF/greedy ridge family. Neither prefix reuse,
greedy ordering nor an accuracy/storage tradeoff is by itself an algorithmic novelty claim.

## Post-fit bookkeeping amendment before compact outer evaluation

After all compact models and K choices were frozen, review found that the planned evaluation
code would have changed `frozen_prefixes.json` only to mark evaluation completion. Before any
compact outer row was opened, this was corrected: the frozen manifest remains byte-for-byte
immutable and completion is written to a separate `evaluation_status.json`. The fit-time source
lock is archived, and a machine-readable amendment binds the old and new source hashes and the
unchanged frozen-manifest hash. No compact model, center order, coefficient, alpha, K choice or
metric changed. Primary v1 outer evaluation had already run independently by amendment time;
this bookkeeping correction did not read its arrays or alter its evidence.
