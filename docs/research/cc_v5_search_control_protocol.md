# V5 equal-query search control, pre-execution specification

Question: does recentering subsequent correction candidates improve actual
error compared with evaluating the same multiscale grid around the original
point? This is a post-training development ablation, not a new independent test.

Use frozen V5 best checkpoints for all five trained critic arms and every
completed seed17/29/43. Exclude point-only because its critic was not trained.
Do not choose a checkpoint again using this control. Source119 reused validation
images only. Same encoder cache, risk function, image resolution and weights.

Adaptive search is the unchanged model.select at1/2/4 stages. Nonadaptive control
concatenates the same5x5 grids with radii .24/.06/.03/.015, all centered on the
original point. Include the original point as the26th query after stage1, as in
the adaptive implementation. Pick the lowest predicted-risk candidate across
the union. Exact evaluated candidate counts25/51/103, including repeated centers
and any clamp-induced duplicates for both policies. No GT enters proposals or
selection. Both use first-index argmin ties and neutral output on invalid input.

One-stage policies must match. Subsequent policies differ only by recentering
and the consequent covered region/resolution; this is not an exhaustive search
over all possible nonadaptive grids. The union is evaluated in one query call;
matched candidate count is not a claim of identical measured latency. Do not
credit recurrence merely because the predicted minimum decreases.

Save paired predictions/actions/risks, true recovery/reproduction summaries,
full risk-coverage curves and80% risk for both policies; report per-seed outcomes.
CPU inference avoids interference with ongoing GPU training. Hash-bind inputs,
checkpoints, executable sources and this protocol. No optimization/export gate
is passed simply by a development result. Repeat on a truly independent domain
only under its separate frozen method lock.
