# Frozen privileged-reference offset diagnostic

Use the90 existing source prediction arrays from skin_sampling_transfer_v1,
after its complete audit. No original independent TEST/CAL access, no new image
model, no changed checkpoint, no new data and no production calibration.
This is a PRIVILEGED REFERENCE-CALIBRATION COMPARATOR only. It explicitly uses
other evaluation people's instrument references and is outside the single-image,
no-test-camera-calibration deployment constraint. Never mix its scores into the
original model's uncalibrated results or call it achieved phone accuracy.

For each held evaluation person, compute target-minus-predicted native-Lab
residuals from all OTHER people in that evaluation domain. Average residuals
within each site, then sites within each person, then the other people equally.
Apply that constant three-channel offset to the held person's predictions at
two fixed strengths0.5 and1.0. Keep both; do not select strength by error.
No clipping, affine scale, nonlinear adaptation or fitting to the held person's
labels. Mixed known domain uses five reference people per held person; each
single-camera known/unseen domain uses two.90 arrays give324 exclusion folds
and180 corrected output arrays. Baseline0 strength remains unchanged input.

Score genuine native Lab using scalar-verified CIEDE2000, mean/median/p90/p95,
patient/site means, tails>5/10, all fixed100/95/90/80/70/60% coverages and full
curves using unchanged original input-novelty ordering. Ranking is uncalibrated.
Report original and privileged rows separately. Preserve input hashes, all
outputs privately, aggregate only. No per-person identifiers in Git.

Hypothesis: shared offset explains a substantial part of transfer errors.
Failure: residuals vary by true color/person/image, so correction estimated on
other people over-corrects the held person. This is not an oracle optimum and
does not identify a pure camera effect; camera/participant/capture composition
remain confounded. Either outcome guides a different mechanism search, not a
claim of novel offset calibration or physically identifiable arbitrary color.
