# Frozen unchanged stable-color correction on source camera-held-out protocols

Follow-up to audited skin_crossfit_correction_v1 internal mean6.1082 ->5.6560.
Preserve its failed OOF-superiority explanation and all positive/negative data.
No change to head architecture, optimization, input representation or arm choice.

## Data and evaluation roles

Original MSKCC CC-BY real photographs and native instrument Lab. TRAIN24people/
966images; source VALIDATION6/264, with person/site disjointness. Source validation
is extensively reused; these are exploratory results, not independent accuracy.
Independent TEST/CAL remain closed. No new data, weights, cloud or publication.

Three protocols: mixed TRAIN24 ->known VALIDATION6; SLR TRAIN8/323 ->known3/132
SLR andunseen3/132iPod; iPod TRAIN16/643 ->known3/132iPod andunseen3/132SLR.
Unseen camera is absent from every fit and scale in its protocol. Person and
capture composition remain confounded with camera. Clinical SLR/iPod images
do not validate ordinary facial smartphone selfies.

Reuse exact skin_sampling_transfer_v1 {protocol}__image__s17/29/43 final.pt
cores:930 steps, no validation checkpoint selection,929,297 parameters. Freeze
all hashes and replay both known/unseen original prediction arrays exactly.
Any older80-epoch recipe is a historical comparator, not a newly matched fit.

## Inner cores and fixed correction arms

Four inner folds assign entire people; each holds2SLR/4iPod in mixed,2SLR in
SLR-only or4iPod in iPod-only. Sort unique IDs within each camera and permute
with RNG917031 for SLR,917032 foriPod. Corresponding people have identical fold
assignments in mixed/single-camera protocols. Inner train counts18mixed,6SLR,
12iPod. Query people and every site are absent from their OOF encoder/scales.

36 inner fits =3protocols x3seeds x4folds. Reuse the frozen prior fit_core:
930 updates,30 rounds x31 batches32, image-uniform sampling, AdamW0.001 wd0.01,
cosine per round to0.00001, standardized Lab MSE+0.1 mode auxiliary CE. No
camera/mode input at inference; fixed final checkpoint, no validation selection.

Three unchanged heads: in_full uses original full-bank core training predictions;
in_matched uses next cyclic inner core that DID see the query person; out_person
uses inner core that excluded the entire query person. One core per training
query, no prediction averaging. Convert all base estimates to native Lab, then
normalize them with full TRAIN-bank target scales. Head descriptors/scales fit
only full TRAIN bank. No learned hidden coordinates from different cores mix.

27 heads =3protocols x3seeds x3arms. Same stable39-input head:36 color descriptors
+3 base Lab,39->384->384->64->3,188,035parameters, zero last linear output.
Same300 AdamW updates, batch32 image-uniform RNGseed,0.001 wd0.01, cosine per
step to0.00001, standardized Lab MSE. Identical initial head/draws within each
protocol/seed. Final checkpoint only. All9 original cores stay unchanged.
Every head evaluates using its full-bank core. The3/4-to-full training-support
shift is shared by in_matched/out_person and explicitly remains a limitation.
Total deployment1,117,332 <=user cap1,129,297;78 scale scalars, no reference bank.

## Metrics, verification and decision

Report all known/unseen arms/seeds: mean/median/p95/>10 and person/site means of
native skin CIEDE2000. Aggregate seed scores, never claim a prediction ensemble.
Common nearest TRAIN color36 standardized-distance order per protocol, fixed
100/95/90/80/70/60% plus full curves. This is an uncalibrated shared ranking,
not expected-error calibration or confidence-head superiority.

Independent audits: checkpoint hashes, camera/person/scaling exclusions, all
inner-core/table replays, NumPy head forward, scalar DeltaE00 and coverage;
complete seed17 fold0 refit for every protocol, plus selected head refits.
Measure training allocations and file sizes; only benchmark inference/export
after assessing the evidence, and clearly separate prepared-feature timing.

Primary comparison: leading unchanged in_matched versus no-head core and other
head controls in BOTH unseen directions. Retain all outcomes even if it loses.
If a different head wins, record it; do not switch winners per camera and call
that camera-blind. No configuration tuning on unseen results. Compare stronger
historical forward4.8328/reverse4.9736 honestly, with recipe mismatch disclosed.

Known [stacked generalization](https://www.sciencedirect.com/science/article/pii/S0893608005800231)
is prior art; fresh source review recorded in the preceding phase. This follow-up
adopts no new external method/code and claims no stacking novelty. If transfer
fails, retain the internal gain but revisit representation rather than launch
an unfocused larger-head or repeated-correction sweep. Independent phone/color
accuracy and a technically distinct superiority claim remain unvalidated.
