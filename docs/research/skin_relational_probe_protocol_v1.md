# Frozen TRAIN relational compatibility falsifier

Primary endpoint remains genuine instrument-referenced skin Lab / DeltaE00.
This phase audits whether comparative features suppress acquisition variation
while retaining color distinctions. It is not a proposed deployed model or an
independent accuracy milestone. Original MSKCC CC-BY, original TRAIN only;
source VALIDATION, CAL, TEST and other held-out datasets are not loaded.

## Pre-outcome support inventory

24 people,966 images,248 sites;1421 same-site unordered capture pairs. No
same-site cross-camera pairs. Every positive pair has a wrong-site comparator
from the same person with the second image's mode and image_type. Site labels
denote anatomical reference sites, not registered per-pixel correspondence.

Same-site targets are identical means of native instrument readings, so their
reference distance is zero by construction, not a claim of identical pixels.
Closest eligible wrong-site reference distance ranges0.2803 to11.8487 DeltaE00;
44/209/607/1319 pairs have a candidate within0.5/1/2/5. No exact zero-distance
wrong-site match exists. Therefore explicitly report approximate color matching
and unmatched thresholds; never claim identical target-color distributions.

## Mathematical falsifier

For additive comparisons d(x,a)=f(x)-f(a), uniform reference anchoring gives
f(x)+mean(y(a)-f(a)), merely an absolute predictor plus a constant offset.
On a complete antisymmetric graph with every triangle cycle summing to zero,
row means reconstruct the potential up to a constant. Test both identities,
plus a pure cyclic counterexample. These are algebra checks, not skin accuracy.
Graph projection, relation learning and deep kernels are established prior art.

## Fixed pair/control protocol

Enumerate sites in sorted order and image indices ascending within each site;
use every unordered pair(a,b). Candidates c must share a's person, differ in
site, and match b's mode and image_type. Select one uniform candidate with RNG
69117, and one with minimum genuine reference DeltaE00 (tie: lowest index).
Keep both arms. The nearest-color arm is privileged diagnostic matching, not an
inference rule. Record repeated negative-pair counts and all reference distances.
Threshold strata0.5/1/2/5 and all are fixed; report unavailable strata.
Also enumerate all within-person, between-site unordered pairs and compute
Spearman association of representation distance with actual native DeltaE00.
Use aggregate and equal-person summaries; pair counts do not equal independent
people. No confidence-significance claim from many dependent pair comparisons.

## Fixed representations and scope

1. RGB median3, raw Euclidean RMS distance.
2. Existing color36 statistics, excluded-person standardized RMS distance.
3. Mean/max/population-std of64 patch tokens,54 features, same standardization.
4-5. Frozen512D learned context from mixed image and person_color seed17 final
   models in skin_sampling_transfer_v1. Cast copies to FP64 for reproducibility.
   Standardize features using other people's features, but the encoders were
   trained on all TRAIN people: these arms are DESCRIPTIVE TRAIN ONLY, never
   excluded-person learned representation performance.
6-7. Corresponding frozen model native-Lab prediction distances (DeltaE00), same
   descriptive limitation. Do not turn these into held-out skin accuracy.
8-9. Leave-one-person-out linear ridge native-Lab estimates from median3/color36.
   Fit24 separate folds per representation: inputs standardized using23 people,
   target centered using23 people, solve (X'X+I)B=X'(Y-center). Fixed alpha1,
   no validation, no hyperparameter selection, no outside weights. Both images
   in every evaluated pair belong to the same excluded person. Count48 fits.
   Distances are DeltaE00 between predictions. Also report direct per-image
   native-Lab error, labelled exploratory TRAIN cross-validation, not confirmation.
10. Other-people mean native Lab, constant within each excluded-person fold.
   Pair distance zero; tie-aware preference must equal0.5. This collapse control
   cannot count as successful nuisance invariance.

For arbitrary-feature arms, RMS distances are not DeltaE00. For color-estimate
arms use real CIEDE2000 distance between two estimated Lab values; these pair
distances are still not absolute measurement error. Actual per-image error is
separately labelled and limited to the48-fold ridge and mean controls.

## Metrics, integrity and decision

Paired preference counts d(a,b)<d(a,c) as1 and ties as0.5. A high preference can
reflect texture/site recognition rather than color accuracy. Report color-hard
controls and between-site true-color association together. A constant feature
has undefined Spearman correlation, reported null, never converted to success.

Freeze scripts/tests/protocol and original cache/checkpoint hashes before run.
Per-image indices, descriptors, references and fold weights remain ignored in
experiments. Commit only aggregate reports/audits. Independently check candidate
enumeration, scalar DeltaE00, exclusion by perturbed held targets, augmented
least-squares ridge replay, context replay and rank/preference arithmetic.

If no useful joint invariance/color-preservation signal survives, reject this
representation as support for a relational network. A promising descriptive
context signal requires a genuinely excluded-person refit before pair learning.
No camera-causal conclusion follows from this dataset's confounded people.
