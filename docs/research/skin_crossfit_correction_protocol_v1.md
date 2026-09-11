# Frozen TRAIN-only encoder-exclusion correction experiment

Primary question: does learning corrections from excluded-person encoder errors
generalize better than fitting the core's in-sample residuals? Eliminate the
reference bank and512-dimensional learned contexts to isolate supervision and
avoid mixing arbitrary context coordinate systems of separately fitted cores.
Known stacked generalization; no architecture novelty claim.

## Data roles and exact prior reuse

Only original MSKCC TRAIN is loaded (24people/966 photographs, CC-BY native skin
instrument Lab). Existing skin_support_curve_v1 patient_roles fixes18people/
734 photos for support and6people/232 for internal evaluation. Original source
VALIDATION, CAL and TEST stay unread. This internal cohort was previously used
in exploratory research; it is NOT a fresh independent holdout or phone test.
Both cameras occur in support and evaluation; no unseen-camera claim this phase.

Reuse the three exact n18__baseline__s17/29/43 final.pt cores (929,297 each).
They were trained for930 fixed steps and were not selected by evaluation error.
Freeze their hashes and replay their internal-evaluation predictions. Existing
source histories did expose the evaluation outcomes; acknowledge that reuse.

Split the18 support people into three fixed camera-balanced inner folds:2SLR
and4iPod people per fold, sorted person IDs permuted per camera with RNG917031.
For each model seed17/29/43, refit three cores from scratch on the other12 people.
Core target scales use only those12. Entire query people/sites are absent from
their encoder training. Store exact membership and prediction arrays privately.
No outer-evaluation person participates in any refit, scale or head training.

## Core refit budget and correction arms

9 new inner cores, same original CaptureColor mixture recipe:30 rounds x31
updates, batch32 uniformly sampled images with replacement. RNGseed*1000+round,
AdamW0.001 wd0.01; cosine per round to0.00001. Standardized Lab MSE+0.1 modeCE.
Mode labels are auxiliary TRAIN targets only, never inference inputs. Fixed
final checkpoint; no evaluation during fitting, no new checkpoint selection.

Build three distinct correction-training prediction tables for every support
image, always converted back to native Lab before common head scaling:

- in_full: the existing18-person encoder, which saw that image/person.
- in_matched: the12-person encoder excluding the NEXT cyclic fold, which DID
  see that query person. One encoder per image, no averaging.
- out_person: the12-person encoder excluding that query's own entire fold.

in_matched and out_person use exactly the same three cores/counts/budgets,
removing the simple12-versus18 training-size confound in their comparison.
Different included people still change training composition; three seeds are
not independent participant cohorts. The out_person table is genuinely OOF
for the base encoder. The head is ordinarily trained on all18 support people;
head outputs on those18 are not themselves OOF accuracy.

Each table trains the same188,035-parameter correction MLP39->384->384->64->3,
SiLU hidden layers and tanh3 output. Inputs36 fixed image descriptors +3 native
base color estimates, normalized using common18-person color/target scales.
No independently fitted hidden context is shared. Last linear layer zeroed;
all initial corrections zero and identical initialization across arms.
9 heads:3 prediction tables x3 seeds.300 AdamW updates0.001 wd0.01, cosine per
step to0.00001, batch32 uniform support-image indices RNGseed. Same draws and
standardized Lab MSE objective, fixed final state, no evaluation-based tuning.

All three heads at evaluation use the SAME full18-person core prediction/input.
Thus in_matched/out_person share a12-to18 encoder-support-size deployment shift;
document this limitation instead of calling it perfect distribution matching.
Unchanged full18 core is the primary control. Total1,117,332 <=1,129,297 cap.
No reference memory, extra inference camera metadata, query reference or cloud.

## Metrics, checks and decision

Native skin CIEDE2000 mean/median/p95/>10/person/site means, each seed separately
and their averages, not ensembles. Common nearest-support standardized color36
distance for risk curves and100/95/90/80/70/60% coverage. This is uncalibrated
ranking, not an expected-error head or refusal guarantee. Retain all outcomes.

Compare TRAIN base error distributions for all three tables as a supervision
diagnostic, not accuracy on new people. Verify every encoder's scales/exclusions,
every table's routing, identical initial heads/draws, original core replays,
NumPy head predictions, scalar CIEDE2000 and a complete inner-core/head refit.
Report true parameter/checkpoint counts and allocated CUDA training memory.
No export optimization or ordinary-phone latency/accuracy claim.

[Wolpert,Stacked Generalization,1992](https://www.sciencedirect.com/science/article/pii/S0893608005800231)
already describes learning corrections from predictions on portions not used to
train the base generalizer. Publisher abstract checked2026-09-11; no author
code/weights/numbers adopted. Existing project OOF uncertainty and forward-noise
experiments are distinct from this matched correction-supervision test.

If out_person beats both controls and the unchanged core, freeze a separate
source camera-transfer follow-up. If it only beats in-sample heads but not the
core, retain the supervision result without claiming a better model. If it
fails, do not assume more passes/parameters solve it; revisit representation or
the need for a correction head. All prior negative results remain preserved.
