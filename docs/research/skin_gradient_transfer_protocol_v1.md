# Frozen TRAIN-only person-gradient diagnostic

Primary product endpoint remains native instrument skin Lab / CIEDE2000. This
phase is a mechanism falsifier, not another independent accuracy result or fit.
Original TRAIN only: mixed24/966, SLR8/323, iPod16/643 people/images. No source
VALIDATION, CAL, TEST, ISSA or UMINHO held-out references are loaded. Original
MSKCC CC-BY data; no new outside data, code or weights.

## Fixed models and objectives

Six final seed17 checkpoints from skin_sampling_transfer_v1: image and
person_color samplers, each mixed/from_SLR/from_ipod. All929,297 parameters.
No model or epoch selection. Checkpoints remain unchanged. Sampler effects are
properties of those already fitted models; this diagnostic uses the SAME
image-uniform objective for all partitions and both trained sampler arms.

Color objective is mean squared error over3 standardized native-Lab coordinates.
Mode objective is four-way cross entropy. Analyze color and color+0.1*modeCE.
The existing target mean/std and standardized TRAIN targets use the original
FP32 recipe; the fixed model and inputs/targets are then cast to FP64 for
derivative/finite-difference diagnostics. FP64 is not deployment precision,
new training, or bit-identical FP32 inference. True native-Lab references are
retained in FP64 for CIEDE2000, which is separately measured after interventions.

## Controls and identifiability

Groups are sorted unique TRAIN people, or that per-image integer group array
permuted with RNG seeds51871/51872/51873. Exact image counts in every group and
the underlying images/targets are preserved. No references are relabeled.
Within-group losses are image means. Pool with n_group/N, so the pooled loss
and gradient cannot change merely from grouping. Never compare a site-balanced
true objective with an image-balanced random objective.

Record color and mode gradient Gram blocks, cross terms, pooled gradients and
norms, equal-pair off-diagonal cosine/negative fractions. Equal-pair summaries
describe conflict; they are not changes to the weighted training objective.
Near a pooled stationary point, opposing group gradients are mathematically
expected. Negative cosine is NOT proof of harmful shortcuts, camera dependence
or a reason to erase legitimate skin-color variation. Random-group controls
are necessary. Camera/person/color/capture effects remain confounded.

## Finite interventions fixed before observations

For every partition and both objectives, select group indices0,K//2,K-1,
irrespective of measured gradients. Take normalized full-parameter negative
gradient steps of absolute L2 lengths0.0001 and0.001. This is gradient descent
geometry, not AdamW replay or a proposed training learning rate. Independently
start all steps from the frozen checkpoint; restore every tensor exactly even
on exceptions. Do not write altered weights. Record all per-image native-Lab
predictions, MSE, CE and DeltaE00, all group changes, and first-order predictions
(-step*g_source dot g_target / ||g_source||). Check actual signs and remainder;
do not demand closeness where changes approach roundoff or cross nonsmooth
amax boundaries. No step-size selection from observed results.

There are6 models,24 partitions,48 objective cases and288 transient steps.
This is TRAIN loss/skin-error diagnostics, never held-out improved accuracy.

## Audit and decision

Freeze code/tests/protocol and six checkpoint hashes before running. Persist
only aggregate numeric outputs in Git; per-image results/group assignments stay
private under experiments. Independently check pooled objective invariance,
gradient Gram summaries, finite-loss arithmetic and scalar CIEDE2000. Recompute
true and first-shuffle gradient blocks using a different batch reduction, and
replay representative finite interventions. Verify exact restored states and
unchanged original checkpoint hashes.

Proceed to a controlled training experiment only after interpreting true versus
shuffled groups and color versus auxiliary gradients, plus actual native skin
error. A useful hypothesis must retain a matched ordinary model and test both
source transfer directions. MLDG, Fish and Fishr are existing research mechanisms,
not inventions of this project. No inference camera identity will be introduced.
